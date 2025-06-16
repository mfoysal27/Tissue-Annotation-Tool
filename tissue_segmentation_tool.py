import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import re
import sys
import traceback
from pathlib import Path
from collections import deque
import time

# Import PyTorch modules
import torch
import torch.nn as nn
from torchvision import transforms

# Import for debugging
import logging
logging.basicConfig(filename='nd2_loading.log', level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TissueSegmentationTool')

# Initialize variables
has_nd2reader = False
has_nd2 = False

# Proper ND2 file handling with fallbacks
try:
    # First try nd2reader which is more stable
    from nd2reader import ND2Reader
    has_nd2reader = True
    logger.info("Successfully imported nd2reader")
except ImportError:
    has_nd2reader = False
    logger.warning("Failed to import nd2reader")
    
try:
    # Fall back to nd2 package
    import nd2
    has_nd2 = True
    logger.info("Successfully imported nd2")
except ImportError:
    has_nd2 = False
    logger.warning("Failed to import nd2")

if not (has_nd2reader or has_nd2):
    logger.error("No ND2 file support available. Neither nd2reader nor nd2 could be imported.")
    print("Warning: Neither nd2reader nor nd2 package found. ND2 file support will be disabled.")

class DoubleConv(nn.Module):
    """Double Convolution and BN and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downsampling path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 512)
        )

        # Upsampling path
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv(512 + 256, 256)  # 512 from x4 + 256 from up1 = 768
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2_conv = DoubleConv(256 + 128, 128)  # 256 from x3 + 128 from up2 = 384
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3_conv = DoubleConv(128 + 64, 64)  # 128 from x2 + 64 from up3 = 192
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up4_conv = DoubleConv(64 + 64, 64)  # 64 from x1 + 64 from up4 = 128
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def center_crop(self, enc_feat, dec_feat):
        """Crop enc_feat to the size of dec_feat (center crop)"""
        _, _, h, w = dec_feat.shape
        enc_h, enc_w = enc_feat.shape[2], enc_feat.shape[3]
        crop_h = (enc_h - h) // 2
        crop_w = (enc_w - w) // 2
        return enc_feat[:, :, crop_h:crop_h + h, crop_w:crop_w + w]

    def forward(self, x):
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Upsampling with skip connections and cropping
        x = self.up1(x5)
        x4_crop = self.center_crop(x4, x)
        x = self.up1_conv(torch.cat([x4_crop, x], dim=1))
        
        x = self.up2(x)
        x3_crop = self.center_crop(x3, x)
        x = self.up2_conv(torch.cat([x3_crop, x], dim=1))
        
        x = self.up3(x)
        x2_crop = self.center_crop(x2, x)
        x = self.up3_conv(torch.cat([x2_crop, x], dim=1))
        
        x = self.up4(x)
        x1_crop = self.center_crop(x1, x)
        x = self.up4_conv(torch.cat([x1_crop, x], dim=1))
        
        x = self.outc(x)
        return x

class TissueSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Tissue Segmentation Tool")
        self.root.geometry("1400x900")  # Increased from 1200x800
        
        # Initialize ML-related variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Fixed size for the model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.current_image = None
        self.original_images = []
        self.segmentation_masks = []
        self.current_image_index = 0
        self.image_paths = []
        self.current_directory = None
        self.current_file_idx = 0
        self.directory_files = []
        self.segments = []
        self.segment_hierarchy = {}
        self.segment_colors = {}
        self.brush_size = 10
        self.current_segment = 0
        self.current_color = None
        self.drawing = False
        self.prev_x = None
        self.prev_y = None
        self.history = []
        self.history_index = -1
        
        # Tool selection
        self.active_tool = "brush"  # "brush" or "fill"
        
        # Add zoom variables
        self.zoom_factor = 1.0
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0
        self.dragging = False
        
        # Add opacity control for annotation overlay
        self.annotation_opacity = 0.7  # Default 70% opacity
        
        # Auto-segmentation queue for folder processing
        self.auto_segment_queue = []
        self.auto_segment_model_ready = False
        
        # Initialize the hardcoded segments and colors
        self.initialize_segments()
        
        self.create_landing_page()
    
    def initialize_segments(self):
        """Initialize hardcoded segments and their colors"""
        # Main features
        self.segments = []
        self.segment_hierarchy = {}
        self.segment_colors = {}
        
        # 1. Villi and its subfeatures
        villi = "Villi"
        self.segments.append(villi)
        self.segment_hierarchy[villi] = []
        self.segment_colors[villi] = (255, 192, 203, 255)  # Pink
        
        villi_subs = [
            ("Epithelium", (0, 100, 0, 255)),  # Dark green
            ("Basement membrane", (255, 0, 255, 255)),  # Bright magenta
            ("Lamina propria", (184, 134, 11, 255)),  # Dark yellow
            ("Central lacteal or blood vessels", (0, 191, 255, 255)),  # Bright blue
        ]
        
        for name, color in villi_subs:
            full_name = f"{villi} - {name}"
            self.segments.append(full_name)
            self.segment_hierarchy[villi].append(full_name)
            self.segment_colors[full_name] = color
        
        # 2. Gland and its subfeatures
        gland = "Gland"
        self.segments.append(gland)
        self.segment_hierarchy[gland] = []
        self.segment_colors[gland] = (255, 255, 255, 255)  # White
        
        gland_subs = [
            ("Epithelium", (0, 255, 0, 255)),  # Bright green
            ("Basement membrane", (139, 0, 0, 255)),  # Dark red
            ("Lamina propria", (255, 255, 0, 255)),  # Bright yellow
        ]
        
        for name, color in gland_subs:
            full_name = f"{gland} - {name}"
            self.segments.append(full_name)
            self.segment_hierarchy[gland].append(full_name)
            self.segment_colors[full_name] = color
        
        # 3. Individual Submucosa features
        submucosa_features = [
            ("Submucosa Ganglion", (64, 64, 64, 255)),  # Dark gray
            ("Submucosa Fiber tract", (192, 192, 192, 255)),  # Bright gray
            ("Submucosa Submucosal blood Vessel", (128, 0, 128, 255)),  # Purple
            ("Submucosa Interstitial tissue", (0, 0, 128, 255)),  # Navy Blue
        ]
        
        for name, color in submucosa_features:
            self.segments.append(name)
            self.segment_hierarchy[name] = []  # No subfeatures
            self.segment_colors[name] = color
        
        # 4. Circular muscle
        circular_muscle = "Circular muscle"
        self.segments.append(circular_muscle)
        self.segment_hierarchy[circular_muscle] = []
        self.segment_colors[circular_muscle] = (0, 128, 128, 255)  # Teal
        
        # 5. Individual Myenteric features
        myenteric_features = [
            ("Myenteric Ganglion", (205, 133, 63, 255)),  # Bright brown
            ("Myenteric Fiber tract", (101, 67, 33, 255)),  # Dark brown
        ]
        
        for name, color in myenteric_features:
            self.segments.append(name)
            self.segment_hierarchy[name] = []  # No subfeatures
            self.segment_colors[name] = color
        
        # 6. Longitudinal muscle
        longitudinal_muscle = "Longitudinal muscle"
        self.segments.append(longitudinal_muscle)
        self.segment_hierarchy[longitudinal_muscle] = []
        self.segment_colors[longitudinal_muscle] = (255, 165, 0, 255)  # Orange
        
        # 7. Individual Others features
        others_features = [
            ("Others Out of Tissue", (0, 0, 0, 255)),  # Black
            ("Others Fat", (50, 205, 50, 255)),  # Lime Green
            ("Others Lymphoid tissue", (196, 162, 196, 255)),  # Light wisteria
            ("Others Vessel", (75, 0, 130, 255)),  # Indigo
            ("Others Mesenteric tissue", (255, 20, 147, 255)),  # Deep Pink
        ]
        
        for name, color in others_features:
            self.segments.append(name)
            self.segment_hierarchy[name] = []  # No subfeatures
            self.segment_colors[name] = color
    
    def create_landing_page(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create landing page
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        label = ttk.Label(frame, text="Tissue Segmentation Tool", font=("Arial", 24))
        label.pack(pady=20)
        
        instruction = ttk.Label(frame, text="Please load an image to start segmentation", font=("Arial", 14))
        instruction.pack(pady=10)
        
        load_button = ttk.Button(frame, text="Load Image", command=self.load_image)
        load_button.pack(pady=20)
        
        supported_formats = "Supported formats: ND2, JPG, PNG, TIF"
        format_label = ttk.Label(frame, text=supported_formats, font=("Arial", 12))
        format_label.pack(pady=10)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.nd2 *.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Store directory and find all image files in that directory
        self.current_directory = os.path.dirname(file_path)
        self.image_paths = [file_path]
        self.directory_files = self.get_image_files_in_directory(self.current_directory)
        
        # Find index of current file in directory
        self.current_file_idx = self.directory_files.index(file_path) if file_path in self.directory_files else 0
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Process based on file type
        loading_successful = False
        
        try:
            if file_ext == '.nd2':
                if not (has_nd2reader or has_nd2):
                    messagebox.showerror("Error", "ND2 file support is not available. Please install nd2reader or nd2 package using pip.")
                    logger.error("Attempted to load ND2 file without required packages")
                    return
                
                logger.info(f"Starting ND2 file loading: {file_path}")
                self.process_nd2_file(file_path)
                loading_successful = len(self.original_images) > 0
                
                if not loading_successful:
                    # Show additional troubleshooting options
                    if messagebox.askyesno("ND2 Loading Failed", 
                                           "Failed to load ND2 file. Would you like to see troubleshooting options?"):
                        self.show_nd2_troubleshooting()
                        return
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                self.process_regular_image(file_path)
                loading_successful = len(self.original_images) > 0
            else:
                messagebox.showerror("Error", f"Unsupported file format: {file_ext}")
                return
        except Exception as e:
            logger.error(f"Error during image loading: {str(e)}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            return
        
        if not loading_successful:
            messagebox.showerror("Error", "Failed to load images")
            return
        
        self.create_annotation_window()
    
    def get_image_files_in_directory(self, directory):
        """Find all image files in the directory"""
        image_extensions = ['.nd2', '.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_files = []
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory, file))
        
        # Sort files by name
        image_files.sort()
        return image_files
    
    def process_nd2_file(self, file_path):
        try:
            logger.info(f"Starting to process ND2 file: {file_path}")
            self.original_images = []
            error_details = []
            
            # Create a progress window for analysis
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Analyzing ND2 File")
            analysis_window.geometry("400x150")
            analysis_window.transient(self.root)
            analysis_window.grab_set()
            
            progress_label = ttk.Label(analysis_window, text="Analyzing ND2 file structure...")
            progress_label.pack(pady=10)
            
            status_label = ttk.Label(analysis_window, text="", wraplength=380)
            status_label.pack(pady=5)
            
            progress_bar = ttk.Progressbar(analysis_window, orient="horizontal", length=300, mode="indeterminate")
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Update the progress window
            analysis_window.update()
            
            # First analyze file to determine number of frames
            total_frames = 0
            frame_dimension = None
            frame_info = {}
            
            # Try to get frame count with nd2 package
            if has_nd2:
                try:
                    status_label.config(text="Analyzing ND2 file dimensions...")
                    analysis_window.update()
                    
                    with nd2.ND2File(file_path) as f:
                        logger.info(f"ND2 file opened with nd2 package. Available sizes: {f.sizes}")
                        
                        # Find likely Z-stack or other dimension for frames
                        z_dimensions = []
                        for dim in ['z', 'v', 'Z', 'V', 't', 'T']:
                            if dim in f.sizes and f.sizes[dim] > 1:
                                z_dimensions.append((dim, f.sizes[dim]))
                        
                        # If no Z dimensions found, check other dimensions
                        if not z_dimensions:
                            for dim, size in f.sizes.items():
                                if size > 1 and dim.lower() not in ['x', 'y', 'c']:
                                    z_dimensions.append((dim, size))
                        
                        # If still no dimensions, try using channels
                        if not z_dimensions and 'c' in f.sizes and f.sizes['c'] > 1:
                            z_dimensions.append(('c', f.sizes['c']))
                        
                        if z_dimensions:
                            # Pick the dimension with the most slices
                            frame_dimension, total_frames = max(z_dimensions, key=lambda x: x[1])
                            frame_info = {"package": "nd2", "dimension": frame_dimension, "count": total_frames}
                            logger.info(f"Found {total_frames} frames in dimension {frame_dimension}")
                except Exception as e:
                    logger.error(f"Error analyzing ND2 with nd2 package: {str(e)}")
            
            # If nd2 package failed, try nd2reader
            if total_frames == 0 and has_nd2reader:
                try:
                    status_label.config(text="Analyzing with nd2reader...")
                    analysis_window.update()
                    
                    with ND2Reader(file_path) as images:
                        total_frames = len(images)
                        frame_info = {"package": "nd2reader", "count": total_frames}
                        logger.info(f"Found {total_frames} frames with nd2reader")
                except Exception as e:
                    logger.error(f"Error analyzing ND2 with nd2reader: {str(e)}")
            
            # Close analysis window
            try:
                analysis_window.destroy()
            except:
                pass
            
            # If no frames found, show error
            if total_frames == 0:
                messagebox.showerror("Error", "Could not determine frame count in ND2 file")
                return
            
            # If we found frames, ask the user for start/end frame
            start_frame = 0
            end_frame = total_frames - 1 if total_frames > 0 else 0
            load_all = True
            
            # Create a dialog to get frame range and confirm loading
            frame_dialog = tk.Toplevel(self.root)
            frame_dialog.title("ND2 Frame Selection")
            frame_dialog.geometry("400x300")  # Increased height
            frame_dialog.transient(self.root)
            frame_dialog.grab_set()
            
            frame = ttk.Frame(frame_dialog, padding="20")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Frame info label
            info_text = f"ND2 file contains {total_frames} frames"
            if frame_dimension:
                info_text += f" in dimension '{frame_dimension}'"
            info_label = ttk.Label(frame, text=info_text, font=("Arial", 10, "bold"))
            info_label.pack(pady=10)
            
            # Load all frames option
            load_all_var = tk.BooleanVar(value=True)
            load_all_cb = ttk.Checkbutton(
                frame, 
                text="Load all frames",
                variable=load_all_var
            )
            load_all_cb.pack(pady=5, anchor=tk.W)
            
            # Frame range selection
            range_frame = ttk.LabelFrame(frame, text="Custom Frame Range")
            range_frame.pack(pady=10, fill=tk.X)
            
            range_grid = ttk.Frame(range_frame)
            range_grid.pack(pady=10, padx=10, fill=tk.X)
            
            ttk.Label(range_grid, text="Start Frame:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            ttk.Label(range_grid, text="End Frame:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            
            start_var = tk.StringVar(value="0")
            end_var = tk.StringVar(value=str(total_frames - 1))
            
            start_entry = ttk.Entry(range_grid, textvariable=start_var, width=10)
            start_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
            end_entry = ttk.Entry(range_grid, textvariable=end_var, width=10)
            end_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            
            # Function to enable/disable range inputs
            def toggle_range_inputs():
                state = "disabled" if load_all_var.get() else "normal"
                start_entry.config(state=state)
                end_entry.config(state=state)
                range_frame.config(text="Custom Frame Range (Disabled)" if load_all_var.get() else "Custom Frame Range")
            
            # Connect the toggle function to the checkbox
            load_all_cb.config(command=toggle_range_inputs)
            
            # Initially disable entries if "Load all" is checked
            toggle_range_inputs()
            
            # Move buttons to the top for better visibility
            button_frame = ttk.Frame(frame_dialog)
            button_frame.pack(side=tk.BOTTOM, pady=15, fill=tk.X)
            
            # Create the action buttons in the bottom frame
            # Create a special button that really stands out
            load_button = tk.Button(
                button_frame, 
                text="LOAD FRAMES", 
                bg="#007bff",
                fg="white",
                font=("Arial", 11, "bold"),
                relief=tk.RAISED,
                borderwidth=2,
                padx=15,
                pady=8,
                cursor="hand2"  # Hand cursor on hover
            )
            load_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
            
            cancel_button = ttk.Button(
                button_frame, 
                text="Cancel", 
                command=lambda: frame_dialog.destroy()
            )
            cancel_button.pack(side=tk.RIGHT, padx=10, pady=10)
            
            # Load button with validation
            def validate_and_load():
                nonlocal start_frame, end_frame, load_all
                
                # Set load_all flag
                load_all = load_all_var.get()
                
                # If loading all frames, use full range
                if load_all:
                    start_frame = 0
                    end_frame = total_frames - 1
                    frame_dialog.destroy()
                    return True
                
                # Otherwise validate custom range
                try:
                    s = int(start_var.get())
                    e = int(end_var.get())
                    
                    # Validate range
                    if s < 0 or s >= total_frames:
                        messagebox.showerror("Error", f"Start frame must be between 0 and {total_frames-1}")
                        return False
                    if e < 0 or e >= total_frames:
                        messagebox.showerror("Error", f"End frame must be between 0 and {total_frames-1}")
                        return False
                    if s > e:
                        messagebox.showerror("Error", "Start frame must be less than or equal to end frame")
                        return False
                    
                    # Set values and close
                    start_frame = s
                    end_frame = e
                    frame_dialog.destroy()
                    return True
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid frame numbers")
                    return False
            
            # Configure load button command
            load_button.config(command=validate_and_load)
            
            # Wait for dialog to close
            dialog_result = [False]  # Use a list to store result (mutable)
            
            # Update validate_and_load to set the result
            original_validate = validate_and_load
            def validate_wrapper():
                result = original_validate()
                dialog_result[0] = result
                return result
            
            # Replace the command
            load_button.config(command=validate_wrapper)
            
            # Ensure dialog is positioned properly
            frame_dialog.update_idletasks()
            screen_width = frame_dialog.winfo_screenwidth()
            screen_height = frame_dialog.winfo_screenheight()
            width = frame_dialog.winfo_width()
            height = frame_dialog.winfo_height()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            frame_dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            # Make dialog a fixed size to prevent resizing issues
            frame_dialog.resizable(False, False)
            
            # Wait for dialog to close
            self.root.wait_window(frame_dialog)
            
            # Check if dialog was successful
            if not dialog_result[0]:
                logger.info("Frame selection cancelled or unsuccessful")
                return
            
            # Start the loading process
            self.load_nd2_frames(file_path, frame_info, start_frame, end_frame)
            
        except Exception as e:
            logger.error(f"Unexpected error in process_nd2_file: {str(e)}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to process ND2 file: {str(e)}")
            
            # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0
    
    def load_nd2_frames(self, file_path, frame_info, start_frame, end_frame):
        """Load the selected frames from the ND2 file"""
        try:
            # Create progress window for loading
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Loading ND2 Frames")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Loading frames, please wait...")
            progress_label.pack(pady=10)
            
            status_label = ttk.Label(progress_window, text="", wraplength=380)
            status_label.pack(pady=5)
            
            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
            progress_bar.pack(pady=10)
            
            # Update the progress window
            progress_window.update()
            
            error_details = []
            success = False
            
            # Load frames based on selected method
            if frame_info.get("package") == "nd2":
                try:
                    status_label.config(text=f"Loading frames {start_frame} to {end_frame} with nd2 package...")
                    progress_window.update()
                    
                    dimension = frame_info.get("dimension")
                    
                    with nd2.ND2File(file_path) as f:
                        # Setup progress tracking
                        num_frames_to_load = end_frame - start_frame + 1
                        progress_bar['maximum'] = num_frames_to_load
                        progress_bar['value'] = 0
                        progress_window.update()
                        
                        for i in range(start_frame, end_frame + 1):
                            try:
                                # Update progress
                                frame_index = i - start_frame
                                progress_label.config(text=f"Loading frame {frame_index+1}/{num_frames_to_load}...")
                                progress_bar['value'] = frame_index + 1
                                progress_window.update()
                                
                                # Create the dimension keyword args
                                kwargs = {dimension.lower(): i}
                                
                                # Read the image for this slice
                                img_array = f.read_image(**kwargs)
                                
                                # Convert to PIL Image
                                img = self.process_nd2_array(img_array)
                                if img:
                                    self.original_images.append(img)
                                    logger.info(f"Successfully loaded frame {i}, size: {img.size}")
                                    success = True
                                else:
                                    logger.warning(f"Failed to convert frame {i} to image")
                            except Exception as e:
                                error_msg = f"Error loading frame {i}: {str(e)}"
                                logger.error(error_msg)
                                error_details.append(error_msg)
                                continue
                except Exception as e:
                    error_msg = f"nd2 package load error: {str(e)}"
                    logger.error(error_msg)
                    error_details.append(error_msg)
            
            # If nd2 package failed or wasn't used, try nd2reader
            if not success and frame_info.get("package") == "nd2reader":
                try:
                    status_label.config(text=f"Loading frames {start_frame} to {end_frame} with nd2reader...")
                    progress_window.update()
                    
                    with ND2Reader(file_path) as images:
                        # Setup progress tracking
                        num_frames_to_load = end_frame - start_frame + 1
                        progress_bar['maximum'] = num_frames_to_load
                        progress_bar['value'] = 0
                        progress_window.update()
                        
                        for i in range(start_frame, end_frame + 1):
                            try:
                                # Update progress
                                frame_index = i - start_frame
                                progress_label.config(text=f"Loading frame {frame_index+1}/{num_frames_to_load}...")
                                progress_bar['value'] = frame_index + 1
                                progress_window.update()
                                
                                # Get frame
                                img_array = images.get_frame(i)
                                
                                # Process frame
                                img = self.process_nd2_array(img_array)
                                if img:
                                    self.original_images.append(img)
                                    logger.info(f"Successfully loaded frame {i}, size: {img.size}")
                                    success = True
                                else:
                                    logger.warning(f"Failed to convert frame {i} to image")
                            except Exception as e:
                                error_msg = f"Error loading frame {i}: {str(e)}"
                                logger.error(error_msg)
                                error_details.append(error_msg)
                                continue
                except Exception as e:
                    error_msg = f"nd2reader error: {str(e)}"
                    logger.error(error_msg)
                    error_details.append(error_msg)
            
            # Try using a different approach if both methods failed
            if not success and has_nd2:
                try:
                    status_label.config(text="Trying alternative loading approach...")
                    progress_window.update()
                    
                    # Try loading using raw data approach
                    with nd2.ND2File(file_path) as f:
                        # Try different coordinate systems
                        for coordinate_system in ['zyx', 'vyx', 'cyx']:
                            try:
                                # Get raw data
                                status_label.config(text=f"Attempting {coordinate_system} coordinates...")
                                progress_window.update()
                                
                                raw_data = f.asarray()
                                logger.info(f"Raw data shape: {raw_data.shape}")
                                
                                # For 3D+ arrays, extract 2D slices
                                if len(raw_data.shape) >= 3:
                                    num_slices = min(raw_data.shape[0], end_frame + 1)
                                    start_idx = min(start_frame, num_slices - 1)
                                    end_idx = min(end_frame, num_slices - 1)
                                    
                                    # Update progress bar
                                    num_frames_to_load = end_idx - start_idx + 1
                                    progress_bar['maximum'] = num_frames_to_load
                                    progress_bar['value'] = 0
                                    progress_window.update()
                                    
                                    for i in range(start_idx, end_idx + 1):
                                        try:
                                            # Update progress
                                            frame_index = i - start_idx
                                            progress_label.config(text=f"Processing slice {frame_index+1}/{num_frames_to_load}...")
                                            progress_bar['value'] = frame_index + 1
                                            progress_window.update()
                                            
                                            # Get slice
                                            slice_data = raw_data[i]
                                            
                                            # Convert to image
                                            img = self.process_nd2_array(slice_data)
                                            if img:
                                                self.original_images.append(img)
                                                logger.info(f"Successfully loaded raw slice {i}, size: {img.size}")
                                                success = True
                                        except Exception as e:
                                            logger.error(f"Error processing raw slice {i}: {str(e)}")
                                            continue
                                else:
                                    # Single image
                                    img = self.process_nd2_array(raw_data)
                                    if img:
                                        self.original_images.append(img)
                                        logger.info(f"Successfully loaded single raw image, size: {img.size}")
                                        success = True
                                
                                # If we got images, break out of coordinate system loop
                                if self.original_images:
                                    break
                                    
                            except Exception as e:
                                logger.error(f"Error with {coordinate_system} coordinates: {str(e)}")
                                continue
                except Exception as e:
                    error_msg = f"Raw data approach error: {str(e)}"
                    logger.error(error_msg)
                    error_details.append(error_msg)
            
            # Close progress window
            try:
                progress_window.destroy()
            except:
                pass
            
            # Check if we got any images
            if not self.original_images:
                # Show error with details
                logger.error("Failed to load any slices from ND2 file")
                
                error_details_str = "\n".join(error_details)
                logger.error(f"Error details:\n{error_details_str}")
                
                messagebox.showerror("Error", "Failed to load any slices from the ND2 file. Check the log for details.")
                return
            
            # Initialize segmentation masks for successful images
            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0)) for img in self.original_images]
            self.current_image_index = 0
            
            # Show success message
            messagebox.showinfo("Success", f"Successfully loaded {len(self.original_images)} frames from ND2 file")
            
        except Exception as e:
            logger.error(f"Unexpected error in load_nd2_frames: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Show error message
            messagebox.showerror("Error", f"Failed to load ND2 frames: {str(e)}")
            
            # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0
    
    def process_regular_image(self, file_path):
        try:
            # Check if this is an auto-segmented original image
            base_name = os.path.basename(file_path)
            if base_name.startswith("auto_segmented_original_"):
                # This is an auto-segmented original image, try to load the corresponding mask
                mask_name = base_name.replace("auto_segmented_original_", "auto_segmented_mask_")
                mask_name = os.path.splitext(mask_name)[0] + ".png"  # Ensure .png extension
                mask_path = os.path.join(os.path.dirname(file_path), mask_name)
                
                # Load original image
                img = Image.open(file_path).convert('RGB')
                
                # Try to load corresponding mask
                if os.path.exists(mask_path):
                    try:
                        mask = Image.open(mask_path).convert('RGBA')
                        
                        # Check if mask has any non-transparent pixels
                        mask_array = np.array(mask)
                        non_transparent = np.sum(mask_array[:, :, 3] > 0)
                        
                        self.original_images = [img]
                        self.segmentation_masks = [mask]
                        self.current_image_index = 0
                        
                        # Update window title to indicate auto-segmented image
                        self.root.title(f"Tissue Segmentation Tool - Auto-Segmented: {base_name}")
                        
                        # Show user-friendly message only once per session or if significant annotations
                        if non_transparent > 100:  # Only show if substantial annotations
                            messagebox.showinfo("Auto-Segmented Image Loaded", 
                                              f"✓ Loaded auto-segmented image with {non_transparent} annotated pixels.\n"
                                              "You can modify the annotations using the paint tools and adjust opacity.")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load mask {mask_path}: {str(e)}")
                        # Don't show intrusive popup for loading issues
                        print(f"Warning: Could not load mask for auto-segmented image")
                        # Fall back to empty mask
                        pass
                else:
                    # Don't show intrusive popup for missing mask
                    print(f"Warning: Mask file missing for auto-segmented image")
            
            # Default behavior for regular images or if mask loading failed
            img = Image.open(file_path).convert('RGB')
            self.original_images = [img]
            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0))]
            self.current_image_index = 0
            
            # Update window title for regular images
            self.root.title(f"Tissue Segmentation Tool - {base_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")
    
    def create_annotation_window(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.geometry("1400x900")  # Increased from 1200x800
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame for canvas and navigation
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas_frame = ttk.Frame(left_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars for large/zoomed images
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg="white",
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set
        )
        
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Add zoom and pan bindings
        self.canvas.bind("<ButtonPress-3>", self.start_pan)  # Right button to pan
        self.canvas.bind("<B3-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-3>", self.stop_pan)
        self.canvas.bind("<MouseWheel>", self.zoom_with_wheel)  # Mouse wheel to zoom
        
        # Navigation frame for multiple images and files
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        # Add a "Load New Folder" button at the top
        folder_frame = ttk.Frame(nav_frame)
        folder_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        load_folder_button = ttk.Button(
            folder_frame, 
            text="Load New Folder", 
            command=self.load_new_folder,
            style="Action.TButton"
        )
        load_folder_button.pack(side=tk.LEFT, padx=5)
        
        current_dir_label = ttk.Label(
            folder_frame, 
            text=f"Current Folder: {os.path.basename(self.current_directory)}"
        )
        current_dir_label.pack(side=tk.LEFT, padx=20, expand=True)
        
        # File navigation buttons
        file_nav_frame = ttk.Frame(nav_frame)
        file_nav_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        prev_file_button = ttk.Button(file_nav_frame, text="Previous File", command=self.load_prev_file)
        prev_file_button.pack(side=tk.LEFT, padx=5)
        
        current_file_label = ttk.Label(
            file_nav_frame, 
            text=f"File: {os.path.basename(self.image_paths[0])}"
        )
        current_file_label.pack(side=tk.LEFT, padx=20, expand=True)
        
        # Add status indicator for auto-segmented images
        self.status_label = ttk.Label(
            file_nav_frame,
            text="",
            foreground="green",
            font=("Arial", 9, "bold")
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Update status based on current file
        base_name = os.path.basename(self.image_paths[0])
        if base_name.startswith("auto_segmented_original_"):
            self.status_label.config(text="✓ Auto-Segmented")
        else:
            self.status_label.config(text="")
        
        next_file_button = ttk.Button(file_nav_frame, text="Next File", command=self.load_next_file)
        next_file_button.pack(side=tk.RIGHT, padx=5)
        
        # Slice navigation for multi-slice images
        if len(self.original_images) > 1:
            slice_nav_frame = ttk.Frame(nav_frame)
            slice_nav_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
            
            prev_button = ttk.Button(slice_nav_frame, text="Previous Slice", command=self.previous_image)
            prev_button.pack(side=tk.LEFT, padx=5)
            
            self.slice_label = ttk.Label(slice_nav_frame, text=f"Slice: 1/{len(self.original_images)}")
            self.slice_label.pack(side=tk.LEFT, padx=20)
            
            next_button = ttk.Button(slice_nav_frame, text="Next Slice", command=self.next_image)
            next_button.pack(side=tk.RIGHT, padx=5)
        
        # Zoom controls frame
        zoom_frame = ttk.Frame(nav_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        zoom_in_button = ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom_in(None))
        zoom_in_button.pack(side=tk.LEFT, padx=5)
        
        zoom_out_button = ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom_out(None))
        zoom_out_button.pack(side=tk.LEFT, padx=5)
        
        zoom_reset_button = ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom)
        zoom_reset_button.pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = ttk.Label(zoom_frame, text=f"Zoom: 100%")
        self.zoom_label.pack(side=tk.RIGHT, padx=20)
        
        # Right frame for controls
        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        right_frame.pack_propagate(False)
        
        # Tool controls
        tool_frame = ttk.LabelFrame(right_frame, text="Tools")
        tool_frame.pack(fill=tk.X, pady=5)
        
        # Tool selection
        tool_selection_frame = ttk.Frame(tool_frame)
        tool_selection_frame.pack(fill=tk.X, pady=5)
        
        self.tool_var = tk.StringVar(value=self.active_tool)
        
        brush_tool_btn = ttk.Radiobutton(
            tool_selection_frame, 
            text="Brush Tool", 
            variable=self.tool_var, 
            value="brush",
            command=self.set_brush_tool
        )
        brush_tool_btn.pack(side=tk.LEFT, padx=5)
        
        fill_tool_btn = ttk.Radiobutton(
            tool_selection_frame, 
            text="Fill Tool", 
            variable=self.tool_var, 
            value="fill",
            command=self.set_fill_tool
        )
        fill_tool_btn.pack(side=tk.LEFT, padx=5)
        
        # Brush size control
        brush_frame = ttk.Frame(tool_frame)
        brush_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        
        self.brush_size_var = tk.IntVar(value=self.brush_size)
        brush_slider = ttk.Scale(
            brush_frame, 
            from_=1, 
            to=100,  # Increased to 100
            orient=tk.HORIZONTAL, 
            variable=self.brush_size_var,
            command=self.update_brush_size
        )
        brush_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Add a label showing the current brush size
        self.brush_size_label = ttk.Label(brush_frame, text=f"{self.brush_size}")
        self.brush_size_label.pack(side=tk.LEFT, padx=5)
        
        # Opacity control for annotation overlay
        opacity_frame = ttk.Frame(tool_frame)
        opacity_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(opacity_frame, text="Annotation Opacity:").pack(side=tk.LEFT, padx=5)
        
        self.opacity_var = tk.DoubleVar(value=self.annotation_opacity)
        opacity_slider = ttk.Scale(
            opacity_frame, 
            from_=0.0, 
            to=1.0,
            orient=tk.HORIZONTAL, 
            variable=self.opacity_var,
            command=self.update_opacity
        )
        opacity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Add a label showing the current opacity percentage
        self.opacity_label = ttk.Label(opacity_frame, text=f"{int(self.annotation_opacity * 100)}%")
        self.opacity_label.pack(side=tk.LEFT, padx=5)
        
        # Add a custom brush size entry for even larger sizes
        custom_brush_frame = ttk.Frame(tool_frame)
        custom_brush_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(custom_brush_frame, text="Custom Size:").pack(side=tk.LEFT, padx=5)
        self.custom_brush_entry = ttk.Entry(custom_brush_frame, width=5)
        self.custom_brush_entry.pack(side=tk.LEFT, padx=5)
        self.custom_brush_entry.insert(0, str(self.brush_size))
        
        apply_button = ttk.Button(custom_brush_frame, text="Apply", command=self.apply_custom_brush_size)
        apply_button.pack(side=tk.LEFT, padx=5)
        
        # Fill tool settings (optional - for future enhancements)
        fill_frame = ttk.Frame(tool_frame)
        fill_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(fill_frame, text="Fill Tolerance:").pack(side=tk.LEFT, padx=5)
        
        self.fill_tolerance_var = tk.IntVar(value=30)  # Default tolerance value
        fill_slider = ttk.Scale(
            fill_frame, 
            from_=0, 
            to=100,
            orient=tk.HORIZONTAL, 
            variable=self.fill_tolerance_var
        )
        fill_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(tool_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        undo_button = ttk.Button(button_frame, text="Undo", command=self.undo)
        undo_button.pack(side=tk.LEFT, padx=5)
        
        erase_button = ttk.Button(button_frame, text="Erase", command=self.set_eraser)
        erase_button.pack(side=tk.LEFT, padx=5)
        
        auto_segment_button = ttk.Button(button_frame, text="Auto-Segment", command=self.show_auto_segment_dialog)
        auto_segment_button.pack(side=tk.LEFT, padx=5)
        
        # Segments selection
        segment_frame = ttk.LabelFrame(right_frame, text="Segments")
        segment_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create scroll frame for segments
        canvas = tk.Canvas(segment_frame)
        scrollbar = ttk.Scrollbar(segment_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling on the segment canvas
        def on_mousewheel_segments(event):
            # Windows uses delta, Unix systems use num
            if hasattr(event, 'delta') and event.delta:
                delta = event.delta
            elif hasattr(event, 'num') and event.num:
                delta = -120 if event.num == 5 else 120
            else:
                return
            
            canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        
        # Bind mouse wheel to canvas and scrollable frame
        canvas.bind("<MouseWheel>", on_mousewheel_segments)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel_segments)
        
        # Also bind to Button-4 and Button-5 for Linux/Unix systems
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        scrollable_frame.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        scrollable_frame.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Organized segment buttons
        self.segment_buttons = [None] * len(self.segments)
        
        # Separate hierarchical features (Villi and Gland) from individual features
        hierarchical_features = [segment for segment in self.segments if " - " not in segment and segment in self.segment_hierarchy and self.segment_hierarchy[segment]]
        individual_features = [segment for segment in self.segments if " - " not in segment and (segment not in self.segment_hierarchy or not self.segment_hierarchy[segment])]
        
        feature_counter = 0
        
        # First, add hierarchical features (Villi and Gland) with their subfeatures
        for main_feature in hierarchical_features:
            main_idx = self.segments.index(main_feature)
            
            # Create a frame for the entire group
            feature_frame = ttk.LabelFrame(scrollable_frame, text="")
            feature_frame.pack(fill=tk.X, pady=2, padx=3)
            
            # Add the main feature button
            color = self.segment_colors[main_feature]
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
            main_feature_frame = ttk.Frame(feature_frame)
            main_feature_frame.pack(fill=tk.X, pady=2)
            
            # Prefix with "a.", "b.", etc. based on position
            prefix = chr(97 + feature_counter) + "."
            prefix_label = ttk.Label(main_feature_frame, text=prefix, width=3)
            prefix_label.pack(side=tk.LEFT)
            
            color_button = tk.Button(
                main_feature_frame, 
                bg=color_hex, 
                width=2, 
                height=1,
                command=lambda idx=main_idx: self.select_segment(idx)
            )
            color_button.pack(side=tk.LEFT, padx=5)
            
            segment_label = ttk.Label(main_feature_frame, text=main_feature, font=("Arial", 9, "bold"))
            segment_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.segment_buttons[main_idx] = color_button
            
            # Add all subfeatures for this main feature
            if main_feature in self.segment_hierarchy:
                for i, subfeature in enumerate(self.segment_hierarchy[main_feature]):
                    sub_idx = self.segments.index(subfeature)
                    
                    # Get just the subfeature name without the parent prefix
                    subfeature_name = subfeature.split(" - ")[1]
                    
                    color = self.segment_colors[subfeature]
                    color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                    
                    sub_frame = ttk.Frame(feature_frame)
                    sub_frame.pack(fill=tk.X, pady=1)
                    
                    # Add roman numeral prefix based on position
                    roman_numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
                    prefix = roman_numerals[i] + "." if i < len(roman_numerals) else f"{i+1}."
                    
                    # Add indent and prefix for subfeature
                    indent_frame = ttk.Frame(sub_frame, width=10)
                    indent_frame.pack(side=tk.LEFT)
                    
                    prefix_label = ttk.Label(sub_frame, text=prefix, width=3)
                    prefix_label.pack(side=tk.LEFT)
                    
                    color_button = tk.Button(
                        sub_frame, 
                        bg=color_hex, 
                        width=2, 
                        height=1,
                        command=lambda idx=sub_idx: self.select_segment(idx)
                    )
                    color_button.pack(side=tk.LEFT, padx=5)
                    
                    segment_label = ttk.Label(sub_frame, text=subfeature_name)
                    segment_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    
                    self.segment_buttons[sub_idx] = color_button
            
            feature_counter += 1
        
        # Then, add all individual features
        for feature in individual_features:
            feature_idx = self.segments.index(feature)
            
            # Create frame for individual feature
            individual_frame = ttk.Frame(scrollable_frame)
            individual_frame.pack(fill=tk.X, pady=2, padx=3)
            
            # Add the feature button
            color = self.segment_colors[feature]
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
            # Prefix with "c.", "d.", etc. continuing from hierarchical features
            prefix = chr(97 + feature_counter) + "."
            prefix_label = ttk.Label(individual_frame, text=prefix, width=3)
            prefix_label.pack(side=tk.LEFT)
            
            color_button = tk.Button(
                individual_frame, 
                bg=color_hex, 
                width=2, 
                height=1,
                command=lambda idx=feature_idx: self.select_segment(idx)
            )
            color_button.pack(side=tk.LEFT, padx=5)
            
            segment_label = ttk.Label(individual_frame, text=feature, font=("Arial", 9))
            segment_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.segment_buttons[feature_idx] = color_button
            feature_counter += 1
        
        # Save button
        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill=tk.X, pady=10)
        
        save_button = ttk.Button(save_frame, text="Save Segmentations", command=self.save_segmentations)
        save_button.pack(fill=tk.X)
        
        # Set initial segment
        if self.segments:
            self.select_segment(0)
        
        self.update_image()
    
    def set_brush_tool(self):
        """Set the active tool to brush"""
        self.active_tool = "brush"
        # Use a custom cursor or the default
        try:
            self.canvas.config(cursor="pencil")
        except:
            self.canvas.config(cursor="")  # Default cursor
    
    def set_fill_tool(self):
        """Set the active tool to fill"""
        self.active_tool = "fill"
        # Use a custom cursor or crosshair
        try:
            self.canvas.config(cursor="dotbox")
        except:
            self.canvas.config(cursor="crosshair")
    
    def on_mouse_down(self, event):
        self.drawing = True
        # Convert canvas coordinates to image coordinates
        img_x = (event.x - self.canvas_x_offset) / self.zoom_factor
        img_y = (event.y - self.canvas_y_offset) / self.zoom_factor
        
        # Save current state for undo
        current_mask = self.segmentation_masks[self.current_image_index].copy()
        # Truncate history if we're not at the end
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append((self.current_image_index, current_mask))
        self.history_index = len(self.history) - 1
        
        if self.active_tool == "brush":
            self.prev_x, self.prev_y = img_x, img_y
            self.draw(img_x, img_y)
        elif self.active_tool == "fill" and self.current_color is not None:
            self.fill(img_x, img_y)
    
    def fill(self, x, y):
        """Fill a region with the current color, respecting painted boundaries"""
        if not self.current_color:
            return  # No color selected or eraser
        
        x, y = int(x), int(y)
        
        # Get current mask
        mask = self.segmentation_masks[self.current_image_index]
        
        # Convert to numpy array for better processing
        mask_array = np.array(mask)
        
        # Get mask dimensions
        height, width = mask_array.shape[:2]
        
        # Check if point is within image bounds
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        
        # Get color at target point
        target_color = tuple(mask_array[y, x])
        
        # If target is already the same color, do nothing
        if tuple(target_color) == tuple(self.current_color):
            return
        
        # Start progress indication
        self.root.config(cursor="watch")
        self.root.update()
        
        # Get tolerance value (0-100)
        tolerance = self.fill_tolerance_var.get() / 100.0
        
        # Track boundaries as non-target colors - any non-transparent color that's not the target
        # is considered a boundary
        is_boundary = lambda color: (not np.array_equal(color, target_color) and 
                                     color[3] > 0)  # Alpha > 0 means it's colored
        
        # Use flood fill algorithm
        if tolerance > 0.01 and len(target_color) >= 3:  # Only use tolerance if color has RGB channels
            # Use tolerance-based fill with boundary detection
            self.flood_fill_with_boundaries(x, y, target_color, self.current_color, 
                                           mask_array, tolerance, is_boundary)
        else:
            # Use exact match fill with boundary detection
            self.flood_fill_with_boundaries(x, y, target_color, self.current_color, 
                                          mask_array, 0, is_boundary)
        
        # Update mask with filled array
        new_mask = Image.fromarray(mask_array)
        self.segmentation_masks[self.current_image_index] = new_mask
        
        # Reset cursor and update image
        self.canvas.config(cursor="crosshair" if self.active_tool == "fill" else "")
        self.update_image()
    
    def flood_fill_with_boundaries(self, x, y, target_color, replacement_color, image_array, 
                                  tolerance=0, is_boundary=None):
        """Flood fill algorithm that respects boundaries"""
        height, width = image_array.shape[:2]
        
        # If target color is same as replacement, do nothing
        if tuple(target_color) == tuple(replacement_color):
            return
        
        # Default boundary check if none provided
        if is_boundary is None:
            is_boundary = lambda color: not np.array_equal(color, target_color)
        
        # Convert colors for tolerance calculation if needed
        if tolerance > 0 and len(target_color) >= 3:
            target_rgb = np.array(target_color[:3])  # Use RGB for tolerance
            
            # Max possible difference in RGB space (sqrt(255^2 * 3))
            max_diff = 255 * np.sqrt(3)
            
            # Calculate tolerance threshold based on percentage
            threshold = max_diff * tolerance
            
            # Color similarity check with tolerance
            is_similar = lambda color: np.sqrt(np.sum((np.array(color[:3]) - target_rgb) ** 2)) <= threshold
        else:
            # Exact color matching
            is_similar = lambda color: np.array_equal(color, target_color)
        
        # Queue for BFS
        queue = deque([(y, x)])
        visited = set()
        
        # While queue is not empty
        while queue:
            cy, cx = queue.popleft()
            
            # Skip if out of bounds or already visited
            if (cy, cx) in visited or cy < 0 or cx < 0 or cy >= height or cx >= width:
                continue
            
            # Get color at current position
            current_color = image_array[cy, cx]
            
            # Skip if this pixel is a boundary or not similar to target
            if is_boundary(current_color) or not is_similar(current_color):
                continue
            
            # Mark as visited
            visited.add((cy, cx))
            
            # Change color
            image_array[cy, cx] = replacement_color
            
            # Add neighbors to queue (4-connected)
            neighbors = [(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)]
            for ny, nx in neighbors:
                if (ny, nx) not in visited:
                    queue.append((ny, nx))
                    
            # Periodically update for large fills (every 5000 pixels)
            if len(visited) % 5000 == 0:
                self.root.update_idletasks()
    
    def update_brush_size(self, event=None):
        self.brush_size = self.brush_size_var.get()
        # Update the brush size label
        self.brush_size_label.config(text=f"{self.brush_size}")
    
    def update_opacity(self, event=None):
        """Update annotation opacity and refresh the display"""
        self.annotation_opacity = self.opacity_var.get()
        # Update the opacity label
        self.opacity_label.config(text=f"{int(self.annotation_opacity * 100)}%")
        # Refresh the image display with new opacity
        self.update_image()
    
    def select_segment(self, index):
        self.current_segment = index
        segment_name = self.segments[index]
        self.current_color = self.segment_colors[segment_name]
        
        # Highlight selected segment button
        for i, button in enumerate(self.segment_buttons):
            if i == index:
                button.config(relief=tk.SUNKEN, borderwidth=3)
            else:
                button.config(relief=tk.RAISED, borderwidth=1)
    
    def set_eraser(self):
        self.current_color = None
        # Unhighlight all segment buttons
        for button in self.segment_buttons:
            button.config(relief=tk.RAISED, borderwidth=1)
    
    def update_image(self):
        if not self.original_images:
            return
        
        # Get current image and mask
        original = self.original_images[self.current_image_index]
        mask = self.segmentation_masks[self.current_image_index]
        
        # Debug: Check mask properties
        mask_array = np.array(mask)
        non_transparent_pixels = np.sum(mask_array[:, :, 3] > 0) if len(mask_array.shape) == 3 and mask_array.shape[2] == 4 else 0
        print(f"Debug update_image: Mask size {mask.size}, non-transparent pixels: {non_transparent_pixels}")
        if non_transparent_pixels > 0:
            unique_colors = np.unique(mask_array.reshape(-1, mask_array.shape[-1]), axis=0)
            print(f"Debug update_image: Unique colors in mask: {unique_colors[:10]}")  # Show first 10 unique colors
        
        # Apply opacity to the mask
        if self.annotation_opacity < 1.0:
            # Create a copy of the mask and adjust its alpha channel
            mask_with_opacity = mask.copy()
            mask_array = np.array(mask_with_opacity)
            
            # Apply opacity to non-transparent pixels
            if len(mask_array.shape) == 3 and mask_array.shape[2] == 4:  # RGBA
                # Only modify alpha where there are annotations (alpha > 0)
                non_transparent = mask_array[:, :, 3] > 0
                mask_array[non_transparent, 3] = (mask_array[non_transparent, 3] * self.annotation_opacity).astype(np.uint8)
                mask_with_opacity = Image.fromarray(mask_array, 'RGBA')
            else:
                # Convert to RGBA if not already
                mask_with_opacity = mask.convert('RGBA')
                mask_array = np.array(mask_with_opacity)
                non_transparent = mask_array[:, :, 3] > 0
                mask_array[non_transparent, 3] = (mask_array[non_transparent, 3] * self.annotation_opacity).astype(np.uint8)
                mask_with_opacity = Image.fromarray(mask_array, 'RGBA')
        else:
            mask_with_opacity = mask
        
        # Debug: Check mask after opacity
        mask_with_opacity_array = np.array(mask_with_opacity)
        non_transparent_after = np.sum(mask_with_opacity_array[:, :, 3] > 0) if len(mask_with_opacity_array.shape) == 3 and mask_with_opacity_array.shape[2] == 4 else 0
        print(f"Debug update_image: After opacity ({self.annotation_opacity}), non-transparent pixels: {non_transparent_after}")
        
        # Combine original image and segmentation mask
        combined = original.copy()
        combined.paste(mask_with_opacity, (0, 0), mask_with_opacity)
        
        # Debug: Check combined image
        combined_array = np.array(combined)
        print(f"Debug update_image: Combined image size: {combined.size}, mode: {combined.mode}")
        
        # Apply zoom
        width, height = combined.size
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        
        zoomed_img = combined.resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(zoomed_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas_x_offset, 
            self.canvas_y_offset, 
            anchor=tk.NW, 
            image=self.tk_image
        )
        
        # Update scrollregion to accommodate the image and offsets
        self.canvas.config(
            scrollregion=(
                min(0, self.canvas_x_offset),
                min(0, self.canvas_y_offset),
                max(self.canvas.winfo_width(), self.canvas_x_offset + new_width),
                max(self.canvas.winfo_height(), self.canvas_y_offset + new_height)
            )
        )
        
        # Update zoom label
        if hasattr(self, 'zoom_label'):
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        
        # Update slice label if multiple images
        if len(self.original_images) > 1 and hasattr(self, 'slice_label'):
            self.slice_label.config(text=f"Slice: {self.current_image_index + 1}/{len(self.original_images)}")
        
        print(f"Debug update_image: Image display updated")
    
    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image()
    
    def next_image(self):
        if self.current_image_index < len(self.original_images) - 1:
            self.current_image_index += 1
            self.update_image()
    
    def on_mouse_move(self, event):
        if self.drawing and self.active_tool == "brush":
            # Convert canvas coordinates to image coordinates
            img_x = (event.x - self.canvas_x_offset) / self.zoom_factor
            img_y = (event.y - self.canvas_y_offset) / self.zoom_factor
            self.draw(img_x, img_y, line=True)
            self.prev_x, self.prev_y = img_x, img_y
    
    def on_mouse_up(self, event):
        self.drawing = False
        if self.active_tool == "brush":
            self.prev_x, self.prev_y = None, None
    
    def draw(self, x, y, line=False):
        mask = self.segmentation_masks[self.current_image_index]
        draw = ImageDraw.Draw(mask)
        
        if self.current_color is None:  # Eraser
            # Create a larger circular eraser
            if line and self.prev_x is not None and self.prev_y is not None:
                draw.line((self.prev_x, self.prev_y, x, y), fill=(0, 0, 0, 0), width=self.brush_size * 2)
            else:
                draw.ellipse((x - self.brush_size, y - self.brush_size, 
                             x + self.brush_size, y + self.brush_size), fill=(0, 0, 0, 0))
        else:  # Drawing
            if line and self.prev_x is not None and self.prev_y is not None:
                draw.line((self.prev_x, self.prev_y, x, y), fill=self.current_color, width=self.brush_size)
            else:
                draw.ellipse((x - self.brush_size//2, y - self.brush_size//2, 
                             x + self.brush_size//2, y + self.brush_size//2), fill=self.current_color)
        
        # Update the combined image
        self.update_image()
    
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            image_index, mask = self.history[self.history_index]
            if image_index == self.current_image_index:
                self.segmentation_masks[self.current_image_index] = mask
                self.update_image()
            else:
                messagebox.showinfo("Info", "Cannot undo - action was on a different slice")
        else:
            messagebox.showinfo("Info", "Nothing to undo")
    
    def save_segmentations(self):
        if not self.original_images:
            messagebox.showerror("Error", "No images to save")
            return
        
        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select Save Location")
        if not save_dir:
            return
        
        # Get base name from the original file
        base_name = os.path.splitext(os.path.basename(self.image_paths[0]))[0]
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save each segmentation (one per original image)
            for i, mask in enumerate(self.segmentation_masks):
                # Create a new RGB image with black background
                segmented_img = Image.new('RGB', mask.size, (0, 0, 0))
                mask_data = np.array(mask)
                segmented_data = np.array(segmented_img)
                
                # For each pixel in the mask, determine which segment it belongs to
                # and assign the appropriate color
                for segment, color in self.segment_colors.items():
                    r, g, b, a = color
                    # Find pixels matching this segment's color (ignoring alpha)
                    matching_pixels = np.all(mask_data[:, :, :3] == np.array([r, g, b]), axis=2)
                    # Set those pixels in the output image
                    segmented_data[matching_pixels] = [r, g, b]
                
                # Convert back to PIL Image
                segmented_img = Image.fromarray(segmented_data)
                
                # Create filename
                if len(self.original_images) > 1:
                    filename = f"segmented_{base_name}_slice_{i+1}.png"
                else:
                    filename = f"segmented_{base_name}.png"
                
                # Clean filename to be valid
                filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
                
                # Save the image
                save_path = os.path.join(save_dir, filename)
                segmented_img.save(save_path)
            
            messagebox.showinfo("Success", f"Segmentations saved to {save_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save segmentations: {str(e)}")
    
    def apply_custom_brush_size(self):
        try:
            size = int(self.custom_brush_entry.get())
            if size < 1:
                size = 1
            self.brush_size = size
            # Update the slider if within its range
            if size <= 100:
                self.brush_size_var.set(size)
            # Update the label
            self.brush_size_label.config(text=f"{size}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for brush size")

    def load_next_file(self):
        """Load the next image file in the directory"""
        if not self.directory_files or self.current_file_idx >= len(self.directory_files) - 1:
            messagebox.showinfo("Info", "No more files in the directory")
            return
        
        # Ask if user wants to save current segmentation
        if self.segmentation_masks and messagebox.askyesno("Save", "Do you want to save the current segmentation?"):
            self.save_segmentations()
        
        # Move to next file
        self.current_file_idx += 1
        next_file = self.directory_files[self.current_file_idx]
        self.image_paths = [next_file]
        
        # Reset zoom
        self.zoom_factor = 1.0
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0
        
        # Check if this file should be auto-segmented
        should_auto_segment = (hasattr(self, 'auto_segment_model_ready') and 
                              self.auto_segment_model_ready and 
                              hasattr(self, 'auto_segment_queue') and 
                              next_file in self.auto_segment_queue)
        
        # Debug: Show queue status
        if hasattr(self, 'auto_segment_queue'):
            print(f"Debug load_next_file: Queue has {len(self.auto_segment_queue)} files")
            print(f"Debug load_next_file: Current file: {os.path.basename(next_file)}")
            print(f"Debug load_next_file: File in queue: {next_file in self.auto_segment_queue}")
            print(f"Debug load_next_file: Model ready: {getattr(self, 'auto_segment_model_ready', False)}")
            print(f"Debug load_next_file: Should auto-segment: {should_auto_segment}")
            
            # Show remaining files in queue
            if self.auto_segment_queue:
                print("Debug load_next_file: Remaining files in queue:")
                for i, file_path in enumerate(self.auto_segment_queue):
                    print(f"  Queue[{i}]: {os.path.basename(file_path)}")
            else:
                print("Debug load_next_file: Queue is empty")
        else:
            print("Debug load_next_file: No auto_segment_queue found")
        
        # Process file
        file_ext = os.path.splitext(next_file)[1].lower()
        if file_ext == '.nd2':
            self.process_nd2_file(next_file)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            self.process_regular_image(next_file)
        else:
            messagebox.showerror("Error", f"Unsupported file format: {file_ext}")
            return
        
        # Reset history
        self.history = []
        self.history_index = -1
        
        # Update UI first
        self.create_annotation_window()
        
        # Apply auto-segmentation AFTER UI is created
        if should_auto_segment:
            print(f"Debug: Applying auto-segmentation to {next_file}")
            # Ensure UI is fully updated before applying auto-segmentation
            self.root.update_idletasks()
            self.root.after(100, self.apply_auto_segmentation_to_current_image)  # Small delay to ensure UI is ready

    def load_prev_file(self):
        """Load the previous image file in the directory"""
        if not self.directory_files or self.current_file_idx <= 0:
            messagebox.showinfo("Info", "No previous files in the directory")
            return
        
        # Ask if user wants to save current segmentation
        if self.segmentation_masks and messagebox.askyesno("Save", "Do you want to save the current segmentation?"):
            self.save_segmentations()
        
        # Move to previous file
        self.current_file_idx -= 1
        prev_file = self.directory_files[self.current_file_idx]
        self.image_paths = [prev_file]
        
        # Reset zoom
        self.zoom_factor = 1.0
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0
        
        # Process file
        file_ext = os.path.splitext(prev_file)[1].lower()
        if file_ext == '.nd2':
            self.process_nd2_file(prev_file)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            self.process_regular_image(prev_file)
        else:
            messagebox.showerror("Error", f"Unsupported file format: {file_ext}")
            return
        
        # Reset history
        self.history = []
        self.history_index = -1
        
        # Update UI
        self.create_annotation_window()

    # Zoom and pan functions
    def zoom_in(self, event=None):
        """Zoom in the image"""
        self.zoom_factor *= 1.2
        self.update_image()

    def zoom_out(self, event=None):
        """Zoom out the image"""
        self.zoom_factor /= 1.2
        if self.zoom_factor < 0.1:  # Limit minimum zoom
            self.zoom_factor = 0.1
        self.update_image()

    def zoom_with_wheel(self, event):
        """Zoom using mouse wheel"""
        # Windows uses delta, Unix systems use num
        if hasattr(event, 'delta') and event.delta:
            delta = event.delta
        elif hasattr(event, 'num') and event.num:
            delta = -120 if event.num == 5 else 120
        else:
            return
        
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def reset_zoom(self):
        """Reset zoom to original size"""
        self.zoom_factor = 1.0
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0
        self.update_image()

    def start_pan(self, event):
        """Start panning the image"""
        self.dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def pan_image(self, event):
        """Pan the image as the mouse moves"""
        if not self.dragging:
            return
        
        # Calculate how much the mouse has moved
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # Update the canvas offset
        self.canvas_x_offset += dx
        self.canvas_y_offset += dy
        
        # Update the drag start position
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        # Update the image
        self.update_image()

    def stop_pan(self, event):
        """Stop panning the image"""
        self.dragging = False

    def process_nd2_array(self, img_array):
        """Process a numpy array from ND2 file into a PIL Image"""
        try:
            if img_array is None or img_array.size == 0:
                logger.error("Empty image array received")
                return None
            
            # Log shape information
            logger.info(f"Processing array with shape {img_array.shape} and dtype {img_array.dtype}")
            
            # Handle array with NaN values
            if np.isnan(img_array).any():
                logger.warning("Array contains NaN values, replacing with zeros")
                img_array = np.nan_to_num(img_array)
            
            # Convert bit depth if necessary
            if img_array.dtype != np.uint8:
                # Get the min and max, handling potential errors
                try:
                    min_val = np.min(img_array)
                    max_val = np.max(img_array)
                    
                    # Check for valid range
                    if max_val > min_val:
                        # Use robust normalization to handle outliers
                        p1, p99 = np.percentile(img_array, (1, 99))
                        logger.info(f"Using percentile normalization: 1%={p1}, 99%={p99}")
                        
                        # Clip to remove extreme outliers
                        img_array_clipped = np.clip(img_array, p1, p99)
                        
                        # Normalize to 0-255
                        normalized = ((img_array_clipped - p1) / (p99 - p1) * 255).astype(np.uint8)
                    else:
                        # Fallback to simple normalization if percentile doesn't work well
                        logger.warning("Using simple min-max normalization")
                        normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        
                        # If still problematic, create a gray image
                        if np.isnan(normalized).any() or np.isinf(normalized).any():
                            logger.warning("Normalization produced invalid values, using flat gray image")
                            normalized = np.ones(img_array.shape, dtype=np.uint8) * 128
                except Exception as e:
                    logger.error(f"Error in normalization: {str(e)}")
                    # Create a blank image as fallback
                    normalized = np.zeros(img_array.shape, dtype=np.uint8)
                
                img_array = normalized
            
            # Handle different dimensions
            if len(img_array.shape) == 2:
                # Grayscale image - convert to RGB
                logger.info("Converting 2D grayscale to RGB")
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            elif len(img_array.shape) == 3:
                # Handle different 3D formats
                if img_array.shape[2] == 1:
                    # Single channel image - expand to RGB
                    logger.info("Converting single channel to RGB")
                    img_array = np.concatenate([img_array] * 3, axis=2)
                elif img_array.shape[2] > 4:
                    # Too many channels, take the first three
                    logger.info(f"Taking first 3 channels from {img_array.shape[2]} channels")
                    img_array = img_array[:, :, 0:3]
                elif img_array.shape[2] == 2:
                    # Two channels, add a third
                    logger.info("Adding third channel to 2-channel image")
                    third_channel = np.zeros_like(img_array[:, :, 0:1])
                    img_array = np.concatenate([img_array, third_channel], axis=2)
                
                # Check if channels are in first dimension instead of last
                if img_array.shape[0] == 3 and img_array.shape[2] > 100:
                    logger.info("Transposing channels from first to last dimension")
                    img_array = np.transpose(img_array, (1, 2, 0))
            elif len(img_array.shape) > 3:
                # Complex multi-dimensional data
                logger.warning(f"Complex array shape: {img_array.shape}")
                
                # Try various approaches to extract a meaningful image
                try:
                    if img_array.shape[0] == 3 and len(img_array.shape) == 4:
                        # Likely RGB with channels as first dimension
                        logger.info("Converting from CHW to HWC format")
                        img_array = np.transpose(img_array[0:3], (1, 2, 0))
                    else:
                        # Take slices until we get to 3D
                        logger.info("Reducing dimensions by taking first slice repeatedly")
                        while len(img_array.shape) > 3:
                            img_array = img_array[0]
                        
                        # If we end up with 2D, convert to RGB
                        if len(img_array.shape) == 2:
                            img_array = np.stack([img_array, img_array, img_array], axis=-1)
                        elif img_array.shape[2] > 3:
                            img_array = img_array[:, :, 0:3]
                except Exception as e:
                    logger.error(f"Error restructuring complex array: {str(e)}")
                    # Create a blank image in case of errors
                    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Final validation check
            if img_array.shape[2] != 3 and img_array.shape[2] != 4:
                logger.error(f"Invalid final channel count: {img_array.shape[2]}")
                return None
                
            # Check for invalid values before creating PIL image
            if np.isnan(img_array).any() or np.isinf(img_array).any():
                logger.warning("Final array contains NaN or Inf values, replacing with zeros")
                img_array = np.nan_to_num(img_array)
            
            # Create PIL image
            img = Image.fromarray(img_array)
            logger.info(f"Successfully created PIL image with size {img.size}")
            return img
        
        except Exception as e:
            logger.error(f"Error processing ND2 array: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def show_nd2_troubleshooting(self):
        """Show a window with ND2 troubleshooting options"""
        # Create troubleshooting window
        troubleshoot_window = tk.Toplevel(self.root)
        troubleshoot_window.title("ND2 Troubleshooting")
        troubleshoot_window.geometry("500x400")
        troubleshoot_window.transient(self.root)
        troubleshoot_window.grab_set()
        
        # Add scrollable text widget
        frame = ttk.Frame(troubleshoot_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        title = ttk.Label(frame, text="ND2 Loading Troubleshooting", font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)
        
        # Add troubleshooting content
        text.insert(tk.END, "The ND2 file could not be loaded. Here are some possible solutions:\n\n")
        
        text.insert(tk.END, "1. Package Installation\n", "header")
        text.insert(tk.END, "Make sure you have installed the required packages:\n")
        text.insert(tk.END, "   - Run: pip install nd2reader\n")
        text.insert(tk.END, "   - Run: pip install nd2\n\n")
        
        text.insert(tk.END, "2. Check File Format\n", "header")
        text.insert(tk.END, "Make sure the ND2 file is a valid Nikon format file.\n\n")
        
        text.insert(tk.END, "3. File Path Issues\n", "header")
        text.insert(tk.END, "Avoid special characters or very long paths in the file location.\n\n")
        
        text.insert(tk.END, "4. Memory Limitations\n", "header")
        text.insert(tk.END, "ND2 files can be very large. Try using a smaller file or closing other applications.\n\n")
        
        text.insert(tk.END, "5. Check Log File\n", "header")
        text.insert(tk.END, "Review the 'nd2_loading.log' file for detailed error information.\n\n")
        
        text.insert(tk.END, "6. Alternative Formats\n", "header")
        text.insert(tk.END, "Consider converting your ND2 file to TIFF format using NIS-Elements or other software.\n\n")
        
        # Make headers bold
        text.tag_configure("header", font=("Arial", 10, "bold"))
        
        # Make text widget read-only
        text.config(state=tk.DISABLED)
        
        # Close button
        close_button = ttk.Button(frame, text="Close", command=troubleshoot_window.destroy)
        close_button.pack(pady=10)

    def load_new_folder(self):
        """Load images from a new folder"""
        # Ask if user wants to save current segmentation
        if self.segmentation_masks and messagebox.askyesno("Save", "Do you want to save the current segmentation?"):
            self.save_segmentations()
        
        # Ask for directory
        new_dir = filedialog.askdirectory(title="Select Image Folder")
        if not new_dir:
            return  # User cancelled
        
        # Check if directory contains images
        image_files = self.get_image_files_in_directory(new_dir)
        if not image_files:
            messagebox.showerror("Error", "No supported image files found in the selected folder")
            return
        
        # Load the first image from the new directory
        self.current_directory = new_dir
        self.image_paths = [image_files[0]]
        self.directory_files = image_files
        self.current_file_idx = 0
        
        # Reset zoom
        self.zoom_factor = 1.0
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0
        
        # Process file based on extension
        file_ext = os.path.splitext(image_files[0])[1].lower()
        if file_ext == '.nd2':
            self.process_nd2_file(image_files[0])
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            self.process_regular_image(image_files[0])
        else:
            messagebox.showerror("Error", f"Unsupported file format: {file_ext}")
            return
        
        # Reset history
        self.history = []
        self.history_index = -1
        
        # Update UI
        self.create_annotation_window()

    def auto_segment_slices(self, num_slices=1):
        """Auto-segment subsequent slices based on current slice annotation"""
        if not self.original_images or not self.segmentation_masks:
            messagebox.showerror("Error", "No image loaded")
            return
        
        if self.current_image_index + num_slices >= len(self.original_images):
            messagebox.showerror("Error", "Not enough subsequent slices available")
            return

        try:
            print(f"Debug: Starting auto-segmentation of {num_slices} slices from current index {self.current_image_index}")
            
            # Initialize model if not already done
            if self.model is None:
                num_classes = len(self.segments)  # One class per segment type
                self.model = UNet(n_channels=3, n_classes=num_classes).to(self.device)
            
            # Define transform for tensor conversion (without resize since we manually resize)
            transform_tensor_only = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Prepare current slice as training data
            current_img = self.original_images[self.current_image_index]
            current_mask = self.segmentation_masks[self.current_image_index]
            
            # Convert to RGB mode to ensure 3 channels and resize to fixed size
            current_img = current_img.convert('RGB').resize((256, 256), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
            # Convert mask to training format - resize to match input
            current_mask_resized = current_mask.resize((256, 256), Image.NEAREST)
            mask_array = np.array(current_mask_resized)
            
            # Ensure mask_array is the right shape (256, 256, 4) for RGBA
            if len(mask_array.shape) == 2:
                # Convert grayscale to RGBA
                mask_array = np.stack([mask_array, mask_array, mask_array, np.ones_like(mask_array) * 255], axis=2)
            elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                # Convert RGB to RGBA
                alpha = np.ones((mask_array.shape[0], mask_array.shape[1], 1), dtype=mask_array.dtype) * 255
                mask_array = np.concatenate([mask_array, alpha], axis=2)
            
            training_mask = np.zeros((256, 256), dtype=np.int64)
            
            # Debug: print mask array shape
            logger.info(f"Mask array shape: {mask_array.shape}")
            logger.info(f"Training mask shape: {training_mask.shape}")
            
            # Create class mapping for each segment color
            for idx, segment in enumerate(self.segments):
                color = self.segment_colors[segment]
                # Create boolean mask for this color
                if len(mask_array.shape) == 3 and mask_array.shape[2] >= 3:
                    # RGBA or RGB mask - ensure we're working with the right dimensions
                    color_match = np.all(mask_array[:, :, :3] == np.array(color[:3]), axis=2)
                    logger.info(f"Boolean mask shape for {segment}: {color_match.shape}")
                    
                    # Verify dimensions match before applying
                    if color_match.shape == training_mask.shape:
                        training_mask[color_match] = idx
                    else:
                        logger.error(f"Shape mismatch: color_match {color_match.shape} vs training_mask {training_mask.shape}")
                else:
                    logger.warning(f"Unexpected mask array shape: {mask_array.shape}")
                    continue
            
            # Convert to tensors - image is already resized to 256x256
            img_tensor = transform_tensor_only(current_img).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(training_mask).unsqueeze(0).to(self.device)
            
            # Train model on current slice
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Quick fine-tuning
            for epoch in range(10):  # Adjust number of iterations as needed
                optimizer.zero_grad()
                output = self.model(img_tensor)
                loss = criterion(output, mask_tensor)
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    print(f"Debug: Training epoch {epoch}, loss: {loss.item():.4f}")
            
            # Auto-segment subsequent slices
            print(f"Debug: Applying auto-segmentation to {num_slices} subsequent slices...")
            self.model.eval()
            processed_slices = []
            
            with torch.no_grad():
                for i in range(num_slices):
                    next_idx = self.current_image_index + i + 1
                    if next_idx >= len(self.original_images):
                        print(f"Debug: Reached end of images at index {next_idx}")
                        break
                    
                    print(f"Debug: Processing slice {next_idx} ({i+1}/{num_slices})")
                    
                    # Prepare next slice
                    next_img = self.original_images[next_idx]
                    original_size = next_img.size  # Store original size for later
                    next_img = next_img.convert('RGB').resize((256, 256), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
                    img_tensor = transform_tensor_only(next_img).unsqueeze(0).to(self.device)
                    
                    # Generate prediction
                    output = self.model(img_tensor)
                    pred_mask = output.argmax(1).squeeze().cpu().numpy()
                    
                    # Convert prediction to RGBA mask at 256x256
                    temp_mask = np.zeros((256, 256, 4), dtype=np.uint8)
                    
                    # Fill in predicted segments with visibility improvements
                    pixels_filled = 0
                    for idx, segment in enumerate(self.segments):
                        color = self.segment_colors[segment]
                        mask_pixels = pred_mask == idx
                        pixel_count = np.sum(mask_pixels)
                        if pixel_count > 0:
                            # Special handling for black segments - make them more visible
                            if color[:3] == (0, 0, 0):  # If it's black
                                visible_color = (64, 64, 64, 255)
                                temp_mask[mask_pixels] = visible_color
                            else:
                                temp_mask[mask_pixels] = color
                            pixels_filled += pixel_count
                    
                    print(f"Debug: Slice {next_idx} - filled {pixels_filled} pixels")
                    
                    # Convert to PIL Image and resize back to original size
                    pred_mask_img = Image.fromarray(temp_mask, mode='RGBA')
                    pred_mask_resized = pred_mask_img.resize(original_size, Image.NEAREST)
                    
                    # Update segmentation mask
                    self.segmentation_masks[next_idx] = pred_mask_resized
                    processed_slices.append(next_idx)
            
            # Show success message with navigation instructions
            if processed_slices:
                slice_list = ", ".join([str(idx+1) for idx in processed_slices])  # Convert to 1-based indexing for user
                messagebox.showinfo("Success", 
                                  f"✓ Auto-segmented {len(processed_slices)} slices: {slice_list}\n\n"
                                  f"Use 'Next Slice' and 'Previous Slice' buttons to view the results.\n"
                                  f"Current slice: {self.current_image_index + 1}")
                print(f"Debug: Successfully processed slices: {processed_slices}")
            else:
                messagebox.showwarning("Warning", "No slices were processed")
            
            # Update current image display
            self.update_image()
            
        except Exception as e:
            logger.error(f"Error in auto-segmentation: {str(e)}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Auto-segmentation failed: {str(e)}")

    def auto_segment_folder_images(self, num_images=1):
        """Auto-segment other images in the folder based on current image annotation"""
        if not self.original_images or not self.segmentation_masks:
            messagebox.showerror("Error", "No image loaded")
            return
    
        if not self.directory_files or len(self.directory_files) <= 1:
            messagebox.showerror("Error", "No other images available in folder")
            return

        progress_window = None
        try:
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Folder Auto-segmentation")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
        
            progress_label = ttk.Label(progress_window, text="Initializing...", wraplength=380)
            progress_label.pack(pady=10)
        
            status_label = ttk.Label(progress_window, text="", wraplength=380)
            status_label.pack(pady=5)
        
            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
            progress_bar.pack(pady=10)
        
            # Initialize model if not already done
            if self.model is None:
                progress_label.config(text="Initializing model...")
                progress_window.update()
            
                num_classes = len(self.segments)
                self.model = UNet(n_channels=3, n_classes=num_classes).to(self.device)
        
            # Define transform for tensor conversion
            transform_tensor_only = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
            # Train on current image (same logic as auto_segment_slices)
            current_img = self.original_images[self.current_image_index]
            current_mask = self.segmentation_masks[self.current_image_index]
        
            current_img = current_img.convert('RGB').resize((256, 256), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            current_mask_resized = current_mask.resize((256, 256), Image.NEAREST)
            mask_array = np.array(current_mask_resized)
        
            if len(mask_array.shape) == 2:
                mask_array = np.stack([mask_array, mask_array, mask_array, np.ones_like(mask_array) * 255], axis=2)
            elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                alpha = np.ones((mask_array.shape[0], mask_array.shape[1], 1), dtype=mask_array.dtype) * 255
                mask_array = np.concatenate([mask_array, alpha], axis=2)
        
            training_mask = np.zeros((256, 256), dtype=np.int64)
        
            for idx, segment in enumerate(self.segments):
                color = self.segment_colors[segment]
                if len(mask_array.shape) == 3 and mask_array.shape[2] >= 3:
                    color_match = np.all(mask_array[:, :, :3] == np.array(color[:3]), axis=2)
                    if color_match.shape == training_mask.shape:
                        training_mask[color_match] = idx
        
            img_tensor = transform_tensor_only(current_img).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(training_mask).unsqueeze(0).to(self.device)
        
            # Train model
            progress_label.config(text="Training model...")
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
        
            num_epochs = 50
            progress_bar['maximum'] = num_epochs
        
            for epoch in range(num_epochs):
                try:
                    optimizer.zero_grad()
                    output = self.model(img_tensor)
                    loss = criterion(output, mask_tensor)
                    loss.backward()
                    optimizer.step()
                
                    # Safely update progress if window still exists
                    if progress_window.winfo_exists():
                        progress_bar['value'] = epoch + 1
                        progress_label.config(text=f"Training: {epoch+1}/{num_epochs}")
                        progress_window.update()
                    else:
                        # Window was closed, break out of training
                        break
                except tk.TclError:
                    # Window was destroyed, break out of training
                    break
                except Exception as e:
                    logger.error(f"Error during training epoch {epoch}: {str(e)}")
                    break
        
            # Safely destroy progress window
            if progress_window and progress_window.winfo_exists():
                progress_window.destroy()
            progress_window = None
            
            # Set up auto-segmentation queue for subsequent file loads
            current_file = self.directory_files[self.current_file_idx]
            # Get files that come AFTER the current file in directory order
            remaining_files = self.directory_files[self.current_file_idx + 1:]

            # Debug: Show all available files
            print(f"Debug: Current file: {os.path.basename(current_file)}")
            print(f"Debug: Current file index: {self.current_file_idx}")
            print(f"Debug: Total files in directory: {len(self.directory_files)}")
            print(f"Debug: Remaining files after current: {len(remaining_files)}")
            for i, file_path in enumerate(remaining_files):
                print(f"Debug: Remaining[{i}]: {os.path.basename(file_path)}")

            # Ensure we don't exceed available files
            actual_num_images = min(num_images, len(remaining_files))
            self.auto_segment_queue = remaining_files[:actual_num_images]
            self.auto_segment_model_ready = True

            print(f"Debug: Requested {num_images} images, setting up queue with {len(self.auto_segment_queue)} files:")
            for i, file_path in enumerate(self.auto_segment_queue):
                print(f"Debug: Queue[{i}]: {os.path.basename(file_path)}")

            messagebox.showinfo("Training Complete", 
                              f"✓ Model trained successfully!\n\n"
                              f"Queue set up with {len(self.auto_segment_queue)} images (requested: {num_images}).\n"
                              f"Directory has {len(self.directory_files)} total files.\n\n"
                              f"Now use 'Next File' button to navigate through the images.\n"
                              f"Each image will be automatically segmented for your review and modification.\n\n"
                              f"Files in queue: {[os.path.basename(f) for f in self.auto_segment_queue]}\n\n"
                              f"Use 'Save Segmentations' to save any image you want to keep.")
        
        except Exception as e:
            logger.error(f"Error in folder auto-segmentation: {str(e)}")
            # Safely destroy progress window if it exists
            if progress_window:
                try:
                    if progress_window.winfo_exists():
                        progress_window.destroy()
                except:
                    pass
            messagebox.showerror("Error", f"Folder auto-segmentation failed: {str(e)}")

    def show_auto_segment_dialog(self):
        """Show dialog to get number of slices to auto-segment"""
        if not self.original_images or not self.segmentation_masks:
            messagebox.showerror("Error", "No image loaded")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Auto-Segment Options")
        dialog.geometry("500x400")  # Increased from 400x300
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Add content
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Auto-Segmentation Options", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Option selection
        option_var = tk.StringVar(value="slices")
        
        # Option 1: Subsequent slices (for ND2 files)
        slices_frame = ttk.LabelFrame(frame, text="Subsequent Slices")
        slices_frame.pack(fill=tk.X, pady=5)
        
        slices_radio = ttk.Radiobutton(
            slices_frame,
            text="Apply to subsequent slices in current file",
            variable=option_var,
            value="slices"
        )
        slices_radio.pack(anchor=tk.W, padx=10, pady=5)
        
        # Calculate maximum available slices
        max_slices = len(self.original_images) - self.current_image_index - 1
        
        slices_control_frame = ttk.Frame(slices_frame)
        slices_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(slices_control_frame, text="Number of slices:").pack(side=tk.LEFT)
        
        slices_var = tk.StringVar(value="1")
        slices_spinbox = ttk.Spinbox(
            slices_control_frame,
            from_=1,
            to=max(1, max_slices),
            textvariable=slices_var,
            width=10
        )
        slices_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(slices_control_frame, text=f"(Max: {max_slices})").pack(side=tk.LEFT, padx=5)
        
        # Option 2: Folder images
        folder_frame = ttk.LabelFrame(frame, text="Folder Images")
        folder_frame.pack(fill=tk.X, pady=5)
        
        folder_radio = ttk.Radiobutton(
            folder_frame,
            text="Apply to other images in folder",
            variable=option_var,
            value="folder"
        )
        folder_radio.pack(anchor=tk.W, padx=10, pady=5)
        
        folder_control_frame = ttk.Frame(folder_frame)
        folder_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Calculate available folder images
        available_files = 0
        if self.directory_files:
            available_files = len(self.directory_files) - 1  # Exclude current file
        
        ttk.Label(folder_control_frame, text="Number of images:").pack(side=tk.LEFT)
        
        folder_var = tk.StringVar(value="1")
        folder_spinbox = ttk.Spinbox(
            folder_control_frame,
            from_=1,
            to=max(1, available_files),
            textvariable=folder_var,
            width=10
        )
        folder_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(folder_control_frame, text=f"(Available: {available_files})").pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=tk.LEFT, padx=5)
        
        def execute_auto_segment():
            try:
                if option_var.get() == "slices":
                    num_slices = int(slices_var.get())
                    dialog.destroy()
                    self.auto_segment_slices(num_slices)
                else:  # folder
                    num_images = int(folder_var.get())
                    dialog.destroy()
                    self.auto_segment_folder_images(num_images)
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        ttk.Button(
            button_frame,
            text="Auto-Segment",
            command=execute_auto_segment
        ).pack(side=tk.RIGHT, padx=5)
        
        # Add diagnostic button for debugging
        ttk.Button(
            button_frame,
            text="Debug Queue",
            command=self.show_queue_diagnostic
        ).pack(side=tk.RIGHT, padx=5)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')

    def apply_auto_segmentation_to_current_image(self):
        """Apply auto-segmentation to the currently loaded image"""
        try:
            if not self.model or not self.auto_segment_model_ready:
                print("Debug: Model not ready or auto_segment_model_ready is False")
                return
            
            print("Debug: Starting auto-segmentation application...")
            
            # Define transform for tensor conversion
            transform_tensor_only = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Get current image
            current_img = self.original_images[self.current_image_index]
            original_size = current_img.size
            print(f"Debug: Processing image of size {original_size}")
            
            # Resize for model processing
            img_resized = current_img.convert('RGB').resize((256, 256), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            img_tensor = transform_tensor_only(img_resized).unsqueeze(0).to(self.device)
            
            # Generate prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)
                pred_mask = output.argmax(1).squeeze().cpu().numpy()
            
            print(f"Debug: Prediction mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")
            
            # Convert prediction to RGBA mask at 256x256
            temp_mask = np.zeros((256, 256, 4), dtype=np.uint8)
            
            # Fill in predicted segments
            pixels_filled = 0
            for idx, segment in enumerate(self.segments):
                color = self.segment_colors[segment]
                mask_pixels = pred_mask == idx
                pixel_count = np.sum(mask_pixels)
                if pixel_count > 0:
                    # Special handling for black segments - make them more visible
                    if color[:3] == (0, 0, 0):  # If it's black
                        # Change black to dark gray for better visibility
                        visible_color = (64, 64, 64, 255)
                        print(f"Debug: Converting black segment '{segment}' to dark gray for visibility")
                        temp_mask[mask_pixels] = visible_color
                    else:
                        temp_mask[mask_pixels] = color
                    print(f"Debug: Segment '{segment}' (idx {idx}): {pixel_count} pixels with color {color}")
                    pixels_filled += pixel_count
            
            print(f"Debug: Total pixels filled: {pixels_filled}")
            
            # Convert to PIL Image and resize back to original size
            pred_mask_img = Image.fromarray(temp_mask, mode='RGBA')
            pred_mask_resized = pred_mask_img.resize(original_size, Image.NEAREST)
            
            # Update segmentation mask
            self.segmentation_masks[self.current_image_index] = pred_mask_resized
            print(f"Debug: Updated segmentation mask for image index {self.current_image_index}")
            
            # Temporarily set full opacity for auto-segmented images to make them more visible
            original_opacity = self.annotation_opacity
            self.annotation_opacity = 1.0
            print(f"Debug: Temporarily setting opacity to 100% (was {original_opacity*100}%)")
            
            # Update window title to show auto-segmented status
            base_name = os.path.basename(self.image_paths[0])
            queue_remaining = len(self.auto_segment_queue) if hasattr(self, 'auto_segment_queue') else 0
            self.root.title(f"Tissue Segmentation Tool - Auto-Segmented: {base_name} (Queue: {queue_remaining} remaining)")
            
            # Update status label if it exists
            if hasattr(self, 'status_label'):
                self.status_label.config(text="✓ Auto-Segmented")
            
            # Update the display
            self.update_image()
            print(f"Debug: Display updated for auto-segmented image: {base_name}")
            
            # Remove this file from the auto-segmentation queue since it's been processed
            current_file = self.image_paths[0]
            print(f"Debug: Checking if {os.path.basename(current_file)} should be removed from queue")
            print(f"Debug: Current file full path: {current_file}")

            if hasattr(self, 'auto_segment_queue') and current_file in self.auto_segment_queue:
                print(f"Debug: Queue before removal has {len(self.auto_segment_queue)} files:")
                for i, file_path in enumerate(self.auto_segment_queue):
                    print(f"  Queue[{i}]: {os.path.basename(file_path)} -> {file_path}")
                
                self.auto_segment_queue.remove(current_file)
                print(f"Debug: Removed {os.path.basename(current_file)} from queue. Queue now has {len(self.auto_segment_queue)} files")
                
                if self.auto_segment_queue:
                    print("Debug: Remaining files in queue:")
                    for i, file_path in enumerate(self.auto_segment_queue):
                        print(f"  Queue[{i}]: {os.path.basename(file_path)} -> {file_path}")
                else:
                    print("Debug: Queue is now empty")
                
                # If queue is empty, disable auto-segmentation
                if not self.auto_segment_queue:
                    self.auto_segment_model_ready = False
                    print("Debug: Queue is empty, disabling auto-segmentation")
                
                # Show notification about auto-segmentation
                if self.auto_segment_queue:
                    messagebox.showinfo("Auto-Segmentation Applied", 
                                      f"✓ Auto-segmentation applied to: {base_name}\n\n"
                                      f"Remaining files in queue: {len(self.auto_segment_queue)}\n"
                                      f"Click 'Next File' to continue with auto-segmentation.")
                else:
                    messagebox.showinfo("Auto-Segmentation Complete", 
                                      f"✓ Auto-segmentation applied to: {base_name}\n\n"
                                      f"All files in the queue have been processed.\n"
                                      f"Auto-segmentation is now disabled.")
            else:
                print(f"Debug: File {os.path.basename(current_file)} not found in queue or queue doesn't exist")
                print(f"Debug: Looking for exact match: {current_file}")
                if hasattr(self, 'auto_segment_queue'):
                    print(f"Debug: Current queue has {len(self.auto_segment_queue)} files")
                    for i, file_path in enumerate(self.auto_segment_queue):
                        print(f"  Queue[{i}]: {os.path.basename(file_path)} -> {file_path}")
                        print(f"    Match check: {current_file == file_path}")
                else:
                    print("Debug: No auto_segment_queue attribute found")
            
            # Restore original opacity after a delay
            self.root.after(1000, lambda: setattr(self, 'annotation_opacity', original_opacity))
            
        except Exception as e:
            logger.error(f"Error applying auto-segmentation: {str(e)}")
            print(f"Debug Error: Could not apply auto-segmentation: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_queue_diagnostic(self):
        """Show diagnostic information about the current queue and directory"""
        diagnostic_window = tk.Toplevel(self.root)
        diagnostic_window.title("Queue Diagnostic")
        diagnostic_window.geometry("600x400")
        diagnostic_window.transient(self.root)
        
        frame = ttk.Frame(diagnostic_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)
        
        # Add diagnostic information
        text.insert(tk.END, "=== QUEUE DIAGNOSTIC INFORMATION ===\n\n")
        
        # Directory information
        text.insert(tk.END, f"Current Directory: {self.current_directory}\n")
        text.insert(tk.END, f"Current File Index: {self.current_file_idx}\n")
        text.insert(tk.END, f"Total Files in Directory: {len(self.directory_files) if hasattr(self, 'directory_files') else 'None'}\n\n")
        
        # List all files in directory
        if hasattr(self, 'directory_files') and self.directory_files:
            text.insert(tk.END, "All Files in Directory:\n")
            for i, file_path in enumerate(self.directory_files):
                current_marker = " <- CURRENT" if i == self.current_file_idx else ""
                text.insert(tk.END, f"  [{i}] {os.path.basename(file_path)}{current_marker}\n")
            text.insert(tk.END, "\n")
        
        # Queue information
        text.insert(tk.END, f"Auto-Segment Model Ready: {getattr(self, 'auto_segment_model_ready', False)}\n")
        text.insert(tk.END, f"Queue Size: {len(self.auto_segment_queue) if hasattr(self, 'auto_segment_queue') else 'No queue'}\n\n")
        
        # List queue contents
        if hasattr(self, 'auto_segment_queue') and self.auto_segment_queue:
            text.insert(tk.END, "Files in Auto-Segmentation Queue:\n")
            for i, file_path in enumerate(self.auto_segment_queue):
                text.insert(tk.END, f"  [{i}] {os.path.basename(file_path)}\n")
                text.insert(tk.END, f"      Full path: {file_path}\n")
            text.insert(tk.END, "\n")
        else:
            text.insert(tk.END, "No files in auto-segmentation queue\n\n")
        
        # Current image information
        text.insert(tk.END, f"Current Image Path: {self.image_paths[0] if self.image_paths else 'None'}\n")
        text.insert(tk.END, f"Number of Loaded Images: {len(self.original_images)}\n")
        text.insert(tk.END, f"Current Image Index: {self.current_image_index}\n\n")
        
        # Next file prediction
        if hasattr(self, 'directory_files') and self.directory_files:
            if self.current_file_idx < len(self.directory_files) - 1:
                next_file = self.directory_files[self.current_file_idx + 1]
                text.insert(tk.END, f"Next File Would Be: {os.path.basename(next_file)}\n")
                if hasattr(self, 'auto_segment_queue'):
                    in_queue = next_file in self.auto_segment_queue
                    text.insert(tk.END, f"Next File in Queue: {in_queue}\n")
                    if in_queue:
                        text.insert(tk.END, "✓ Next file SHOULD be auto-segmented\n")
                    else:
                        text.insert(tk.END, "✗ Next file will NOT be auto-segmented\n")
            else:
                text.insert(tk.END, "No more files available\n")
        
        text.config(state=tk.DISABLED)
        
        # Close button
        close_button = ttk.Button(frame, text="Close", command=diagnostic_window.destroy)
        close_button.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = TissueSegmentationTool(root)
    root.mainloop() 