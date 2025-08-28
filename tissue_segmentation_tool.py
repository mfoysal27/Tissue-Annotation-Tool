import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import re
import sys
from pathlib import Path
from collections import deque
import time

# Import PyTorch modules
import torch
import torch.nn as nn
from torchvision import transforms
from oiffile import OifFile

import nd2
from nd2reader import ND2Reader

# Import for 3D data saving
try:
    import tifffile
    print('tifffile loaded for 3D export')
except ImportError:
    print('tifffile not available - 3D export will be limited')

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
        
        # Performance optimization variables
        self.update_scheduled = False
        self.last_update_time = 0
        self.update_interval = 50  # milliseconds between updates during drawing
        
        # Aggressive performance mode variables
        self.fast_mode = True  # Enable aggressive optimizations by default
        self.draw_count = 0
        self.update_every_n_draws = 5  # Only update every 5th draw operation
        
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
        
        # Multi-channel and raw data support
        self.raw_image_data = None  # Store raw multi-dimensional data
        self.current_channels = []  # List of available channels
        self.selected_channel = None  # Currently selected channel (None = all channels)
        self.channel_images = {}  # Dictionary storing images for each channel
        self.original_shape = None  # Original dimensions (Z, C, Y, X or similar)
        
        # Crop functionality
        self.crop_bounds = None  # (z_start, z_end, y_start, y_end, x_start, x_end)
        self.is_cropped = False
        self.original_raw_data = None  # Backup of uncropped data
        
        # Initialize the hardcoded segments and colors
        self.initialize_segments()
        
        # Set ultra-fast mode for maximum brush performance
        self.set_performance_mode("ultra_fast")
        
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
        
        supported_formats = "Supported formats: ND2, OIB, JPG, PNG, TIF"
        format_label = ttk.Label(frame, text=supported_formats, font=("Arial", 12))
        format_label.pack(pady=10)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.nd2 *.oib *.jpg *.jpeg *.png *.tif *.tiff"),
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
        
        # Initialize loading status
        loading_successful = False
        
        try:
            if file_ext == '.nd2':
                self.process_nd2_file(file_path)
                loading_successful = len(self.original_images) > 0
                
                if not loading_successful:
                    # Show additional troubleshooting options
                    if messagebox.askyesno("ND2 Loading Failed", 
                                           "Failed to load ND2 file. Would you like to see troubleshooting options?"):
                        self.show_nd2_troubleshooting()
                        return
            elif file_ext == '.oib':
                print('started loading oib')
                self.process_oib_file(file_path)
                loading_successful = len(self.original_images) > 0
                
                if not loading_successful:
                    messagebox.showerror("Error", "Failed to load OIB file")
                    return
            elif file_ext in ['.tif', '.tiff']:
                self.process_multipagetiff_image(file_path)
                loading_successful = len(self.original_images) > 0
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                self.process_regular_image(file_path)
                loading_successful = len(self.original_images) > 0
            else:
                messagebox.showerror("Error", f"Unsupported file format: {file_ext}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            return
        
        if not loading_successful:
            messagebox.showerror("Error", "Failed to load images")
            return
        
        self.create_annotation_window()
    
    def get_image_files_in_directory(self, directory):
        """Find all image files in the directory"""
        image_extensions = ['.nd2', '.oib', '.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_files = []
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory, file))
        
        # Sort files by name
        image_files.sort()
        return image_files
    def process_multipagetiff_image(self, file_path):
        """Process multipage TIFF file with multi-channel and multi-frame support"""
        try:
            self.original_images = []
            error_details = []
            
            # Create a progress window for analysis
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Analyzing Multipage TIFF File")
            analysis_window.geometry("400x150")
            analysis_window.transient(self.root)
            analysis_window.grab_set()
            
            progress_label = ttk.Label(analysis_window, text="Analyzing TIFF file structure...")
            progress_label.pack(pady=10)
            
            status_label = ttk.Label(analysis_window, text="", wraplength=380)
            status_label.pack(pady=5)
            
            progress_bar = ttk.Progressbar(analysis_window, orient="horizontal", length=300, mode="indeterminate")
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Update the progress window
            analysis_window.update()
            
            # First analyze file to determine structure
            total_frames = 0
            total_channels = 1
            frame_info = {}
            image_shape = None
            
            try:
                status_label.config(text="Reading TIFF file structure...")
                analysis_window.update()
                
                import tifffile
                
                # Try to load the full image to get shape information
                with tifffile.TiffFile(file_path) as tif:
                    # Get number of pages
                    # total_frames = len(tif.pages)
                    total_frames= len(tif.pages)
                    # print(f"TIFF File Pages: {total_frames}")
                    

                    
                    # Get shape from first page
                    first_page = tif.pages[0]
                    page_shape = first_page.shape
                    # print('page shapoe  infor ' , page_shape)
                    
                    # Try to read the whole data to get full shape
                    try:
                        image_data = tif.asarray()
                        image_shape = image_data.shape
                        # print(f"TIFF Image4 shape: {image_shape}")
                        
                        # Determine if this is a multi-channel image based on shape
                        if len(image_shape) > 3:
                            # Could be (T, C, Y, X) or similar
                            total_channels = image_shape[0] if image_shape[0] <= 10 else 1  # Assume first dim is channels if reasonable
                        
                        frame_info['shape'] = image_shape
                        frame_info['raw_data'] = image_data
                    except Exception as e:
                        print(f"Error reading full TIFF data: {str(e)}")
                        # If reading full data fails, use page info
                        image_shape = (total_frames,) + page_shape
                        frame_info['shape'] = image_shape
                        frame_info['pages_only'] = True
                    
            except Exception as e:
                print(f"Error analyzing TIFF file: {str(e)}")
                error_details.append(f"Analysis error: {str(e)}")
            
            # Close analysis window
            try:
                analysis_window.destroy()
            except:
                pass
            
            # If no frames found, show error
            # print('total frame', total_frames)
            # if total_frames == 0:
            #     messagebox.showerror("Error", "Could not determine frame count in TIFF file")
            #     return
            
            # Create selection dialog for frames and channels
            selection_dialog = tk.Toplevel(self.root)
            selection_dialog.title("Multipage TIFF Selection")
            selection_dialog.geometry("450x500")
            selection_dialog.transient(self.root)
            selection_dialog.grab_set()
            
            frame = ttk.Frame(selection_dialog, padding="20")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # File info
            info_text = f"TIFF file contains:\n• Shape: {image_shape}\n• Estimated frames: {total_frames}"
            if total_channels > 1:
                info_text += f"\n• Estimated channels: {total_channels}"
            
            info_label = ttk.Label(frame, text=info_text, font=("Arial", 10, "bold"))
            info_label.pack(pady=10)
            
            # Frame range selection
            if total_frames > 1:
                frame_frame = ttk.LabelFrame(frame, text="Frame Range")
                frame_frame.pack(pady=10, fill=tk.X)
                
                load_all_frames_var = tk.BooleanVar(value=True)
                load_all_frames_cb = ttk.Checkbutton(
                    frame_frame, 
                    text="Load all frames",
                    variable=load_all_frames_var
                )
                load_all_frames_cb.pack(pady=5, anchor=tk.W)
                
                range_grid = ttk.Frame(frame_frame)
                range_grid.pack(pady=10, padx=10, fill=tk.X)
                
                ttk.Label(range_grid, text="Start Frame:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
                ttk.Label(range_grid, text="End Frame:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
                
                start_frame_var = tk.StringVar(value="0")
                end_frame_var = tk.StringVar(value=str(total_frames - 1))
                
                start_frame_entry = ttk.Entry(range_grid, textvariable=start_frame_var, width=10)
                start_frame_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
                
                end_frame_entry = ttk.Entry(range_grid, textvariable=end_frame_var, width=10)
                end_frame_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
                
                def toggle_frame_inputs():
                    state = "disabled" if load_all_frames_var.get() else "normal"
                    start_frame_entry.config(state=state)
                    end_frame_entry.config(state=state)
                
                load_all_frames_cb.config(command=toggle_frame_inputs)
                toggle_frame_inputs()
            else:
                load_all_frames_var = tk.BooleanVar(value=True)
                start_frame_var = tk.StringVar(value="0")
                end_frame_var = tk.StringVar(value="0")
            
            # Channel selection
            if total_channels > 1:
                channel_frame = ttk.LabelFrame(frame, text="Channel Selection")
                channel_frame.pack(pady=10, fill=tk.X)
                
                channel_var = tk.StringVar(value="all")
                
                ttk.Radiobutton(channel_frame, text="All channels (combined)", 
                               variable=channel_var, value="all").pack(anchor=tk.W, pady=2)
                
                for i in range(total_channels):
                    ttk.Radiobutton(channel_frame, text=f"Channel {i}", 
                                   variable=channel_var, value=str(i)).pack(anchor=tk.W, pady=2)
            else:
                channel_var = tk.StringVar(value="all")
            
            # Buttons
            button_frame = ttk.Frame(selection_dialog)
            button_frame.pack(side=tk.BOTTOM, pady=15, fill=tk.X)
            
            load_button = tk.Button(
                button_frame, 
                text="LOAD TIFF", 
                bg="#007bff",
                fg="white",
                font=("Arial", 11, "bold"),
                relief=tk.RAISED,
                borderwidth=2,
                padx=15,
                pady=8,
                cursor="hand2"
            )
            load_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
            
            cancel_button = ttk.Button(
                button_frame, 
                text="Cancel", 
                command=lambda: selection_dialog.destroy()
            )
            cancel_button.pack(side=tk.RIGHT, padx=10, pady=10)
            
            # Load button validation and execution
            def validate_and_load():
                try:

                    start_frame = 0
                    end_frame = total_frames-1
                    
                    # print('frames', start_frame, end_frame)

                    
                    selection_dialog.destroy()
                    
                    # Load the frames
                    self.load_tiff_frames(file_path, frame_info, start_frame, end_frame)
                    

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load TIFF: {str(e)}")
                    selection_dialog.destroy()
            
            load_button.config(command=validate_and_load)
            
            # Center dialog
            selection_dialog.update_idletasks()
            screen_width = selection_dialog.winfo_screenwidth()
            screen_height = selection_dialog.winfo_screenheight()
            width = selection_dialog.winfo_width()
            height = selection_dialog.winfo_height()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            selection_dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            # Wait for user interaction
            self.root.wait_window(selection_dialog)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process multipage TIFF file: {str(e)}")
            # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0
    
    # def load_tiff_frames(self, file_path, frame_info, start_frame, end_frame):
        # """Load the selected frames and channels from the TIFF file"""
        # try:
        #     # Create progress window for loading
        #     progress_window = tk.Toplevel(self.root)
        #     progress_window.title("Loading TIFF Frames")
        #     progress_window.geometry("400x150")
        #     progress_window.transient(self.root)
        #     progress_window.grab_set()
            
        #     progress_label = ttk.Label(progress_window, text="Loading frames, please wait...")
        #     progress_label.pack(pady=10)
            
        #     status_label = ttk.Label(progress_window, text="", wraplength=380)
        #     status_label.pack(pady=5)
            
        #     progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
        #     progress_bar.pack(pady=10)
            
        #     # Update the progress window
        #     progress_window.update()
            
        #     success = False
            
        #     try:
        #         import tifffile
                
        #         raw_data = frame_info['raw_data']
        #         image_shape = frame_info['shape']
        #         # Extract frame based on shape
        #         if len(image_shape) == 2:  # Single 2D image
        #             frame_data = raw_data
        #         elif len(image_shape) == 3:  # (X, Y, C)
        #             frame_data = raw_data[:, :, :]
        #             print('shape of frame1 data', frame_data.shape)

        #         else:
        #             # Fallback for unknown dimensions
        #             frame_data = raw_data[1] if len(raw_data.shape) > 2 else raw_data
        #             print('shape of frame3 data', frame_data.shape)

                
                
        #                         # Process frame using existing method
        #         img = self.process_multipagetiff_image(file_path)
        #         if img:
        #             self.original_images.append(img)
        #             success = True
                
        #     except Exception as e:
        #         print(f"Error loading TIFF frame {i}: {str(e)}")
                # continue














    def load_tiff_frames(self, file_path, frame_info, start_frame, end_frame):
        """Load the selected frames and channels from the TIFF file"""
        try:
            # Create progress window for loading
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Loading TIFF Frames")
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
            
            success = False
            
            try:
                import tifffile
                
                raw_data = frame_info['raw_data']
                image_shape = frame_info['shape']

    # Store raw data for crop functionality
                self.file_path=file_path
                self.raw_image_data = raw_data
                self.original_shape = image_shape
                self.original_raw_data = raw_data.copy()
                
                # Extract frame based on shape
                if len(image_shape) == 2:  # Single 2D image
                    # Process as a single 2D image
                    img = self.process_nd2_array(raw_data)
                    if img:
                        self.original_images.append(img)
                        success = True
                elif len(image_shape) == 3:  # (Z, Y, X) or (Y, X, C)
                    if image_shape[2] in [3, 4]:  # Likely (Y, X, C) - RGB or RGBA image
                        img = self.process_nd2_array(raw_data)
                        if img:
                            self.original_images.append(img)
                            success = True
                    else:  # Likely (Z, Y, X) - multiple grayscale frames
                        for i in range(image_shape[0]):
                            img = self.process_nd2_array(raw_data[i])
                            if img:
                                self.original_images.append(img)
                                success = True
                elif len(image_shape) == 4:  # (Z, C, Y, X) or similar
                    for i in range(image_shape[0]):
                        img = self.process_nd2_array(raw_data[i])
                        if img:
                            self.original_images.append(img)
                            success = True
                else:
                    # Fallback for unknown dimensions
                    img = self.process_nd2_array(raw_data[0] if len(raw_data.shape) > 2 else raw_data)
                    if img:
                        self.original_images.append(img)
                        success = True
                    
            except Exception as e:
                print(f"Error loading TIFF data: {str(e)}")
            
            # Close progress window
            progress_window.destroy()
            
            # Check if we got any images
            if not self.original_images:
                messagebox.showerror("Error", "Failed to load any frames from the TIFF file")
                return
            
            # Initialize segmentation masks for successful images
            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0)) for img in self.original_images]
            self.current_image_index = 0


        
            # Show success message
            messagebox.showinfo("Success", f"Successfully loaded {len(self.original_images)} frames from TIFF file")
            
        except Exception as e:
            # Show error message
            messagebox.showerror("Error", f"Failed to load TIFF frames: {str(e)}")
            
            # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0








            #     # Setup progress tracking
            #     num_frames_to_load = end_frame - start_frame + 1
            #     print('number of frames', num_frames_to_load)
            #     progress_bar['maximum'] = num_frames_to_load
            #     progress_bar['value'] = 0
            #     progress_window.update()
                
            #     # Check if we have raw data or need to read pages
            #     if 'raw_data' in frame_info and not frame_info.get('pages_only', False):
            #         # Get the raw data from frame_info
            #         raw_data = frame_info['raw_data']
            #         image_shape = frame_info['shape']
                    
            #         # Store raw data for crop functionality
            #         self.raw_image_data = raw_data
            #         self.original_shape = image_shape
            #         self.original_raw_data = raw_data.copy()
                    
            #         status_label.config(text=f"Processing frames {start_frame} to {end_frame}...")
            #         progress_window.update()
                    
            #         # Process frames based on dimension structure
            #         for i in range(start_frame, end_frame + 1):
            #             try:
            #                 # Update progress
            #                 frame_index = i - start_frame
            #                 progress_label.config(text=f"Loading frame {frame_index+1}/{num_frames_to_load}...")
            #                 progress_bar['value'] = frame_index + 1
            #                 progress_window.update()
                            
            #                 # Extract frame based on shape
            #                 if len(image_shape) == 2:  # Single 2D image
            #                     frame_data = raw_data
            #                 elif len(image_shape) == 3:  # (X, Y, C)
            #                     frame_data = raw_data[:, :, :]
            #                     print('shape of frame1 data', frame_data.shape)

            #                 else:
            #                     # Fallback for unknown dimensions
            #                     frame_data = raw_data[1] if len(raw_data.shape) > 2 else raw_data
            #                     print('shape of frame3 data', frame_data.shape)
                                
            #                 # Process frame using existing method
            #                 img = self.process_multipagetiff_image(file_path)
            #                 if img:
            #                     self.original_images.append(img)
            #                     success = True
                            
            #             except Exception as e:
            #                 print(f"Error loading TIFF frame {i}: {str(e)}")
            #                 continue
            #     else:
            #         # Read pages directly from file
            #         with tifffile.TiffFile(file_path) as tif:
            #             for i in range(start_frame, min(end_frame + 1, len(tif.pages))):
            #                 try:
            #                     # Update progress
            #                     frame_index = i - start_frame
            #                     progress_label.config(text=f"Loading frame {frame_index+1}/{num_frames_to_load}...")
            #                     progress_bar['value'] = frame_index + 1
            #                     progress_window.update()
                                
            #                     # Read the page
            #                     page = tif.pages[i]
            #                     frame_data = page.asarray()
                                
            #                     # Handle channel selection if it's an ImageJ hyperstack
            #                     if tif.is_imagej and selected_channel != "all":
            #                         try:
            #                             metadata = tif.imagej_metadata
            #                             if 'channels' in metadata and metadata['channels'] > 1:
            #                                 c_idx = int(selected_channel)
            #                                 # Need to calculate the right page based on channel
            #                                 # This is complex and depends on ImageJ hyperstack organization
            #                                 # Simplified approach for now
            #                                 pass
            #                         except (ValueError, IndexError):
            #                             pass
                                
            #                     # Process frame
            #                     img = self.process_multipagetiff_image(frame_data)
            #                     # img = frame_data

            #                     print('shape of multipage tiff', img.shape)
            #                     if img:
            #                         self.original_images.append(img)
            #                         success = True
            #                 except Exception as e:
            #                     print(f"Error loading TIFF page {i}: {str(e)}")
            #                     continue
                
            # except Exception as e:
            #     print(f"Error processing TIFF data: {str(e)}")
            
            # # Close progress window
            # try:
            progress_window.destroy()
            # except:
            #     pass
            
            # Check if we got any images
            if not self.original_images:
                messagebox.showerror("Error", "Failed to load any frames from the TIFF file")
                return
            
            # Initialize segmentation masks for successful images
            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0)) for img in self.original_images]
            self.current_image_index = 0
            
            # Show success message
            messagebox.showinfo("Success", f"Successfully loaded {len(self.original_images)} frames from TIFF file")
            
        except Exception as e:
            # Show error message
            messagebox.showerror("Error", f"Failed to load TIFF frames: {str(e)}")
            
            # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0
    
    def process_oib_file(self, file_path):
        """Process OIB file with enhanced multi-channel and multi-frame support"""
        try:
            self.original_images = []
            error_details = []
            
            # Create a progress window for analysis
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Analyzing OIB File")
            analysis_window.geometry("400x150")
            analysis_window.transient(self.root)
            analysis_window.grab_set()
            
            progress_label = ttk.Label(analysis_window, text="Analyzing OIB file structure...")
            progress_label.pack(pady=10)
            
            status_label = ttk.Label(analysis_window, text="", wraplength=380)
            status_label.pack(pady=5)
            
            progress_bar = ttk.Progressbar(analysis_window, orient="horizontal", length=300, mode="indeterminate")
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Update the progress window
            analysis_window.update()
            
            # First analyze file to determine structure
            total_frames = 0
            total_channels = 1
            frame_info = {}
            image_shape = None
            
            try:
                status_label.config(text="Reading OIB file structure...")
                analysis_window.update()
                
                from oiffile import imread
                
                # Try to load the full image to get shape information
                with OifFile(file_path) as oib:
                    # Get file info
                    file_info = oib.mainfile.get('File Info', {})
                    print(f"OIB File Info: {file_info}")
                    
                    # Try to read the image to get dimensions
                    image_data = imread(file_path)
                    image_shape = image_data.shape
                    print(f"OIB Image shape: {image_shape}")
                    
                    # Determine frame and channel count based on shape
                    if len(image_shape) >= 3:
                        # Multi-dimensional data
                        if len(image_shape) == 3:
                            # Could be (Z, Y, X) or (C, Y, X) or (T, Y, X)
                            total_frames = image_shape[0]
                            frame_info['dimension'] = 'z'  # Assume Z-stack
                        elif len(image_shape) == 4:
                            # Could be (T, C, Y, X) or (C, Z, Y, X) or (T, Z, Y, X)
                            total_frames = image_shape[1]
                            total_channels = image_shape[0]
                            frame_info['dimension'] = 'tz'  # Time and Z or Time and Channel
                        elif len(image_shape) == 5:
                            # Likely (T, C, Z, Y, X)
                            total_frames = image_shape[0] * image_shape[2]  # T * Z
                            total_channels = image_shape[1]
                            frame_info['dimension'] = 'tczyx'
                    else:
                        # 2D image
                        total_frames = 1
                        total_channels = 1
                    
                    frame_info['shape'] = image_shape
                    frame_info['raw_data'] = image_data
                    
            except Exception as e:
                print(f"Error analyzing OIB file: {str(e)}")
                error_details.append(f"Analysis error: {str(e)}")
            
            # Close analysis window
            try:
                analysis_window.destroy()
            except:
                pass
            
            # If no frames found, show error
            if total_frames == 0:
                messagebox.showerror("Error", "Could not determine frame count in OIB file")
                return
            
            # Create selection dialog for frames and channels
            selection_dialog = tk.Toplevel(self.root)
            selection_dialog.title("OIB File Selection")
            selection_dialog.geometry("450x500")
            selection_dialog.transient(self.root)
            selection_dialog.grab_set()
            
            frame = ttk.Frame(selection_dialog, padding="20")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # File info
            info_text = f"OIB file contains:\n• Shape: {image_shape}\n• Estimated frames: {total_frames}"
            if total_channels > 1:
                info_text += f"\n• Estimated channels: {total_channels}"
            
            info_label = ttk.Label(frame, text=info_text, font=("Arial", 10, "bold"))
            info_label.pack(pady=10)
            
            # Frame range selection
            if total_frames > 1:
                frame_frame = ttk.LabelFrame(frame, text="Frame Range")
                frame_frame.pack(pady=10, fill=tk.X)
                
                load_all_frames_var = tk.BooleanVar(value=True)
                load_all_frames_cb = ttk.Checkbutton(
                    frame_frame, 
                    text="Load all frames",
                    variable=load_all_frames_var
                )
                load_all_frames_cb.pack(pady=5, anchor=tk.W)
                
                range_grid = ttk.Frame(frame_frame)
                range_grid.pack(pady=10, padx=10, fill=tk.X)
                
                ttk.Label(range_grid, text="Start Frame:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
                ttk.Label(range_grid, text="End Frame:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
                
                start_frame_var = tk.StringVar(value="0")
                end_frame_var = tk.StringVar(value=str(total_frames - 1))
                
                start_frame_entry = ttk.Entry(range_grid, textvariable=start_frame_var, width=10)
                start_frame_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
                
                end_frame_entry = ttk.Entry(range_grid, textvariable=end_frame_var, width=10)
                end_frame_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
                
                def toggle_frame_inputs():
                    state = "disabled" if load_all_frames_var.get() else "normal"
                    start_frame_entry.config(state=state)
                    end_frame_entry.config(state=state)
                
                load_all_frames_cb.config(command=toggle_frame_inputs)
                toggle_frame_inputs()
            else:
                load_all_frames_var = tk.BooleanVar(value=True)
                start_frame_var = tk.StringVar(value="0")
                end_frame_var = tk.StringVar(value="0")
            
            # Channel selection
            if total_channels > 1:
                channel_frame = ttk.LabelFrame(frame, text="Channel Selection")
                channel_frame.pack(pady=10, fill=tk.X)
                
                channel_var = tk.StringVar(value="all")
                
                ttk.Radiobutton(channel_frame, text="All channels (combined)", 
                               variable=channel_var, value="all").pack(anchor=tk.W, pady=2)
                
                for i in range(total_channels):
                    ttk.Radiobutton(channel_frame, text=f"Channel {i}", 
                                   variable=channel_var, value=str(i)).pack(anchor=tk.W, pady=2)
            else:
                channel_var = tk.StringVar(value="all")
            
            # Buttons
            button_frame = ttk.Frame(selection_dialog)
            button_frame.pack(side=tk.BOTTOM, pady=15, fill=tk.X)
            
            load_button = tk.Button(
                button_frame, 
                text="LOAD OIB", 
                bg="#007bff",
                fg="white",
                font=("Arial", 11, "bold"),
                relief=tk.RAISED,
                borderwidth=2,
                padx=15,
                pady=8,
                cursor="hand2"
            )
            load_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
            
            cancel_button = ttk.Button(
                button_frame, 
                text="Cancel", 
                command=lambda: selection_dialog.destroy()
            )
            cancel_button.pack(side=tk.RIGHT, padx=10, pady=10)
            
            # Load button validation and execution
            def validate_and_load():
                try:
                    # Get frame range
                    if total_frames > 1 and not load_all_frames_var.get():
                        start_frame = int(start_frame_var.get())
                        end_frame = int(end_frame_var.get())
                        
                        if start_frame < 0 or start_frame >= total_frames:
                            messagebox.showerror("Error", f"Start frame must be between 0 and {total_frames-1}")
                            return
                        if end_frame < 0 or end_frame >= total_frames:
                            messagebox.showerror("Error", f"End frame must be between 0 and {total_frames-1}")
                            return
                        if start_frame > end_frame:
                            messagebox.showerror("Error", "Start frame must be less than or equal to end frame")
                            return
                    else:
                        start_frame = 0
                        end_frame = total_frames - 1
                    
                    # Get channel selection
                    selected_channel = channel_var.get()
                    
                    selection_dialog.destroy()
                    
                    # Load the frames
                    self.load_oib_frames(file_path, frame_info, start_frame, end_frame, selected_channel)
                    
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid frame numbers")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load OIB: {str(e)}")
            
            load_button.config(command=validate_and_load)
            
            # Center dialog
            selection_dialog.update_idletasks()
            screen_width = selection_dialog.winfo_screenwidth()
            screen_height = selection_dialog.winfo_screenheight()
            width = selection_dialog.winfo_width()
            height = selection_dialog.winfo_height()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            selection_dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            # Wait for user interaction
            self.root.wait_window(selection_dialog)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process OIB file: {str(e)}")
                         # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0
    
    def load_oib_frames(self, file_path, frame_info, start_frame, end_frame, selected_channel):
        """Load the selected frames and channels from the OIB file"""
        try:
            # Create progress window for loading
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Loading OIB Frames")
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
            
            success = False
            
            try:
                # Get the raw data from frame_info
                raw_data = frame_info['raw_data']
                image_shape = frame_info['shape']
                dimension = frame_info['dimension']
                
                # Store raw data for crop functionality
                self.file_path=file_path
                self.raw_image_data = raw_data
                self.original_shape = image_shape
                self.original_raw_data = raw_data.copy()
                
                status_label.config(text=f"Processing frames {start_frame} to {end_frame}...")
                progress_window.update()
                
                # Setup progress tracking
                num_frames_to_load = end_frame - start_frame + 1
                progress_bar['maximum'] = num_frames_to_load
                progress_bar['value'] = 0
                progress_window.update()
                
                # Process frames based on dimension structure
                for i in range(start_frame, end_frame + 1):
                    try:
                        # Update progress
                        frame_index = i - start_frame
                        progress_label.config(text=f"Loading frame {frame_index+1}/{num_frames_to_load}...")
                        progress_bar['value'] = frame_index + 1
                        progress_window.update()
                        
                        # Extract frame based on shape
                        if len(image_shape) == 2:
                            # 2D image
                            frame_data = raw_data
                        elif len(image_shape) == 3:
                            # 3D: (Z, Y, X) or (C, Y, X) or (T, Y, X)
                            if i < image_shape[0]:
                                frame_data = raw_data[i]
                            else:
                                continue
                        elif len(image_shape) == 4:
                            # 4D: (T, C, Y, X) or (C, Z, Y, X) or (T, Z, Y, X)
                            t_idx = i // image_shape[1] if i < image_shape[0] * image_shape[1] else 0
                            c_idx = i % image_shape[1]
                            
                            if t_idx < image_shape[0] and c_idx < image_shape[1]:
                                frame_data = raw_data[t_idx, c_idx]
                            else:
                                continue
                        elif len(image_shape) == 5:
                            # 5D: (T, C, Z, Y, X)
                            t_idx = i // (image_shape[1] * image_shape[2])
                            remaining = i % (image_shape[1] * image_shape[2])
                            c_idx = remaining // image_shape[2]
                            z_idx = remaining % image_shape[2]
                            
                            if (t_idx < image_shape[0] and c_idx < image_shape[1] and z_idx < image_shape[2]):
                                frame_data = raw_data[t_idx, c_idx, z_idx]
                            else:
                                continue
                        else:
                            # Fallback: take first slice
                            frame_data = raw_data[0] if len(raw_data.shape) > 2 else raw_data
                        
                        # Handle channel selection
                        if selected_channel != "all" and len(frame_data.shape) >= 3:
                            try:
                                channel_idx = int(selected_channel)
                                if channel_idx < frame_data.shape[-1]:  # Channels usually last dimension
                                    frame_data = frame_data[:, :, channel_idx]
                                elif channel_idx < frame_data.shape[0]:  # Or first dimension
                                    frame_data = frame_data[channel_idx]
                            except (ValueError, IndexError):
                                pass  # Keep all channels if selection fails
                        
                        # Process frame using existing method
                        img = self.process_nd2_array(frame_data)
                        if img:
                            self.original_images.append(img)
                            success = True
                        
                    except Exception as e:
                        print(f"Error loading OIB frame {i}: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error processing OIB data: {str(e)}")
            
            # Close progress window
            try:
                progress_window.destroy()
            except:
                pass
            
            # Check if we got any images
            if not self.original_images:
                messagebox.showerror("Error", "Failed to load any frames from the OIB file")
                return
            
            # Initialize segmentation masks for successful images
            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0)) for img in self.original_images]
            self.current_image_index = 0
            
            # Show success message
            messagebox.showinfo("Success", f"Successfully loaded {len(self.original_images)} frames from OIB file")
            
        except Exception as e:
            # Show error message
            messagebox.showerror("Error", f"Failed to load OIB frames: {str(e)}")
            
            # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0

    def process_nd2_file(self, file_path):
        try:
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
            total_frames = 0
            frame_dimension = None
            
            # Try to get frame count with nd2 package

            try:
                status_label.config(text="Analyzing ND2 file dimensions...")
                analysis_window.update()
                
                with nd2.ND2File(file_path) as f:                        
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
            except Exception as e:
                pass
            
            # If nd2 package failed, try nd2reader
            if total_frames == 0:
                try:
                    status_label.config(text="Analyzing with nd2reader...")
                    analysis_window.update()
                    
                    with ND2Reader(file_path) as images:
                        total_frames = len(images)
                except Exception as e:
                    pass
            
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
                        return False
                    if e < 0 or e >= total_frames:
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
            
            # Wait for user to interact with dialog
            self.root.wait_window(frame_dialog)
            
            # If dialog was closed without loading, return
            if not dialog_result[0]:
                return
                        # Start the loading process
            self.load_nd2_frames(file_path, frame_info, start_frame, end_frame)
            # Continue with actual frame loading here...
            # (This section would need proper implementation)
            # For now, just initialize empty
        except Exception as e:
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
                                    success = True
                                else:
                                    return
                            except Exception as e:
                                error_msg = f"Error loading frame {i}: {str(e)}"
                                error_details.append(error_msg)
                                continue
                except Exception as e:
                    error_msg = f"nd2 package load error: {str(e)}"
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
                                    success = True
                                else:
                                    pass
                            except Exception as e:
                                error_msg = f"Error loading frame {i}: {str(e)}"
                                error_details.append(error_msg)
                                continue
                except Exception as e:
                    error_msg = f"nd2reader error: {str(e)}"
                    error_details.append(error_msg)
            
            # Try using a different approach if both methods failed
            if not success:
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

                                                success = True
                                        except Exception as e:

                                            continue
                                else:
                                    # Single image
                                    img = self.process_nd2_array(raw_data)
                                    if img:
                                        self.original_images.append(img)

                                        success = True
                                
                                # If we got images, break out of coordinate system loop
                                if self.original_images:
                                    break
                                    
                            except Exception as e:

                                continue
                except Exception as e:
                    error_msg = f"Raw data approach error: {str(e)}"

                    error_details.append(error_msg)
            
            # Close progress window
            try:
                progress_window.destroy()
            except:
                pass
            
            # Check if we got any images
            if not self.original_images:
                # Show error with details

                
                error_details_str = "\n".join(error_details)

                
                messagebox.showerror("Error", "Failed to load any slices from the ND2 file. Check the log for details.")
                return
            
            # Initialize segmentation masks for successful images
            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0)) for img in self.original_images]
            self.current_image_index = 0
            
            # Show success message
            messagebox.showinfo("Success", f"Successfully loaded {len(self.original_images)} frames from ND2 file")
            
        except Exception as e:


            
            # Show error message
            messagebox.showerror("Error", f"Failed to load ND2 frames: {str(e)}")
            
            # Initialize empty
            self.original_images = []
            self.segmentation_masks = []
            self.current_image_index = 0
    
    def process_regular_image(self, file_path):
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
                        
                        # Show user-friendly message only once per session or if significant annotations
                        if non_transparent > 100:  # Only show if substantial annotations
                            messagebox.showinfo("Auto-Segmented Image Loaded", 
                                              f"✓ Loaded auto-segmented image with {non_transparent} annotated pixels.\n"
                                              "You can modify the annotations using the paint tools and adjust opacity.")
                        return
                    except Exception as e:
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
            self.current_image_index = 0
            
            # Check for existing annotation files
            base_name_no_ext = os.path.splitext(base_name)[0]
            directory = os.path.dirname(file_path)
            
            # Look for segmentation files with common naming patterns
            possible_annotation_files = [
                os.path.join(directory, f"segmented_{base_name_no_ext}.png"),
                os.path.join(directory, f"{base_name_no_ext}_segmented.png"),
                os.path.join(directory, f"{base_name_no_ext}_annotations.png"),
                os.path.join(directory, f"{base_name_no_ext}_mask.png"),
            ]
            
            found_annotation = None
            for annotation_file in possible_annotation_files:
                if os.path.exists(annotation_file):
                    found_annotation = annotation_file
                    break
            
            # Handle existing annotations
            if found_annotation:
                # Ask user if they want to load existing annotations
                load_existing = messagebox.askyesno(
                    "Existing Annotations Found", 
                    f"Found existing annotation file:\n{os.path.basename(found_annotation)}\n\n"
                    f"Would you like to load these annotations?\n\n"
                    f"✓ Yes - Load existing annotations for editing/training\n"
                    f"✗ No - Start with blank annotations"
                )
                
                if load_existing:
                    try:
                        # Load the annotation file
                        annotation_img = Image.open(found_annotation)
                        
                        # Convert segmentation image back to mask format
                        mask = self.convert_segmentation_to_mask(annotation_img, img.size)
                        
                        if mask is not None:
                            self.segmentation_masks = [mask]
                            
                            # Count non-transparent pixels
                            mask_array = np.array(mask)
                            non_transparent = np.sum(mask_array[:, :, 3] > 0) if len(mask_array.shape) == 3 and mask_array.shape[2] == 4 else 0
                            
                            # Update window title
                            
                            messagebox.showinfo("Annotations Loaded", 
                                              f"✓ Successfully loaded {non_transparent} annotated pixels.\n\n"
                                              f"You can now:\n"
                                              f"• Edit the existing annotations\n"
                                              f"• Use them to train auto-segmentation for other images\n"
                                              f"• Save modified annotations")
                        else:
                            # Fall back to empty mask if conversion failed
                            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0))]
                            messagebox.showwarning("Annotation Load Failed", 
                                                 f"Could not convert the annotation file to the expected format.\n"
                                                 f"Starting with blank annotations.")
                    except Exception as e:
                        self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0))]
                        messagebox.showerror("Error Loading Annotations", 
                                           f"Failed to load annotation file:\n{str(e)}\n\n"
                                           f"Starting with blank annotations.")
                else:
                    # User chose not to load existing annotations
                    self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0))]
            else:
                # No existing annotations found
                self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0))]
                # Update window title for regular images
    
    def convert_segmentation_to_mask(self, segmentation_img, target_size):
        """Convert a saved segmentation image back to RGBA mask format"""
        try:
            # Ensure the segmentation image is the right size
            if segmentation_img.size != target_size:
                segmentation_img = segmentation_img.resize(target_size, Image.NEAREST)
            
            # Convert to RGB if not already
            if segmentation_img.mode != 'RGB':
                segmentation_img = segmentation_img.convert('RGB')
            
            # Create RGBA mask
            mask = Image.new('RGBA', target_size, (0, 0, 0, 0))
            mask_array = np.array(mask)
            seg_array = np.array(segmentation_img)
            
            # Map each segment color back to the mask
            pixels_converted = 0
            for segment, color in self.segment_colors.items():
                r, g, b, a = color
                
                # Find pixels matching this segment's color
                matching_pixels = np.all(seg_array == np.array([r, g, b]), axis=2)
                pixel_count = np.sum(matching_pixels)
                
                if pixel_count > 0:
                    # Set those pixels in the mask with full alpha
                    mask_array[matching_pixels] = [r, g, b, 255]
                    pixels_converted += pixel_count
            
            # Convert back to PIL Image
            converted_mask = Image.fromarray(mask_array, 'RGBA')
            return converted_mask
            
        except Exception as e:
            return None
    
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
        
        # Add navigation buttons at the top
        folder_frame = ttk.Frame(nav_frame)
        folder_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        load_image_button = ttk.Button(
            folder_frame, 
            text="Load New Image", 
            command=self.load_new_image,
            style="Action.TButton"
        )
        load_image_button.pack(side=tk.LEFT, padx=5)
        
        load_annotation_button = ttk.Button(
            folder_frame, 
            text="Load Annotation", 
            command=self.load_annotation,
            style="Action.TButton"
        )
        load_annotation_button.pack(side=tk.LEFT, padx=5)
        
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
        self.brush_size_label = ttk.Label(brush_frame, text=f"{self.brush_size}px")
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
        
        # Add a slider to control visible segments
        slider_frame = ttk.Frame(segment_frame)
        slider_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(slider_frame, text="Segment Page:").pack(side=tk.LEFT, padx=5)
        
        # Calculate how many segments to show per page
        segments_per_page = 8  # Adjust this value based on UI space
        total_pages = max(1, (len(self.segments) + segments_per_page - 1) // segments_per_page)
        
        self.segment_page_var = tk.IntVar(value=1)
        segment_slider = ttk.Scale(
            slider_frame,
            from_=1,
            to=total_pages,
            orient=tk.HORIZONTAL,
            variable=self.segment_page_var
        )
        segment_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Add a label showing the current page
        self.segment_page_label = ttk.Label(slider_frame, text=f"Page 1/{total_pages}")
        self.segment_page_label.pack(side=tk.LEFT, padx=5)
        
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
        
        # Function to update visible segments based on slider
        def update_visible_segments(event=None):
            # Clear current segment buttons
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
                
            # Calculate range of segments to show
            current_page = self.segment_page_var.get()
            start_idx = (current_page - 1) * segments_per_page
            end_idx = min(start_idx + segments_per_page, len(self.segments))
            
            # Update page label
            self.segment_page_label.config(text=f"Page {current_page}/{total_pages}")
            
            # Get segments for current page
            visible_segments = self.segments[start_idx:end_idx]
            
            # Create buttons for visible segments
            for i, segment in enumerate(visible_segments):
                segment_idx = self.segments.index(segment)
                
                # Create frame for segment button
                segment_frame = ttk.Frame(scrollable_frame)
                segment_frame.pack(fill=tk.X, pady=2, padx=3)
                
                # Get color for segment
                color = self.segment_colors[segment]
                color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
                # Add index prefix
                prefix = f"{start_idx + i + 1}."
                prefix_label = ttk.Label(segment_frame, text=prefix, width=3)
                prefix_label.pack(side=tk.LEFT)
            
                # Add color button
                color_button = tk.Button(
                    segment_frame,
                    bg=color_hex, 
                    width=2, 
                    height=1,
                    command=lambda idx=segment_idx: self.select_segment(idx)
                )
                color_button.pack(side=tk.LEFT, padx=5)
            
                # Add segment name (handle hierarchical segments)
                if " - " in segment:
                    # For subfeatures, show only the part after the dash
                    segment_name = segment.split(" - ")[1]
                    segment_label = ttk.Label(segment_frame, text=segment_name)
                else:
                    # For main features, show full name with bold
                    segment_label = ttk.Label(segment_frame, text=segment, font=("Arial", 9, "bold"))
                
                segment_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                # Store button reference
                self.segment_buttons[segment_idx] = color_button
        
        # Bind slider to update function
        segment_slider.config(command=update_visible_segments)
        
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
        
        # Initialize segment buttons array
        self.segment_buttons = [None] * len(self.segments)
        
        # Initialize the first page of segments
        update_visible_segments()
        
        # Save button
        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill=tk.X, pady=10)
        
        save_button = ttk.Button(save_frame, text="Save Segmentations", command=self.save_segmentations)
        save_button.pack(fill=tk.X)
        
        # Add multi-channel and crop controls if applicable
        if self.raw_image_data is not None or len(self.current_channels) > 1:
            self.add_advanced_controls(right_frame)
        
        # Set initial segment
        if self.segments:
            self.select_segment(0)
        
        # Auto-optimize performance for existing annotations
        self.detect_and_optimize_for_existing_annotations()
        
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
        if hasattr(self, 'brush_size_label'):
            self.brush_size_label.config(text=f"{self.brush_size}px")
    
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
            if button is not None:  # Check if button exists
                if i == index:
                    button.config(relief=tk.SUNKEN, borderwidth=3)
                else:
                    button.config(relief=tk.RAISED, borderwidth=1)
    
    def set_eraser(self):
        self.current_color = None
        # Unhighlight all segment buttons
        for button in self.segment_buttons:
            if button is not None:  # Check if button exists
                button.config(relief=tk.RAISED, borderwidth=1)
    
    def update_image(self):
        if not self.original_images:
            return
        
        # Get current image and mask
        original = self.original_images[self.current_image_index]
        mask = self.segmentation_masks[self.current_image_index]
        
        # Apply opacity to the mask (optimized)
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
        
        # Combine original image and segmentation mask
        combined = original.copy()
        combined.paste(mask_with_opacity, (0, 0), mask_with_opacity)
        
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
            # Reset draw counter for next drawing session
            self.draw_count = 0
            # Ensure final update when drawing stops
            if self.update_scheduled:
                self.root.after_cancel(self.update_scheduled)
                self.update_scheduled = False
            self.update_image()  # Full quality update when drawing stops
    
    def draw(self, x, y, line=False):
        mask = self.segmentation_masks[self.current_image_index]
        draw = ImageDraw.Draw(mask)
        
        # Perform the actual drawing
        if self.current_color is None:  # Eraser
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
        
        # AGGRESSIVE PERFORMANCE: Only update every 5th draw while dragging
        if self.drawing:
            self.draw_count += 1
            if self.draw_count >= self.update_every_n_draws:
                self.draw_count = 0
                self.fast_update_image()  # Use fast update during drawing
        else:
            # Single click - update immediately with full quality
            self.draw_count = 0
            self.update_image()
    
    def delayed_update(self):
        """Delayed update method for batching draw operations"""
        if self.update_scheduled:
            self.update_scheduled = False
            self.last_update_time = time.time() * 1000
            self.update_image()
    
    def fast_update_image(self):
        """Optimized update for drawing operations - skips expensive processing"""
        if not self.original_images:
            return
        
        # Get current image and mask
        original = self.original_images[self.current_image_index]
        mask = self.segmentation_masks[self.current_image_index]
        
        # ULTRA-FAST: Skip ALL opacity processing during drawing
        # Combine original image and segmentation mask directly
        combined = original.copy()
        combined.paste(mask, (0, 0), mask)
        
        # ULTRA-FAST: Skip zoom processing if close to 100%
        if abs(self.zoom_factor - 1.0) < 0.05:  # Increased threshold for speed
            zoomed_img = combined
        else:
            width, height = combined.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            # Use fastest possible resampling
            zoomed_img = combined.resize((new_width, new_height), Image.NEAREST)
        
        # Convert to PhotoImage (unavoidable bottleneck)
        self.tk_image = ImageTk.PhotoImage(zoomed_img)
        
        # Minimal canvas update
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas_x_offset, 
            self.canvas_y_offset, 
            anchor=tk.NW, 
            image=self.tk_image
        )
    
    def undo(self):
        if self.history_index >= 0 and len(self.history) > 0:
            # Get the saved state from history
            image_index, mask = self.history[self.history_index]
            
            if image_index == self.current_image_index:
                # Restore the saved mask
                self.segmentation_masks[self.current_image_index] = mask
                self.update_image()
                
                # Move back in history
                self.history_index -= 1
                
                # Provide feedback about remaining undos
                remaining_undos = self.history_index + 1
                if remaining_undos > 0:
                    print(f"✓ Undo applied. {remaining_undos} more undo(s) available.")
                else:
                    print("✓ Undo applied. No more undos available.")
                    
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


                            # Create filenames
                if len(self.original_images) > 1:
                    seg_filename = f"segmented_{base_name}_slice_{i+1}.png"
                    orig_filename = f"original_{base_name}_slice_{i+1}.png"
                else:
                    seg_filename = f"segmented_{base_name}.png"
                    orig_filename = f"original_{base_name}.png"
                
                # Clean filenames to be valid
                seg_filename = re.sub(r'[\\/*?:"<>|]', '_', seg_filename)
                orig_filename = re.sub(r'[\\/*?:"<>|]', '_', orig_filename)
                
                # Save the segmentation image
                seg_save_path = os.path.join(save_dir, seg_filename)
                segmented_img.save(seg_save_path)
                
                # Save the original image
                orig_save_path = os.path.join(save_dir, orig_filename)
                orig_img=self.original_images[self.current_image_index]
                orig_img.save(orig_save_path)



            
            messagebox.showinfo("Success", f"Segmentations saved to {save_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save segmentations: {str(e)}")           # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
 
    
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
        print('current_file_idx', self.current_file_idx)
        self.current_file_idx += 1
        next_file = self.directory_files[self.current_file_idx]
        # next_file=self.current_directory[self.current_file_idx]

        # def get_next_file(folder, current_index, ext=".tif"):
        #     files = [f for f in os.listdir(folder) if f.endswith(ext) and os.path.splitext(f)[0].isdigit()]
        #     numeric_indices = sorted(int(os.path.splitext(f)[0]) for f in files)
            
        #     for idx in numeric_indices:
        #         if idx > current_index:
        #             return f"{idx}{ext}"  # Return filename of next higher index
        #     return None  # No next file

        # next_file=get_next_file(self.current_directory, self.current_file_idx, ".tif")
        # self.current_file_idx += 1

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
        
        if hasattr(self, 'auto_segment_queue'):
            # Show remaining files in queue
            if self.auto_segment_queue:
                pass  # Queue processing logic would go here
            else:
                pass  # No queue
        
        # Process file
        file_ext = os.path.splitext(next_file)[1].lower()
        if file_ext == '.nd2':
            self.process_nd2_file(next_file)
        elif file_ext == '.oib':
            self.process_oib_file(next_file)
        elif file_ext in ['.tif', '.tiff']:
            self.process_multipagetiff_image(next_file)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            self.process_regular_image(next_file)
        else:

            return
        
        # Reset history
        self.history = []
        self.history_index = -1
        
        # Update UI first
        self.create_annotation_window()
        
        # Apply auto-segmentation AFTER UI is created
        if should_auto_segment:            # Ensure UI is fully updated before applying auto-segmentation
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
        elif file_ext == '.oib':
            self.process_oib_file(prev_file)
        elif file_ext in ['.tif', '.tiff']:
            self.process_multipagetiff_image(prev_file)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            self.process_regular_image(prev_file)
        else:

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
                return None
            
            # Log shape information            
            # Handle array with NaN values
            if np.isnan(img_array).any():                
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
                        # Clip to remove extreme outliers
                        img_array_clipped = np.clip(img_array, p1, p99)
                        
                        # Normalize to 0-255
                        normalized = ((img_array_clipped - p1) / (p99 - p1) * 255).astype(np.uint8)
                    else:
                        # Fallback to simple normalization if percentile doesn't work well                        normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                        # If still problematic, create a gray image
                        if np.isnan(normalized).any() or np.isinf(normalized).any():                            
                            normalized = np.ones(img_array.shape, dtype=np.uint8) * 128
                        # Create a blank image as fallback
                except Exception as e:
                    # Create a blank image as fallback
                    normalized = np.zeros(img_array.shape, dtype=np.uint8)
                img_array = normalized
            
            # Handle different dimensions
            if len(img_array.shape) == 2:
                # Grayscale image - convert to RGB                img_array = np.stack([img_array, img_array, img_array], axis=-1)
                img_array = np.stack([img_array, img_array, img_array], axis=-1)

            elif len(img_array.shape) == 3:
                # Handle different 3D formats
                if img_array.shape[2] == 1:
                    img_array = np.concatenate([img_array] * 3, axis=2)

                    # Single channel image - expand to RGB                    img_array = np.concatenate([img_array] * 3, axis=2)
                elif img_array.shape[2] > 4:
                    img_array = img_array[:, :, 0:3]

                    # Too many channels, take the first three                    img_array = img_array[:, :, 0:3]
                elif img_array.shape[2] == 2:
                    # Two channels, add a third  
                    #                     third_channel = np.zeros_like(img_array[:, :, 0:1])
                    third_channel = np.zeros_like(img_array[:, :, 0:1])
                    img_array = np.concatenate([img_array, third_channel], axis=2)
                
                # Check if channels are in first dimension instead of last
                if img_array.shape[0] == 3 and img_array.shape[2] > 100:  
                    img_array = np.transpose(img_array, (1, 2, 0))
            elif len(img_array.shape) > 3:
                # Complex multi-dimensional data                
                # Try various approaches to extract a meaningful image
                try:
                    if img_array.shape[0] == 3 and len(img_array.shape) == 4:
                        img_array = np.transpose(img_array[0:3], (1, 2, 0))

                        # Likely RGB with channels as first dimension                        img_array = np.transpose(img_array[0:3], (1, 2, 0))
                    else:
                        # Take slices until we get to 3D
                        while len(img_array.shape) > 3:
                            img_array = img_array[0]
                        
                        # If we end up with 2D, convert to RGB
                        if len(img_array.shape) == 2:
                            img_array = np.stack([img_array, img_array, img_array], axis=-1)
                        elif img_array.shape[2] > 3:
                            img_array = img_array[:, :, 0:3]
                except Exception as e:
                    # Create a blank image in case of errors
                    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Final validation check
            if img_array.shape[2] != 3 and img_array.shape[2] != 4:                
                return None
                
            # Check for invalid values before creating PIL image
            if np.isnan(img_array).any() or np.isinf(img_array).any():                
                img_array = np.nan_to_num(img_array)
            
            # Create PIL image
            img = Image.fromarray(img_array)            
            return img
        except Exception as e:

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
        elif file_ext == '.oib':
            self.process_oib_file(image_files[0])
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            self.process_regular_image(image_files[0])
        else:

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
            
 # Create class mapping for each segment color
            for idx, segment in enumerate(self.segments):
                color = self.segment_colors[segment]
                # Create boolean mask for this color
                if len(mask_array.shape) == 3 and mask_array.shape[2] >= 3:
                    # RGBA or RGB mask - ensure we're working with the right dimensions
                    color_match = np.all(mask_array[:, :, :3] == np.array(color[:3]), axis=2)
                    
                    # Verify dimensions match before applying
                    if color_match.shape == training_mask.shape:
                        training_mask[color_match] = idx
                        
                else:
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
                            # if color[:3] == (0, 0, 0):  # If it's black
                            #     visible_color = (64, 64, 64, 255)
                            #     temp_mask[mask_pixels] = visible_color
                            # else:
                            #     temp_mask[mask_pixels] = color
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
            messagebox.showerror("Error", f"Auto-segmentation failed: {str(e)}")
    def auto_segment_folder_images(self, num_images=1):
        """Auto-segment other images in the folder based on current image annotation"""
        if not self.original_images or not self.segmentation_masks:
            messagebox.showerror("Error", "No image loaded")
            return

        if not self.directory_files or len(self.directory_files) <= 1:
            messagebox.showerror("Error", "No other images available in folder")
            return
        # Debug print to check directory files
        print(f"Directory files: {[os.path.basename(f) for f in self.directory_files]}")
        print(f"Current file index: {self.current_file_idx}")
        print(f"Current file: {os.path.basename(self.directory_files[self.current_file_idx])}")

        # Get remaining files (files after current one)
        remaining_files = self.directory_files[self.current_file_idx + 1:]
        if not remaining_files:
            messagebox.showerror("Error", "No more files available in folder")
            return
            
        print(f"Remaining files: {[os.path.basename(f) for f in remaining_files]}")
        print(f"Requested auto-segmentation for {num_images} files")
        
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
        
            # Safely destroy progress window
            if progress_window and progress_window.winfo_exists():
                progress_window.destroy()
            progress_window = None
            
            # Set up auto-segmentation queue for subsequent file loads
            current_file = self.directory_files[self.current_file_idx]
            # Get files that come AFTER the current file in directory order
            remaining_files = self.directory_files[self.current_file_idx + 1:]

            # Ensure we don't exceed available files
            actual_num_images = min(num_images, len(remaining_files))
            self.auto_segment_queue = remaining_files[:actual_num_images]
            self.auto_segment_model_ready = True

            messagebox.showinfo("Training Complete", 
                              f"✓ Model trained successfully!\n\n"
                              f"Queue set up with {len(self.auto_segment_queue)} images (requested: {num_images}).\n"
                              f"Directory has {len(self.directory_files)} total files.\n\n"
                              f"Now use 'Next File' button to navigate through the images.\n"
                              f"Each image will be automatically segmented for your review and modification.\n\n"
                              f"Files in queue: {[os.path.basename(f) for f in self.auto_segment_queue]}\n\n"
                              f"Use 'Save Segmentations' to save any image you want to keep.")
        except Exception as e:
            pass
            # Safely destroy progress window if it exists
            if progress_window:
                try:
                    if progress_window.winfo_exists():
                        progress_window.destroy()
                except:
                    pass

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
        
          # Determine if current file is a TIFF file
        current_file = self.directory_files[self.current_file_idx] if self.directory_files else ""
        file_ext = os.path.splitext(current_file)[1].lower() if current_file else ""
        is_2d = file_ext in ['.tif', '.tiff', '.jpg', '.png']
    


        # Option selection
        option_var = tk.StringVar(value="slices")
        
        # Option 1: Subsequent slices (for ND2 files)
        slices_frame = ttk.LabelFrame(frame, text="Subsequent Slices")
        slices_frame.pack(fill=tk.X, pady=5)

        max_slices = len(self.original_images) - self.current_image_index - 1

        # If this is a TIFF file with only one frame, disable the slices option
        slices_disabled = is_2d and max_slices <= 0
        
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
            available_files = len(self.directory_files) - self.current_file_idx - 1  # Exclude current file and already processed files
        
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
        



        slices_var = tk.StringVar(value="1")
        slices_spinbox = ttk.Spinbox(
            slices_control_frame,
            from_=1,
            to=max(1, max_slices),
            textvariable=slices_var,
            width=10,
            state="disabled" if slices_disabled else "normal"
        )
        slices_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(slices_control_frame, text=f"(Max: {max_slices})").pack(side=tk.LEFT, padx=5)
        
        # Option 2: Folder images
        # folder_frame = ttk.LabelFrame(frame, text="Folder Images")
        # folder_frame.pack(fill=tk.X, pady=5)
        
        # folder_radio = ttk.Radiobutton(
        #     folder_frame,
        #     text="Apply to other images in folder",
        #     variable=option_var,
        #     value="folder"
        # )
        # folder_radio.pack(anchor=tk.W, padx=10, pady=5)
        
        # folder_control_frame = ttk.Frame(folder_frame)
        # folder_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # # Calculate available folder images
        # available_files = 0
        # if self.directory_files:
        #     available_files = len(self.directory_files) - 1  # Exclude current file
        
        # ttk.Label(folder_control_frame, text="Number of images:").pack(side=tk.LEFT)
        
        # folder_var = tk.StringVar(value="1")
        # folder_spinbox = ttk.Spinbox(
        #     folder_control_frame,
        #     from_=1,
        #     to=max(1, available_files),
        #     textvariable=folder_var,
        #     width=10
        # )
        # folder_spinbox.pack(side=tk.LEFT, padx=5)
        
        # ttk.Label(folder_control_frame, text=f"(Available: {available_files})").pack(side=tk.LEFT, padx=5)
        
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
                    
                    # Check if there are enough slices
                    if num_slices > max_slices:
                        # If not enough slices but folder images are available, suggest using folder option
                        if available_files > 0:
                            if messagebox.askyesno("Not enough slices", 
                                                f"There aren't enough slices in the current file. Would you like to use folder images instead?"):
                                option_var.set("folder")
                                num_images = min(int(folder_var.get()), available_files)
                                dialog.destroy()
                                self.auto_segment_folder_images(num_images)
                                return
                            else:
                                return
                        else:
                            messagebox.showerror("Error", "Not enough subsequent slices available")
                            return
                            
                    dialog.destroy()
                    self.auto_segment_slices(num_slices)
                else:  # folder
                    num_images = int(folder_var.get())
                    
                    # Check if there are enough folder images
                    if num_images > available_files:
                        messagebox.showerror("Error", f"Only {available_files} images available in folder")
                        return
                        
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
    
    def show_queue_diagnostic(self):
        """Show diagnostic information about the auto-segmentation queue"""
        # Create diagnostic window
        diag_window = tk.Toplevel(self.root)
        diag_window.title("Auto-Segmentation Queue Diagnostic")
        diag_window.geometry("600x400")
        diag_window.transient(self.root)
        diag_window.grab_set()
        
        frame = ttk.Frame(diag_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        title = ttk.Label(frame, text="Queue Diagnostic Information", font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)
        
        # Add diagnostic information
        text.insert(tk.END, f"Model Ready: {getattr(self, 'auto_segment_model_ready', 'Not set')}\n")
        text.insert(tk.END, f"Model Exists: {self.model is not None}\n")
        
        if hasattr(self, 'auto_segment_queue'):
            text.insert(tk.END, f"Queue Length: {len(self.auto_segment_queue)}\n")
            text.insert(tk.END, f"Queue Files:\n")
            for i, file_path in enumerate(self.auto_segment_queue):
                text.insert(tk.END, f"  {i+1}. {os.path.basename(file_path)}\n")
        else:
            text.insert(tk.END, "Queue: Not initialized\n")
        
        text.insert(tk.END, f"\nCurrent Directory: {self.current_directory}\n")
        text.insert(tk.END, f"Directory Files: {len(self.directory_files) if self.directory_files else 0}\n")
        text.insert(tk.END, f"Current File Index: {self.current_file_idx}\n")
        
        if self.directory_files:
            text.insert(tk.END, f"Current File: {os.path.basename(self.directory_files[self.current_file_idx])}\n")
        
        # Make text widget read-only
        text.config(state=tk.DISABLED)
        
        # Close button
        close_button = ttk.Button(frame, text="Close", command=diag_window.destroy)
        close_button.pack(pady=10)

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
            print(f"Debug Error: Could not apply auto-segmentation: {str(e)}")
            import traceback
            traceback.print_exc()
    def load_new_image(self):
        """Load a new individual image with automatic annotation loading"""
        # Ask if user wants to save current segmentation
        if self.segmentation_masks and messagebox.askyesno("Save", "Do you want to save the current segmentation?"):
            self.save_segmentations()
        
        # Ask for image file
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.nd2 *.oib *.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled
        
        # Store directory and find all image files in that directory
        self.current_directory = os.path.dirname(file_path)
        self.image_paths = [file_path]
        self.directory_files = self.get_image_files_in_directory(self.current_directory)
        
        # Find index of current file in directory
        self.current_file_idx = self.directory_files.index(file_path) if file_path in self.directory_files else 0
        
        # Reset zoom and view
        self.zoom_factor = 1.0
        self.canvas_x_offset = 0
        self.canvas_y_offset = 0
        
        # Reset any auto-segmentation queue
        self.auto_segment_queue = []
        self.auto_segment_model_ready = False
        
        # Process file based on extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.nd2':
                
                
                self.process_nd2_file(file_path)
                loading_successful = len(self.original_images) > 0
                
                if not loading_successful:
                    if messagebox.askyesno("ND2 Loading Failed", 
                                           "Failed to load ND2 file. Would you like to see troubleshooting options?"):
                        self.show_nd2_troubleshooting()
                        return
            elif file_ext == '.oib':
                self.process_oib_file(file_path)
                loading_successful = len(self.original_images) > 0

                if not loading_successful:
                    if messagebox.showinfo("oib Loading Failed", 
                                           "Failed to load oib file"):
                        return
            elif file_ext in ['.tif', '.tiff']:
                self.process_multipagetiff_image(file_path)
                loading_successful = len(self.original_images) > 0

                if not loading_successful:
                    if messagebox.showinfo("tiff Loading Failed", 
                                           "Failed to load oib file"):
                        return
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                self.process_regular_image(file_path)
                loading_successful = len(self.original_images) > 0
            else:
                messagebox.showerror("Error", f"Unsupported file format: {file_ext}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            return
        
        if not loading_successful:
            messagebox.showerror("Error", "Failed to load image")
            return
        
        # Reset history
        self.history = []
        self.history_index = -1
        
        # Update UI
        self.create_annotation_window()

    def load_annotation(self):
        """Load an annotation file (JPG or PNG) for the current image/slice"""
        if not self.original_images:
            messagebox.showerror("Error", "No image loaded. Please load an image first.")
            return
        
        # Ask for annotation file
        annotation_path = filedialog.askopenfilename(
            title="Select Annotation File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("PNG files", "*.png"),
                ("JPG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if not annotation_path:
            return  # User cancelled
        
        try:
            # Load the annotation file
            annotation_img = Image.open(annotation_path)
            
            # Get current image size for validation
            current_img = self.original_images[self.current_image_index]
            target_size = current_img.size
            
            # Convert segmentation image back to mask format
            mask = self.convert_segmentation_to_mask(annotation_img, target_size)
            
            if mask is not None:
                # Save current state for undo before applying new annotation
                current_mask = self.segmentation_masks[self.current_image_index].copy()
                # Truncate history if we're not at the end
                if self.history_index < len(self.history) - 1:
                    self.history = self.history[:self.history_index + 1]
                self.history.append((self.current_image_index, current_mask))
                self.history_index = len(self.history) - 1
                
                # Apply the loaded annotation
                self.segmentation_masks[self.current_image_index] = mask
                
                # Count non-transparent pixels for user feedback
                mask_array = np.array(mask)
                non_transparent = np.sum(mask_array[:, :, 3] > 0) if len(mask_array.shape) == 3 and mask_array.shape[2] == 4 else 0
                
                # Update the display
                self.update_image()
                
                # Show success message
                annotation_name = os.path.basename(annotation_path)
                slice_info = f" for slice {self.current_image_index + 1}" if len(self.original_images) > 1 else ""
                messagebox.showinfo("Annotation Loaded", 
                                  f"✓ Successfully loaded annotation: {annotation_name}{slice_info}\n\n"
                                  f"Annotated pixels: {non_transparent:,}\n"
                                  f"Size: {mask.size[0]} × {mask.size[1]}\n\n"
                                  f"You can now edit the loaded annotations or save them.")
            else:
                messagebox.showerror("Error", 
                                   f"Could not convert the annotation file to the expected format.\n\n"
                                   f"Please ensure the annotation file uses the same colors as defined in the segment palette.\n"
                                   f"Supported formats: JPG, PNG")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotation file:\n{str(e)}\n\n"
                               f"Please check that the file is a valid image format (JPG or PNG).")

    def set_performance_mode(self, mode="balanced"):
        """Configure performance settings for different use cases"""
        if mode == "fast":
            self.update_interval = 100  # Less frequent updates for speed
            self.update_every_n_draws = 8  # Even more aggressive - update every 8th draw
        elif mode == "responsive":
            self.update_interval = 25   # More frequent updates for responsiveness
            self.update_every_n_draws = 3  # More responsive - update every 3rd draw
        elif mode == "ultra_fast":
            self.update_interval = 200  # Very infrequent updates
            self.update_every_n_draws = 10  # Very aggressive - update every 10th draw
        else:  # balanced
            self.update_interval = 50   # Default balanced setting
            self.update_every_n_draws = 5  # Default - update every 5th draw
    
    def detect_and_optimize_for_existing_annotations(self):
        """Detect if we have heavy annotations and adjust performance accordingly"""
        if not self.segmentation_masks or not self.segmentation_masks[self.current_image_index]:
            return
        
        # Check if current mask has substantial existing annotations
        mask = self.segmentation_masks[self.current_image_index]
        mask_array = np.array(mask)
        
        if len(mask_array.shape) == 3 and mask_array.shape[2] == 4:
            non_transparent_pixels = np.sum(mask_array[:, :, 3] > 0)
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            annotation_density = non_transparent_pixels / total_pixels
            
            # If more than 5% of pixels are annotated, use ultra-aggressive settings
            if annotation_density > 0.05:
                self.update_every_n_draws = 15  # Very aggressive for heavy annotations
            elif annotation_density > 0.01:
                self.update_every_n_draws = 12  # Aggressive for medium annotations
            else:
                self.update_every_n_draws = 10  # Default ultra_fast setting

   
    
    def array_to_pil_image(self, img_array):
        """Convert numpy array to PIL Image (similar to process_nd2_array)"""
        try:
            if img_array is None or img_array.size == 0:
                return None
            
            # Handle NaN values
            if np.isnan(img_array).any():
                img_array = np.nan_to_num(img_array)
            
            # Convert bit depth if necessary
            if img_array.dtype != np.uint8:
                min_val = np.min(img_array)
                max_val = np.max(img_array)
                
                if max_val > min_val:
                    # Normalize to 0-255
                    normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    normalized = np.ones(img_array.shape, dtype=np.uint8) * 128
                    
                img_array = normalized
            
            # Handle different dimensions
            if len(img_array.shape) == 2:
                # Grayscale - convert to RGB
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            
            # Create PIL image
            img = Image.fromarray(img_array)
            return img
            
        except Exception as e:
            print(f"Error converting array to PIL image: {e}")
            return None

    def add_advanced_controls(self, parent_frame):
        """Add crop and channel selection controls"""
        
        # Channel selection frame
        if len(self.current_channels) > 1:
            channel_frame = ttk.LabelFrame(parent_frame, text="Channel Selection")
            channel_frame.pack(fill=tk.X, pady=5)
            
            # Add "All Channels" option
            self.channel_var = tk.StringVar(value="all")
            
            all_channels_rb = ttk.Radiobutton(
                channel_frame, 
                text="All Channels (Combined)", 
                variable=self.channel_var, 
                value="all",
                command=self.on_channel_change
            )
            all_channels_rb.pack(anchor=tk.W, padx=5, pady=2)
            
            # Add individual channel options
            for i, channel_name in enumerate(self.current_channels):
                channel_rb = ttk.Radiobutton(
                    channel_frame, 
                    text=channel_name, 
                    variable=self.channel_var, 
                    value=str(i),
                    command=self.on_channel_change
                )
                channel_rb.pack(anchor=tk.W, padx=5, pady=2)
        
        # # Crop controls frame
        # if self.raw_image_data is not None:
        #     crop_frame = ttk.LabelFrame(parent_frame, text="Crop Controls")
        #     crop_frame.pack(fill=tk.X, pady=5)
            
        #     crop_button = ttk.Button(
        #         crop_frame, 
        #         text="Configure Crop Region", 
        #         command=self.show_crop_dialog
        #     )
        #     crop_button.pack(fill=tk.X, padx=5, pady=5)
            
        #     if self.is_cropped:
        #         status_label = ttk.Label(crop_frame, text="✓ Image is cropped", foreground="green")
        #         status_label.pack(padx=5, pady=2)
                
        #         reset_button = ttk.Button(
        #             crop_frame, 
        #             text="Reset to Original", 
        #             command=self.reset_crop
        #         )
        #         reset_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Export controls frame
        export_frame = ttk.LabelFrame(parent_frame, text="Export")
        export_frame.pack(fill=tk.X, pady=5)
        
        export_raw_button = ttk.Button(
            export_frame, 
            text="Export Raw Images", 
            command=self.export_raw_images
        )
        export_raw_button.pack(fill=tk.X, padx=5, pady=2)
        
        export_masks_button = ttk.Button(
            export_frame, 
            text="Export Annotation Masks", 
            command=self.export_annotation_masks
        )
        export_masks_button.pack(fill=tk.X, padx=5, pady=2)
        
        if len(self.current_channels) > 1:
            export_combined_button = ttk.Button(
                export_frame, 
                text="Export Multi-Channel Combined", 
                command=self.export_multichannel_combined
            )
            export_combined_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Add crop and save 3D data button if raw data is available
        if self.raw_image_data is not None:
            crop_save_button = ttk.Button(
                export_frame, 
                text="Crop & Save 3D Data", 
                command=self.show_crop_and_save_dialog
            )
            crop_save_button.pack(fill=tk.X, padx=5, pady=2)
    
    def on_channel_change(self):
        """Handle channel selection change"""
        if not hasattr(self, 'channel_var'):
            return
            
        selected = self.channel_var.get()
        
        if selected == "all":
            self.selected_channel = None
        else:
            self.selected_channel = int(selected)
        
        # Regenerate images with new channel selection
        if self.raw_image_data is not None:
            self.regenerate_images_from_raw_data()
            self.update_image()
    
    # def show_crop_dialog(self):
    #     """Show dialog for selecting crop region using rectangle selection"""
    #     if self.raw_image_data is None:
    #         messagebox.showwarning("Warning", "No raw data available for cropping")
    #         return
    #     print('shape of raw image dataaaaa', self.raw_image_data.shape())
    #     # Create a dialog with canvas for rectangle selection
    #     dialog = tk.Toplevel(self.root)
    #     dialog.title("Select Crop Region")
    #     dialog.geometry("800x700")  # Made taller to accommodate zoom controls
    #     dialog.transient(self.root)
    #     dialog.grab_set()
        
    #     # Main frame
    #     main_frame = ttk.Frame(dialog, padding="10")
    #     main_frame.pack(fill=tk.BOTH, expand=True)
        
    #     # Instructions
    #     ttk.Label(main_frame, text="Draw a rectangle to select the region to crop", 
    #              font=("Arial", 10, "bold")).pack(pady=5)
        
    #     # Add zoom controls
    #     zoom_frame = ttk.Frame(main_frame)
    #     zoom_frame.pack(fill=tk.X, pady=5)
        
    #     ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        
    #     zoom_var = tk.DoubleVar(value=1.0)
        
    #     def update_zoom(event=None):
    #         zoom_level = zoom_var.get()
    #         zoom_label.config(text=f"{zoom_level:.1f}x")
    #         display_image_with_zoom()
        
    #     zoom_slider = ttk.Scale(
    #         zoom_frame,
    #         from_=0.1,
    #         to=5.0,
    #         orient=tk.HORIZONTAL,
    #         variable=zoom_var,
    #         command=update_zoom,
    #         length=200
    #     )
    #     zoom_slider.pack(side=tk.LEFT, padx=5)
        
    #     zoom_label = ttk.Label(zoom_frame, text="1.0x", width=5)
    #     zoom_label.pack(side=tk.LEFT, padx=5)
        
    #     zoom_reset = ttk.Button(zoom_frame, text="Reset Zoom", 
    #                        command=lambda: (zoom_var.set(1.0), update_zoom()))
    #     zoom_reset.pack(side=tk.LEFT, padx=5)
        
    #     # Canvas for image display with scrollbars
    #     canvas_frame = ttk.Frame(main_frame)
    #     canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
    #     h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
    #     h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
    #     v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
    #     v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    #     canvas = tk.Canvas(
    #         canvas_frame, 
    #         bg="black",
    #         xscrollcommand=h_scrollbar.set,
    #         yscrollcommand=v_scrollbar.set
    #     )
    #     canvas.pack(fill=tk.BOTH, expand=True)
        
    #     h_scrollbar.config(command=canvas.xview)
    #     v_scrollbar.config(command=canvas.yview)
        
    #     # Enable mouse wheel scrolling
    #     def on_mousewheel(event):
    #         if event.state & 0x4:  # Check if Ctrl key is pressed
    #             # Zoom with Ctrl+Wheel
    #             if event.delta > 0:
    #                 new_zoom = min(5.0, zoom_var.get() + 0.1)
    #             else:
    #                 new_zoom = max(0.1, zoom_var.get() - 0.1)
    #             zoom_var.set(new_zoom)
    #             update_zoom()
    #         else:
    #             # Scroll vertically
    #             canvas.yview_scroll(-1 * (event.delta // 120), "units")
        
    #     canvas.bind("<MouseWheel>", on_mousewheel)  # Windows
    #     canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
    #     canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # Linux
        
    #     # Variables for selection
    #     selection_rect = None
    #     start_x, start_y = 0, 0
    #     end_x, end_y = 0, 0
        
    #     # Get the image data for the current frame
    #     if len(self.original_shape) == 2:  # Single 2D image
    #         img_array = self.raw_image_data
    #     elif len(self.original_shape) == 3:  # (Z, Y, X) 
    #         img_array = self.raw_image_data[self.current_image_index]
    #     elif len(self.original_shape) == 4:  # (C, Z, Y, X)
    #         img_array = self.raw_image_data[self.current_image_index]
    #     else:
    #         img_array = self.raw_image_data
        
    #     # display_img = Image.fromarray(img_array)
    #     display_img = Image.fromarray(img_array)

    #     print('shape of display image', display_img.shape)
    #     # Convert to PIL image
        
        
    #     # if isinstance(img_array, np.ndarray):
    #     #     # Normalize for display
    #     #     if img_array.dtype != np.uint8:
    #     #         img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
    #     #     # Handle different channel configurations
    #     #     if len(img_array.shape) == 2:  # Grayscale
    #     #         display_img = Image.fromarray(img_array)
    #     #     elif len(img_array.shape) == 3:  # RGB or similar
    #     #         if img_array.shape[2] == 1:
    #     #             display_img = Image.fromarray(img_array[:,:,0])
    #     #         elif img_array.shape[2] == 2:
    #     #             display_img = Image.fromarray(0.5*img_array[:,:,0]+ 0.5*img_array[:,:,1])
    #     #         elif img_array.shape[2] == 3:
    #     #             display_img = Image.fromarray(0.5*img_array[:,:,0]+ 0.5*img_array[:,:,1])
    #     #         elif img_array.shape[2] == 4:
    #     #             display_img = Image.fromarray(img_array)
    #     #         else:
    #     #             # Take first channel for display
    #     #             display_img = Image.fromarray(img_array[:,:,0])
    #     #     elif len(img_array.shape) == 4:  # RGB or similar
    #     #         if img_array.shape[0] == 2:
    #     #             display_img = Image.fromarray(0.5*img_array[:,:,0]+ 0.5*img_array[:,:,1])

    #     #         else:
    #     #             # Take first channel for display
    #     #             display_img = Image.fromarray(img_array[:,:,0])
    #     #     else:
    #     #         display_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    #     # else:
    #     #     display_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        
    #     # Store original image dimensions
    #     original_width, original_height = display_img.size
    #     print('original width and height', original_width, original_height)
        
    #     # Function to display image with current zoom
    #     def display_image_with_zoom():
    #         zoom = zoom_var.get()
            
    #         # Calculate new dimensions
    #         new_width = int(original_width * zoom)
    #         new_height = int(original_height * zoom)
            
    #         # Resize image for display
    #         if zoom == 1.0:
    #             resized_img = display_img
    #         else:
    #             resized_img = display_img.resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
    #         # Convert to PhotoImage
    #         photo_img = ImageTk.PhotoImage(resized_img)
            
    #         # Update canvas
    #         canvas.delete("all")
    #         canvas.config(scrollregion=(0, 0, new_width, new_height))
    #         canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
    #         canvas.image = photo_img  # Keep reference
            
    #         # Redraw selection rectangle if it exists
    #         if start_x != end_x and start_y != end_y:
    #             # Scale the coordinates to match zoom
    #             scaled_start_x = int(start_x * zoom)
    #             scaled_start_y = int(start_y * zoom)
    #             scaled_end_x = int(end_x * zoom)
    #             scaled_end_y = int(end_y * zoom)
                
    #             canvas.create_rectangle(
    #                 scaled_start_x, scaled_start_y, 
    #                 scaled_end_x, scaled_end_y,
    #                 outline="red", width=2, tags="selection"
    #             )
        
    #     # Initial display
    #     display_image_with_zoom()
        
    #     # Selection coordinates display
    #     coords_var = tk.StringVar(value="Selection: None")
    #     ttk.Label(main_frame, textvariable=coords_var).pack(pady=5)
        
    #     # Z-dimension controls if applicable
    #     z_start_var = tk.StringVar(value="0")
    #     z_end_var = tk.StringVar(value="0")
    #     has_z_dimension = True
    #     z_size = 0
        

        
    #     if has_z_dimension:
    #         z_frame = ttk.LabelFrame(main_frame, text="Z-Dimension Range")
    #         z_frame.pack(fill=tk.X, pady=5)
            
    #         z_grid = ttk.Frame(z_frame)
    #         z_grid.pack(pady=10, padx=10, fill=tk.X)
            
    #         ttk.Label(z_grid, text="Start Z:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    #         ttk.Label(z_grid, text="End Z:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            
    #         z_start_var.set("0")
    #         z_end_var.set(str(z_size))
            
    #         z_start_entry = ttk.Entry(z_grid, textvariable=z_start_var, width=10)
    #         z_start_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
    #         z_end_entry = ttk.Entry(z_grid, textvariable=z_end_var, width=10)
    #         z_end_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            
    #         ttk.Label(z_grid, text=f"(Max: {z_size})").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
    #     # Functions for rectangle selection
    #     def start_selection(event):
    #         nonlocal selection_rect, start_x, start_y
            
    #         # Convert canvas coordinates to original image coordinates
    #         zoom = zoom_var.get()
    #         canvas_x = canvas.canvasx(event.x)
    #         canvas_y = canvas.canvasy(event.y)
            
    #         # Convert to original image coordinates
    #         start_x = int(canvas_x / zoom)
    #         start_y = int(canvas_y / zoom)
            
    #         # Create new rectangle
    #         selection_rect = canvas.create_rectangle(
    #             canvas_x, canvas_y, canvas_x, canvas_y,
    #             outline="red", width=2, tags="selection"
    #         )
        
    #     def update_selection(event):
    #         nonlocal end_x, end_y
            
    #         if selection_rect:
    #             # Convert canvas coordinates to original image coordinates
    #             zoom = zoom_var.get()
    #             canvas_x = canvas.canvasx(event.x)
    #             canvas_y = canvas.canvasy(event.y)
                
    #             # Convert to original image coordinates
    #             end_x = int(canvas_x / zoom)
    #             end_y = int(canvas_y / zoom)
                
    #             # Update rectangle
    #             canvas.coords(selection_rect, 
    #                          start_x * zoom, start_y * zoom, 
    #                          canvas_x, canvas_y)
                
    #             # Update coordinates display
    #             width = abs(end_x - start_x)
    #             height = abs(end_y - start_y)
    #             coords_var.set(f"Selection: ({min(start_x, end_x)}, {min(start_y, end_y)}) to " +
    #                           f"({max(start_x, end_x)}, {max(start_y, end_y)}), Size: {width}x{height}")
        
    #     def end_selection(event):
    #         nonlocal end_x, end_y
            
    #         if selection_rect:
    #             # Convert canvas coordinates to original image coordinates
    #             zoom = zoom_var.get()
    #             canvas_x = canvas.canvasx(event.x)
    #             canvas_y = canvas.canvasy(event.y)
                
    #             # Convert to original image coordinates
    #             end_x = int(canvas_x / zoom)
    #             end_y = int(canvas_y / zoom)
                
    #             # Ensure coordinates are within image bounds
    #             end_x = max(0, min(end_x, original_width))
    #             end_y = max(0, min(end_y, original_height))
                
    #             # Update rectangle
    #             canvas.coords(selection_rect, 
    #                          start_x * zoom, start_y * zoom, 
    #                          end_x * zoom, end_y * zoom)
        
    #     # Bind events
    #     canvas.bind("<ButtonPress-1>", start_selection)
    #     canvas.bind("<B1-Motion>", update_selection)
    #     canvas.bind("<ButtonRelease-1>", end_selection)
        
    #     # Buttons
    #     button_frame = ttk.Frame(main_frame)
    #     button_frame.pack(fill=tk.X, pady=10)
        
    #     def apply_crop():
    #         # Get selection coordinates
    #         x_start = min(start_x, end_x)
    #         y_start = min(start_y, end_y)
    #         x_end = max(start_x, end_x)
    #         y_end = max(start_y, end_y)
            
    #         # Validate selection
    #         if x_start == x_end or y_start == y_end:
    #             messagebox.showerror("Error", "Invalid selection. Please select a region.")
    #             return
            
    #         # Get Z range if applicable
    #         z_start = 0
    #         z_end = 0
    #         if has_z_dimension:
    #             try:
    #                 z_start = int(z_start_var.get())
    #                 z_end = int(z_end_var.get())
                    
    #                 if z_start < 0 or z_end > z_size or z_start > z_end:
    #                     messagebox.showerror("Error", f"Invalid Z range. Must be between 0 and {z_size}.")
    #                     return
    #             except ValueError:
    #                 messagebox.showerror("Error", "Please enter valid Z range values.")
    #                 return
            
    #         # Perform crop on raw data
    #         try:
    #             if len(self.original_shape) == 2:  # Single 2D image
    #                 cropped_data = self.raw_image_data[y_start:y_end+1, x_start:x_end+1]
    #             elif len(self.original_shape) == 3:
    #                 if self.original_shape[2] in [3, 4]:  # (Y, X, C)
    #                     cropped_data = self.raw_image_data[y_start:y_end+1, x_start:x_end+1, :]
    #                 else:  # (Z, Y, X)
    #                     cropped_data = self.raw_image_data[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
    #             elif len(self.original_shape) == 4:  # (Z, C, Y, X) or similar
    #                 cropped_data = self.raw_image_data[:, z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
    #             else:
    #                 messagebox.showerror("Error", "Unsupported data shape for cropping.")
    #                 return
                
    #             # Update raw data
    #             self.raw_image_data = cropped_data
    #             self.original_shape = cropped_data.shape
                
    #             # Update display
    #             self.regenerate_images_from_raw_data()
                
    #             # Close dialog
    #             dialog.destroy()
                
    #             # Show success message
    #             messagebox.showinfo("Success", f"Image cropped to region ({x_start}, {y_start}) - ({x_end}, {y_end})" + 
    #                               (f" and Z range {z_start}-{z_end}" if has_z_dimension else ""))
                
    #         except Exception as e:
    #             messagebox.showerror("Error", f"Failed to crop: {str(e)}")
        
    #     ttk.Button(button_frame, text="Apply Crop", command=apply_crop).pack(side=tk.LEFT, padx=5)
    #     ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
    #     # Center dialog
    #     dialog.update_idletasks()
    #     width = dialog.winfo_width()
    #     height = dialog.winfo_height()
    #     x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    #     y = (dialog.winfo_screenheight() // 2) - (height // 2)
    #     dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    # def reset_crop(self):
    #     """Reset to original uncropped data"""
    #     if self.original_raw_data is not None:
    #         self.raw_image_data = self.original_raw_data.copy()
    #         self.is_cropped = False
            
    #         # Regenerate images
    #         self.regenerate_images_from_raw_data()
            
    #         # Update segmentation masks
    #         self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0)) for img in self.original_images]
    #         self.current_image_index = 0
            
    #         # Refresh UI
    #         self.create_annotation_window()
            
    #         messagebox.showinfo("Success", "Reset to original data")
    
    def regenerate_images_from_raw_data(self):
        """Regenerate display images from raw data based on current settings"""
        if self.raw_image_data is None:
            return
        
        self.original_images = []
        data = self.raw_image_data
        
        # Simplified processing for now
        if len(data.shape) >= 4:  # Multi-dimensional
            # Take first timepoint if exists
            if data.shape[0] > 1:  # Likely time dimension
                data = data[0]
            
            # Handle channels
            if len(data.shape) >= 3 and data.shape[0] > 1:  # Likely channel dimension
                if self.selected_channel is None:
                    # Combine all channels
                    if data.shape[0] == 3:
                        # RGB-like, transpose to put channels last
                        data = np.transpose(data, (1, 2, 0))
                    else:
                        # Average channels
                        data = np.mean(data, axis=0)
                else:
                    # Select specific channel
                    if self.selected_channel < data.shape[0]:
                        data = data[self.selected_channel]
                    else:
                        data = data[0]  # Fallback
            
            # Handle Z dimension
            if len(data.shape) == 3 and data.shape[0] > 1:  # Z, Y, X
                for z in range(data.shape[0]):
                    img_array = data[z]
                    pil_img = self.array_to_pil_image(img_array)
                    if pil_img:
                        self.original_images.append(pil_img)
            else:  # Single 2D image
                pil_img = self.array_to_pil_image(data)
                if pil_img:
                    self.original_images.append(pil_img)
        else:
            # Simple 2D case
            pil_img = self.array_to_pil_image(data)
            if pil_img:
                self.original_images.append(pil_img)
    
    def export_raw_images(self):
        """Export raw images in current channel configuration"""
        if not self.original_images:
            messagebox.showerror("Error", "No images to export")
            return
        
        save_dir = filedialog.askdirectory(title="Select Export Directory")
        if not save_dir:
            return
        
        try:
            base_name = "raw_image"
            if self.image_paths:
                base_name = os.path.splitext(os.path.basename(self.image_paths[0]))[0]
            
            for i, img in enumerate(self.original_images):
                if len(self.original_images) > 1:
                    filename = f"{base_name}_slice_{i+1:03d}.png"
                else:
                    filename = f"{base_name}_raw.png"
                
                save_path = os.path.join(save_dir, filename)
                img.save(save_path)
            
            messagebox.showinfo("Success", f"Exported {len(self.original_images)} raw images to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export raw images: {str(e)}")
    
    def export_annotation_masks(self):
        """Export annotation masks"""
        if not self.segmentation_masks:
            messagebox.showerror("Error", "No annotation masks to export")
            return
        
        save_dir = filedialog.askdirectory(title="Select Export Directory")
        if not save_dir:
            return
        
        try:
            base_name = "annotation_mask"
            if self.image_paths:
                base_name = os.path.splitext(os.path.basename(self.image_paths[0]))[0]
            
            for i, mask in enumerate(self.segmentation_masks):
                if len(self.segmentation_masks) > 1:
                    filename = f"{base_name}_mask_{i+1:03d}.png"
                else:
                    filename = f"{base_name}_mask.png"
                
                save_path = os.path.join(save_dir, filename)
                mask.save(save_path)
            
            messagebox.showinfo("Success", f"Exported {len(self.segmentation_masks)} annotation masks to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export annotation masks: {str(e)}")
    
    def export_multichannel_combined(self):
        """Export multi-channel data as combined single channel"""
        if self.raw_image_data is None:
            messagebox.showerror("Error", "No multi-channel data available")
            return
        
        save_dir = filedialog.askdirectory(title="Select Export Directory")
        if not save_dir:
            return
        
        try:
            # Temporarily set to combine all channels
            original_selection = self.selected_channel
            self.selected_channel = None
            
            # Regenerate images with combined channels
            self.regenerate_images_from_raw_data()
            
            base_name = "multichannel_combined"
            if self.image_paths:
                base_name = os.path.splitext(os.path.basename(self.image_paths[0]))[0] + "_combined"
            
            for i, img in enumerate(self.original_images):
                if len(self.original_images) > 1:
                    filename = f"{base_name}_slice_{i+1:03d}.png"
                else:
                    filename = f"{base_name}.png"
                
                save_path = os.path.join(save_dir, filename)
                img.save(save_path)
            
            # Restore original selection
            self.selected_channel = original_selection
            self.regenerate_images_from_raw_data()
            
            messagebox.showinfo("Success", f"Exported {len(self.original_images)} combined multi-channel images to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export multi-channel combined: {str(e)}")
    
    def show_crop_and_save_dialog(self):
        """Show dialog for cropping and saving 3D data as ND2 or OIB file"""
        if self.raw_image_data is None:
            messagebox.showwarning("Warning", "No 3D data available for cropping and saving")
            return
        # Create dialog with canvas for rectangle selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Crop and Save 3D Data")
        dialog.geometry("800x1000")  # Made larger to accommodate all controls
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main frame
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Crop and Save 3D Data", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(frame, text=f"Original Shape: {self.original_shape}", font=("Arial", 10)).pack(pady=5)
        
        # 3D Dimensions frame
        dim_frame = ttk.LabelFrame(frame, text="3D Dimensions")
        dim_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Determine available dimensions based on shape
        has_z_dimension = False
        has_c_dimension = False
        has_t_dimension = False
        z_size = c_size = t_size = 0
        
        # Set up dimension ranges based on data shape
        if len(self.original_shape) == 3 and self.original_shape[2] in [2, 3, 4]:  # (Y, X, C) or similar
            has_c_dimension = True
            c_size = self.original_shape[2] - 1
        elif len(self.original_shape) == 3:  # (Z, Y, X) or similar
            has_z_dimension = True
            z_size = self.original_shape[0] - 1
        elif len(self.original_shape) == 4:  # Could be (T/C, Z, Y, X)
            has_c_dimension = self.original_shape[0] > 1
            has_z_dimension = self.original_shape[1] > 1
            c_size = self.original_shape[0] - 1
            z_size = self.original_shape[1] - 1

        
        # Create entry fields for each dimension
        dim_entries = {}
        row = 0
        
        # Z dimension
        if has_z_dimension:
            ttk.Label(dim_frame, text=f"Z Range (0-{z_size}):").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
            z_start_var = tk.StringVar(value="0")
            z_start_entry = ttk.Entry(dim_frame, textvariable=z_start_var, width=10)
            z_start_entry.grid(row=row, column=1, padx=5, pady=5)
            
            ttk.Label(dim_frame, text="to").grid(row=row, column=2, padx=5, pady=5)
            
            z_end_var = tk.StringVar(value=str(z_size))
            z_end_entry = ttk.Entry(dim_frame, textvariable=z_end_var, width=10)
            z_end_entry.grid(row=row, column=3, padx=5, pady=5)
            
            dim_entries[1] = (z_start_var, z_end_var, z_size)
            row += 1
        
        # C dimension
        if has_c_dimension:
            ttk.Label(dim_frame, text=f"C Range (0-{c_size}):").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
            c_start_var = tk.StringVar(value="0")
            c_start_entry = ttk.Entry(dim_frame, textvariable=c_start_var, width=10)
            c_start_entry.grid(row=row, column=1, padx=5, pady=5)
            
            ttk.Label(dim_frame, text="to").grid(row=row, column=2, padx=5, pady=5)
            
            c_end_var = tk.StringVar(value=str(c_size))
            c_end_entry = ttk.Entry(dim_frame, textvariable=c_end_var, width=10)
            c_end_entry.grid(row=row, column=3, padx=5, pady=5)
            
            dim_entries[0] = (c_start_var, c_end_var, c_size)
            row += 1
        
        # T dimension
        if has_t_dimension:
            ttk.Label(dim_frame, text=f"T Range (0-{t_size}):").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
            t_start_var = tk.StringVar(value="0")
            t_start_entry = ttk.Entry(dim_frame, textvariable=t_start_var, width=10)
            t_start_entry.grid(row=row, column=1, padx=5, pady=5)
            
            ttk.Label(dim_frame, text="to").grid(row=row, column=2, padx=5, pady=5)
            
            t_end_var = tk.StringVar(value=str(t_size))
            t_end_entry = ttk.Entry(dim_frame, textvariable=t_end_var, width=10)
            t_end_entry.grid(row=row, column=3, padx=5, pady=5)
            
            dim_entries[0] = (t_start_var, t_end_var, t_size)
        




        # Add slice navigation slider
        slice_nav_frame = ttk.LabelFrame(frame, text="Slice Navigation")
        slice_nav_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Determine max slice index based on data shape
        max_slice = 0
        current_slice_var = tk.IntVar(value=0)
        
        if has_z_dimension:
            max_slice = z_size
        elif has_c_dimension and not has_z_dimension:
            max_slice = c_size
        
        # Create slider for navigating slices
        ttk.Label(slice_nav_frame, text="Current Slice:").pack(side=tk.LEFT, padx=5)
        
        slice_label = ttk.Label(slice_nav_frame, text="0", width=5)
        slice_label.pack(side=tk.RIGHT, padx=5)
        
        # def update_slice(event=None):
        #     print('updating slice')
        #     slice_idx = current_slice_var.get()
        #     slice_label.config(text=str(slice_idx))
            
        #     # Update displayed image based on slice index
        #     if has_z_dimension and has_c_dimension:  # (C, Z, Y, X)
        #         c_idx = 0  # Default to first channel
        #         display_img = Image.fromarray(
        #             ((self.raw_image_data[c_idx, slice_idx] - self.raw_image_data[c_idx, slice_idx].min()) / 
        #             (self.raw_image_data[c_idx, slice_idx].max() - self.raw_image_data[c_idx, slice_idx].min() + 1e-8) * 255).astype(np.uint8)
        #         )
        #     elif has_c_dimension and self.raw_image_data.shape[2] in [2, 3, 4]:  # (c, Y, X)
        #         display_img = Image.fromarray(
        #             ((self.raw_image_data[0] - self.raw_image_data[slice0_idx].min()) / 
        #             (self.raw_image_data[0].max() - self.raw_image_data[0].min() + 1e-8) * 255).astype(np.uint8)
        #         )
        #     elif has_z_dimension:  # (Z, Y, X)
        #         display_img = Image.fromarray(
        #             ((self.raw_image_data[slice_idx] - self.raw_image_data[slice_idx].min()) / 
        #             (self.raw_image_data[slice_idx].max() - self.raw_image_data[slice_idx].min() + 1e-8) * 255).astype(np.uint8)
        #         )
        #     else:  # Default to first slice
        #         display_img = Image.fromarray(
        #             ((self.raw_image_data[0] - self.raw_image_data[0].min()) / 
        #             (self.raw_image_data[0].max() - self.raw_image_data[0].min() + 1e-8) * 255).astype(np.uint8)
        #         )
            
        #     # Store original image dimensions
        #     nonlocal original_width, original_height
        #     original_width, original_height = display_img.size
            
        #     # Update zoom to refresh display
        #     update_zoom()
        def update_slice(event=None):
            slice_idx = current_slice_var.get()
            slice_label.config(text=str(slice_idx))
            
            # Update displayed image based on slice index
            nonlocal display_img
            
            try:
                if has_z_dimension and has_c_dimension:  # (C, Z, Y, X)
                    c_idx = 0  # Default to first channel
                    img_array = self.raw_image_data[c_idx, slice_idx]
                elif has_z_dimension and self.raw_image_data[2] in [ 2, 3, 4]:  # (Z, Y, X)
                    img_array = self.raw_image_data[slice_idx]
                elif has_z_dimension:  # (Z, Y, X)
                    img_array = self.raw_image_data[slice_idx]
                else:  # Default to first slice
                    img_array = self.raw_image_data[0]
                
                # Normalize for display
                if img_array.dtype != np.uint8:
                    img_array = ((img_array - img_array.min()) / 
                                (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
                
                display_img = Image.fromarray(img_array)
                
                # Update original dimensions
                nonlocal original_width, original_height
                original_width, original_height = display_img.size
                
                # Update display with new image
                update_zoom()
            except Exception as e:
                print(f"Error updating slice: {str(e)}")
        slice_slider = ttk.Scale(
            slice_nav_frame,
            from_=0,
            to=max_slice,
            orient=tk.HORIZONTAL,
            variable=current_slice_var,
            command=update_slice,
            length=400
        )
        slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)



            









        # XY region selection
        ttk.Label(frame, text="Select XY Region:", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Add zoom controls
        zoom_frame = ttk.Frame(frame)
        zoom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        
        zoom_var = tk.DoubleVar(value=1.0)
        
        zoom_slider = ttk.Scale(
            zoom_frame,
            from_=0.1,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=zoom_var,
            length=200
        )
        zoom_slider.pack(side=tk.LEFT, padx=5)
        
        zoom_label = ttk.Label(zoom_frame, text="1.0x", width=5)
        zoom_label.pack(side=tk.LEFT, padx=5)
        
        zoom_reset = ttk.Button(zoom_frame, text="Reset Zoom", 
                           command=lambda: (zoom_var.set(1.0), update_zoom()))
        zoom_reset.pack(side=tk.LEFT, padx=5)
        
        # Canvas for image display with scrollbars
        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(
            canvas_frame, 
            bg="black",
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set,
            width=600,
            height=400
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        
        h_scrollbar.config(command=canvas.xview)
        v_scrollbar.config(command=canvas.yview)
        def update_zoom(event=None):
            zoom = zoom_var.get()
            zoom_label.config(text=f"{zoom:.1f}x")
            
            # Calculate new dimensions
            new_width = int(original_width * zoom)
            new_height = int(original_height * zoom)
            
            # Resize image for display
            if zoom == 1.0:
                resized_img = display_img
            else:
                resized_img = display_img.resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
            # Convert to PhotoImage
            photo_img = ImageTk.PhotoImage(resized_img)
            
            # Update canvas
            canvas.delete("all")
            canvas.config(scrollregion=(0, 0, new_width, new_height))
            canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
            canvas.image = photo_img  # Keep reference
            
            # Redraw selection rectangle if it exists
            if start_x != end_x and start_y != end_y:
                # Scale the coordinates to match zoom
                scaled_start_x = int(start_x * zoom)
                scaled_start_y = int(start_y * zoom)
                scaled_end_x = int(end_x * zoom)
                scaled_end_y = int(end_y * zoom)
                
                canvas.create_rectangle(
                    scaled_start_x, scaled_start_y, 
                    scaled_end_x, scaled_end_y,
                    outline="red", width=2, tags="selection"
                )
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            if event.state & 0x4:  # Check if Ctrl key is pressed
                # Zoom with Ctrl+Wheel
                if event.delta > 0:
                    new_zoom = min(5.0, zoom_var.get() + 0.1)
                else:
                    new_zoom = max(0.1, zoom_var.get() - 0.1)
                zoom_var.set(new_zoom)
                update_zoom()
            else:
                # Scroll vertically
                canvas.yview_scroll(-1 * (event.delta // 120), "units")
        
        canvas.bind("<MouseWheel>", on_mousewheel)  # Windows
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # Linux
        
        # Variables for selection
        selection_rect = None
        start_x, start_y = 0, 0
        end_x, end_y = 0, 0

    # Update displayed image based on slice index
        if has_z_dimension and has_c_dimension:  # (C, Z, Y, X)
            display_img=Image.fromarray(self.raw_image_data[0, 0, :, :])
        elif has_c_dimension and self.raw_image_data.shape[2] in [2, 3, 4]:  # (c, Y, X)
            display_img=Image.fromarray(self.raw_image_data[:, :, :])
        elif has_z_dimension:  # (Z, Y, X)
            display_img=Image.fromarray(self.raw_image_data[0, :, :])
        else:  # Default to first slice
            display_img=Image.fromarray(self.raw_image_data[0, 0, :, :])
            
        # # Get the image data for the current frame
        # current_idx = self.current_image_index
        # if len(self.original_shape) == 2:  # Single 2D image
        #     img_array = self.raw_image_data
        # elif len(self.original_shape) == 3:  # (Z, Y, X) or (Y, X, C)
        #     if self.original_shape[2] in [3, 4]:  # Likely (Y, X, C)
        #         img_array = self.raw_image_data
        #     else:  # Likely (Z, Y, X)
        #         img_array = self.raw_image_data[current_idx % self.original_shape[0]]
        # elif len(self.original_shape) == 4:  # (Z, C, Y, X) or similar
        #     img_array = self.raw_image_data[current_idx % self.original_shape[0]]
        # else:
        #     img_array = self.raw_image_data
        
        # # Convert to PIL image for display
        # if isinstance(img_array, np.ndarray):
        #     # Normalize for display
        #     if img_array.dtype != np.uint8:
        #         img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
        #     # Handle different channel configurations
        #     if len(img_array.shape) == 2:  # Grayscale
        #         display_img = Image.fromarray(img_array)
        #     elif len(img_array.shape) == 3:  # RGB or similar
        #         if img_array.shape[2] == 1:
        #             display_img = Image.fromarray(img_array[:,:,0])
        #         elif img_array.shape[2] == 3:
        #             display_img = Image.fromarray(img_array)
        #         elif img_array.shape[2] == 4:
        #             display_img = Image.fromarray(img_array)
        #         else:
        #             # Take first channel for display
        #             display_img = Image.fromarray(img_array[:,:,0])
        #     else:
        #         display_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        # else:
        #     display_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        
        # Store original image dimensions
        original_width, original_height = display_img.size
        
        # # Function to display image with current zoom
        # def update_zoom(event=None):
        #     zoom = zoom_var.get()
        #     zoom_label.config(text=f"{zoom:.1f}x")
            
        #     # Calculate new dimensions
        #     new_width = int(original_width * zoom)
        #     new_height = int(original_height * zoom)
            
        #     # Resize image for display
        #     if zoom == 1.0:
        #         resized_img = display_img
        #     else:
        #         resized_img = display_img.resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
        #     # Convert to PhotoImage
        #     photo_img = ImageTk.PhotoImage(resized_img)
            
        #     # Update canvas
        #     canvas.delete("all")
        #     canvas.config(scrollregion=(0, 0, new_width, new_height))
        #     canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
        #     canvas.image = photo_img  # Keep reference
            
        #     # Redraw selection rectangle if it exists
        #     if start_x != end_x and start_y != end_y:
        #         # Scale the coordinates to match zoom
        #         scaled_start_x = int(start_x * zoom)
        #         scaled_start_y = int(start_y * zoom)
        #         scaled_end_x = int(end_x * zoom)
        #         scaled_end_y = int(end_y * zoom)
                
        #         canvas.create_rectangle(
        #             scaled_start_x, scaled_start_y, 
        #             scaled_end_x, scaled_end_y,
        #             outline="red", width=2, tags="selection"
        #         )
        # # Then define on_mousewheel function
        # def on_mousewheel(event):
        #     if event.state & 0x4:  # Check if Ctrl key is pressed
        #         # Zoom with Ctrl+Wheel
        #         if event.delta > 0:
        #             new_zoom = min(5.0, zoom_var.get() + 0.1)
        #         else:
        #             new_zoom = max(0.1, zoom_var.get() - 0.1)
        #         zoom_var.set(new_zoom)
        #         update_zoom()
        #     else:
        #         # Scroll vertically
        #         canvas.yview_scroll(-1 * (event.delta // 120), "units")
        # Initial display
        update_zoom()
        
        # Selection coordinates display
        coords_var = tk.StringVar(value="Selection: None")
        ttk.Label(frame, textvariable=coords_var).pack(pady=5)
        
        # Functions for rectangle selection
        def start_selection(event):
            nonlocal selection_rect, start_x, start_y
            
            # Convert canvas coordinates to original image coordinates
            zoom = zoom_var.get()
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)
            
            # Convert to original image coordinates
            start_x = int(canvas_x / zoom)
            start_y = int(canvas_y / zoom)
            
            # Create new rectangle
            selection_rect = canvas.create_rectangle(
                canvas_x, canvas_y, canvas_x, canvas_y,
                outline="red", width=2, tags="selection"
            )
        
        def update_selection(event):
            nonlocal end_x, end_y
            
            if selection_rect:
                # Convert canvas coordinates to original image coordinates
                zoom = zoom_var.get()
                canvas_x = canvas.canvasx(event.x)
                canvas_y = canvas.canvasy(event.y)
                
                # Convert to original image coordinates
                end_x = int(canvas_x / zoom)
                end_y = int(canvas_y / zoom)
                
                # Update rectangle
                canvas.coords(selection_rect, 
                             start_x * zoom, start_y * zoom, 
                             canvas_x, canvas_y)
                
                # Update coordinates display
                width = abs(end_x - start_x)
                height = abs(end_y - start_y)
                coords_var.set(f"Selection: ({min(start_x, end_x)}, {min(start_y, end_y)}) to "
                              f"({max(start_x, end_x)}, {max(start_y, end_y)}), Size: {width}x{height}")
        
        def end_selection(event):
            nonlocal end_x, end_y
            
            if selection_rect:
                # Convert canvas coordinates to original image coordinates
                zoom = zoom_var.get()
                canvas_x = canvas.canvasx(event.x)
                canvas_y = canvas.canvasy(event.y)
                
                # Convert to original image coordinates
                end_x = int(canvas_x / zoom)
                end_y = int(canvas_y / zoom)
                
                # Ensure coordinates are within image bounds
                end_x = max(0, min(end_x, original_width))
                end_y = max(0, min(end_y, original_height))
                
                # Update rectangle
                canvas.coords(selection_rect, 
                             start_x * zoom, start_y * zoom, 
                             end_x * zoom, end_y * zoom)
        
        # Bind events
        canvas.bind("<ButtonPress-1>", start_selection)
        canvas.bind("<B1-Motion>", update_selection)
        canvas.bind("<ButtonRelease-1>", end_selection)
        
        # Output file selection
        file_frame = ttk.LabelFrame(frame, text="Output Directory")
        file_frame.pack(fill=tk.X, pady=10)
        
        file_path_var = tk.StringVar()
        file_path_entry = ttk.Entry(file_frame, textvariable=file_path_var)
        file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5), pady=10)
        
        def browse_save_dir():
            directory = filedialog.askdirectory(
                title="Select Output Directory"
            )
            if directory:
                file_path_var.set(directory)
        
        browse_button = ttk.Button(file_frame, text="Browse...", command=browse_save_dir)
        browse_button.pack(side=tk.RIGHT, padx=(5, 10), pady=10)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_cropped_data():
            try:
                # Validate crop parameters
                x_start = min(start_x, end_x)
                y_start = min(start_y, end_y)
                x_end = max(start_x, end_x)
                y_end = max(start_y, end_y)
                
                # Validate selection
                if x_start == x_end or y_start == y_end:
                    messagebox.showerror("Error", "Please draw a selection rectangle first")
                    return
                
                slices = []
                
                # Handle different data shapes
                if len(self.original_shape) == 2:  # (Y, X)
                    slices = [
                        slice(y_start, y_end + 1),
                        slice(x_start, x_end + 1)
                    ]
                elif len(self.original_shape) == 3:  # (Z, Y, X) or (Y, X, C)
                    if self.original_shape[2] in [2, 3, 4]:  # (Y, X, C)
                        slices = [
                            slice(y_start, y_end + 1),
                            slice(x_start, x_end + 1),
                            slice(None)
                        ]
                    else:  # (Z, Y, X)
                        z_start = int(z_start_var.get())
                        z_end = int(z_end_var.get())
                        
                        if z_start < 0 or z_end >= self.original_shape[0] or z_start > z_end:
                            messagebox.showerror("Error", f"Invalid Z range")
                            return
                        
                        slices = [
                            slice(z_start, z_end + 1),
                            slice(y_start, y_end + 1),
                            slice(x_start, x_end + 1)
                        ]
                elif len(self.original_shape) == 4:  # (C, Z, Y, X)
                    try:
                        c_start = int(c_start_var.get())
                        c_end = int(c_end_var.get())
                        z_start = int(z_start_var.get())
                        z_end = int(z_end_var.get())
                        
                        if c_start < 0 or c_end >= self.original_shape[0] or c_start > c_end:
                            messagebox.showerror("Error", f"Invalid C range")
                            return
                        if z_start < 0 or z_end >= self.original_shape[1] or z_start > z_end:
                            messagebox.showerror("Error", f"Invalid Z range")
                            return
                        
                        slices = [
                            slice(c_start, c_end + 1),
                            slice(z_start, z_end + 1),
                            slice(y_start, y_end + 1),
                            slice(x_start, x_end + 1)
                        ]
                    except (ValueError, IndexError):
                        messagebox.showerror("Error", "Invalid dimension values")
                        return
                
                # Get output directory
                output_path = file_path_var.get().strip()
                
                if not output_path:
                    messagebox.showerror("Error", "Please specify an output directory")
                    return
                
                
                file_path= self.file_path
                file_path = os.path.splitext(os.path.basename(file_path))[0]
                print('filename', file_path)
                crop_info=  file_path + '_' + str(x_start) +  '_' + str(x_end) + '_' + str(y_start) +  '_'  + str(y_end)
                
                # Create progress window
                progress_window = tk.Toplevel(dialog)
                progress_window.title("Saving Data")
                progress_window.geometry("300x150")
                progress_window.transient(dialog)
                progress_window.grab_set()
                
                progress_label = ttk.Label(progress_window, text="Cropping and saving data...")
                progress_label.pack(pady=20)
                
                progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="indeterminate")
                progress_bar.pack(pady=10)
                progress_bar.start()
                
                # Update UI
                progress_window.update()
                
                # Crop the data
                cropped_data = self.raw_image_data[tuple(slices)]
                
                # Save the data
                self.save_3d_data(cropped_data, output_path, z_start,  crop_info)
                
                # Close progress window
                progress_window.destroy()
                
                # Close dialog
                dialog.destroy()
                
                # Show success message
                messagebox.showinfo("Success", f"Cropped data saved to {output_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save cropped data: {str(e)}")
        
        save_button = ttk.Button(button_frame, text="Save Cropped Data", command=save_cropped_data)
        save_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def save_3d_data(self, data, output_path, z_start, crop_info):
        """Save 3D data in the specified format (nd2 or oib)"""

        try:
            import imageio
            # For nd2, we'll use a different approach since direct writing may not be available
            # Convert to TIFF stack as an alternative that's widely supported
            import tifffile
            # from skimage.external import tifffile as tif
            
            # # Change extension to .tif if nd2 writing isn't supported
            # if output_path.endswith('.nd2'):
            #     output_path = output_path[:-4] + '.tif'
            #     messagebox.showwarning("Format Note", 
            #         "ND2 writing not directly supported. Saving as multi-page TIFF instead, which can be loaded by most microscopy software.")
            
            # Save as multi-page TIFF
            # print('shape of tiff data', data.shape)
            # if len(data.shape) >= 3:
            #     # Reshape data to ensure it's in the right format
            #     if len(data.shape) == 5:  # (T, C, Z, Y, X)
            #         data = data.transpose(0, 2, 1, 3, 4)  # (T, Z, C, Y, X)
            #     elif len(data.shape) == 4:  # Could be (T, C, Y, X) or (C, Z, Y, X)
            #         # Assume it's (T/Z, C, Y, X) and reorder to (T/Z, C, Y, X)
            #         pass
            #     elif len(data.shape) == 3:  # (Z, Y, X)
            #         # Add channel dimension: (Z, 1, Y, X)
            #         data = data[:, np.newaxis, :, :]
            
            # tifffile.imwrite(output_path, data[0, :, :, :])
            # imageio.mimwrite(output_path, data[0, :, :, :], format='tiff')
            # tif.imsave(output_path, data[0, :, :, :], bigtiff=True)
            # alpha = 0.5  # Change to control contribution
            # blended_data = (alpha * data[0, :, :, :] + (1 - alpha) * data[1, :, :, :])
            # tifffile.imwrite(output_path, data)

            # os.makedirs('tiffs_tifffile', exist_ok=True)

            print('z start', z_start)

            for i in range(data.shape[1]):

                img1_norm = data[0, i, :, :]/np.max(data[0, i, :, :])
                img2_norm = data[1, i, :, :]/np.max(data[1, i, :, :])
                # Create RGB image with zeros (black background)
                rgb_image = np.zeros((data.shape[2], data.shape[3], 3), dtype=np.uint8)

                # Set red channel (img1) - any non-zero value becomes full red
                rgb_image[:, :, 0]=img1_norm*255  # Red channel

                # Set green channel (img2) - any non-zero value becomes full green
                rgb_image[:, :, 1]=  img2_norm*255 # Green channel



                # Create empty blue channel
                blue = np.zeros_like(img1_norm, dtype=np.uint8)
                rgb_image[:, :, 2]=blue

                # Stack channels into RGB image
                # rgb = np.stack((img1*255, img2*255, blue*255), axis=-1)



                # tifffile.imwrite(f'{output_path}/_slice_{i}.tif', rgb_image)
                imageio.imwrite(f'{output_path}/{crop_info}_slice_{z_start+i}.png', rgb_image)


                # Write the images to a multi-page TIFF file
            # with tifffile.TiffWriter(output_path) as tif:
            #     # for image in image_data:
            #     tif.write(data)
                

                
        except Exception as e:
            raise Exception(f"Failed to save 3D data: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TissueSegmentationTool(root)
    root.mainloop() 
