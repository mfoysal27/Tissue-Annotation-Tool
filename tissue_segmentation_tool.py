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

class TissueSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Tissue Segmentation Tool")
        self.root.geometry("1200x800")
        
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
        self.segment_colors[villi] = (220, 20, 60, 255)  # Crimson
        
        villi_subs = [
            ("Epithelium", (255, 55, 95, 255)),  # Lighter Crimson
            ("Basement membrane", (185, 0, 25, 255)),  # Darker Crimson
            ("Villi Lamina propria", (255, 0, 30, 255)),  # Reddish Crimson
            ("Blood vessels", (190, 60, 30, 255)),  # Greenish Crimson
            ("Lumen", (190, 0, 100, 255)),  # Bluish Crimson
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
        self.segment_colors[gland] = (50, 205, 50, 255)  # Lime Green
        
        gland_subs = [
            ("Epithelium", (85, 240, 85, 255)),  # Lighter Lime Green
            ("Basement membrane", (15, 170, 15, 255)),  # Darker Lime Green
            ("Gland Lamina propria", (90, 175, 20, 255)),  # Reddish Green
        ]
        
        for name, color in gland_subs:
            full_name = f"{gland} - {name}"
            self.segments.append(full_name)
            self.segment_hierarchy[gland].append(full_name)
            self.segment_colors[full_name] = color
        
        # 3. Submucosa and its subfeatures
        submucosa = "Submucosa"
        self.segments.append(submucosa)
        self.segment_hierarchy[submucosa] = []
        self.segment_colors[submucosa] = (65, 105, 225, 255)  # Royal Blue
        
        submucosa_subs = [
            ("SMP Ganglion", (100, 140, 255, 255)),  # Lighter Royal Blue
            ("SMP Fiber tract", (30, 70, 190, 255)),  # Darker Royal Blue
            ("Artery", (105, 75, 195, 255)),  # Reddish Royal Blue
            ("Vein", (35, 145, 195, 255)),  # Greenish Royal Blue
            ("Pericryptal space", (35, 75, 255, 255)),  # Bluish Royal Blue
        ]
        
        for name, color in submucosa_subs:
            full_name = f"{submucosa} - {name}"
            self.segments.append(full_name)
            self.segment_hierarchy[submucosa].append(full_name)
            self.segment_colors[full_name] = color
        
        # 4. Circular muscle
        circular_muscle = "Circular muscle"
        self.segments.append(circular_muscle)
        self.segment_hierarchy[circular_muscle] = []
        self.segment_colors[circular_muscle] = (255, 140, 0, 255)  # Dark Orange
        
        # 5. Myenteric plexus and its subfeatures
        myenteric_plexus = "Myenteric plexus"
        self.segments.append(myenteric_plexus)
        self.segment_hierarchy[myenteric_plexus] = []
        self.segment_colors[myenteric_plexus] = (138, 43, 226, 255)  # Blue Violet
        
        myenteric_subs = [
            ("MYP Ganglion", (173, 78, 255, 255)),  # Lighter Blue Violet
            ("MYP Fiber tract", (103, 8, 191, 255)),  # Darker Blue Violet
        ]
        
        for name, color in myenteric_subs:
            full_name = f"{myenteric_plexus} - {name}"
            self.segments.append(full_name)
            self.segment_hierarchy[myenteric_plexus].append(full_name)
            self.segment_colors[full_name] = color
        
        # 6. Longitudinal muscle
        longitudinal_muscle = "Longitudinal muscle"
        self.segments.append(longitudinal_muscle)
        self.segment_hierarchy[longitudinal_muscle] = []
        self.segment_colors[longitudinal_muscle] = (210, 180, 140, 255)  # Tan
    
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
            img = Image.open(file_path).convert('RGB')
            self.original_images = [img]
            self.segmentation_masks = [Image.new('RGBA', img.size, (0, 0, 0, 0))]
            self.current_image_index = 0
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")
    
    def create_annotation_window(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.geometry("1200x800")
        
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
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Organized segment buttons
        self.segment_buttons = [None] * len(self.segments)
        
        # Get all main features (segments without " - ")
        main_features = [segment for segment in self.segments if " - " not in segment]
        
        # For each main feature, create a group with its subfeatures
        for feature_idx, main_feature in enumerate(main_features):
            main_idx = self.segments.index(main_feature)
            
            # Create a frame for the entire group
            feature_frame = ttk.LabelFrame(scrollable_frame, text="")
            feature_frame.pack(fill=tk.X, pady=5, padx=3)
            
            # Add the main feature button
            color = self.segment_colors[main_feature]
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
            main_feature_frame = ttk.Frame(feature_frame)
            main_feature_frame.pack(fill=tk.X, pady=2)
            
            # Prefix with "a.", "b.", etc. based on position
            prefix = chr(97 + feature_idx) + "."
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
        
        # Combine original image and segmentation mask
        combined = original.copy()
        combined.paste(mask, (0, 0), mask)
        
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
        
        # Update UI
        self.create_annotation_window()

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

if __name__ == "__main__":
    root = tk.Tk()
    app = TissueSegmentationTool(root)
    root.mainloop() 