import sys
import time


import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox

from PIL import Image, ImageTk, ImageDraw

import numpy as np
import os
import re
from pathlib import Path
from collections import deque
import time

import torch
import torch.nn as nn
from torchvision import transforms

try:
    sys.stdout.flush()  # Force output
    start_time = time.time()
    
    from nd2reader import ND2Reader
    end_time = time.time()
    print(f"  ✓ nd2reader imported successfully in {end_time - start_time:.2f} seconds")
    has_nd2reader = True
except ImportError as e:
    print(f"  ✗ nd2reader ImportError: {e}")
    has_nd2reader = False    
except Exception as e:
    print(f"  ✗ nd2reader Exception: {e}")
    has_nd2reader = False

print("Step 4: Testing nd2 import...")
try:
    print("  Attempting nd2 import...")
    sys.stdout.flush()  # Force output
    start_time = time.time()
    
    import nd2
    end_time = time.time()
    print(f"  ✓ nd2 imported successfully in {end_time - start_time:.2f} seconds")
    has_nd2 = True
except ImportError as e:
    print(f"  ✗ nd2 ImportError: {e}")
    has_nd2 = False
except Exception as e:
    print(f"  ✗ nd2 Exception: {e}")
    has_nd2 = False

print("Step 5: Testing GUI creation...")
try:
    root = tk.Tk()
    root.title("Import Test Successful")
    root.geometry("300x200")
    
    label = tk.Label(root, text="All imports completed successfully!")
    label.pack(pady=50)
    
    def close_app():
        root.destroy()
    
    button = tk.Button(root, text="Close", command=close_app)
    button.pack()
    
    print("✓ GUI created successfully")
    print("Starting mainloop...")
    root.mainloop()
    print("GUI closed normally")
    
except Exception as e:
    print(f"✗ GUI creation failed: {e}")

print("=== Test completed ===") 