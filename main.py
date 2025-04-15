# Scripts\main.py

"""
This is a work-in-progress tool for extracting individual compartment images from chip tray panoramas. It uses ArUco markers to detect compartment boundaries and can optionally pull metadata via Tesseract OCR if labels are visible in the photo.

Current Features:
Simple folder-based interface — no need to set things up manually

Uses ArUco markers for alignment and compartment detection

Optional OCR to pull metadata straight from the tray

Keeps image quality as high as possible during processing

Shows visual debug output to help troubleshoot detection issues

Automatically names and organizes outputs

Supports common image formats like JPG and PNG

Basic error handling and logging

QAQC step to review extracted compartments

Can skip or process specific images based on filters

Auto-generates an Excel register of processed trays

Supports multiple languages (UI - not OCR)

Checks for script updates automatically

Some advanced config options for users who want more control

Status:
Still under development — some things are a bit rough around the edges and may change depending on what works best in the field...
Happy to hear suggestions or bug reports if you're trying it.

I have to make this visible so that updates can be pushed:

_______________________________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________________________________
MIT License

Copyright (c) 2025 George Symonds

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
_______________________________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________________________________

"""
__version__ = "1.3" 

# ===========================================
# main.py - Chip Tray Processor
# ===========================================

# Import Standard Libraries
import sys
import logging
import os
import json
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import platform
from datetime import datetime
import argparse
from PIL import Image, ImageTk
import numpy as np
import cv2
import pandas as pd
import shutil
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import math
import traceback
import importlib.util
import queue

# Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)

# ===========================================
# Path Configuration
# ===========================================

def get_base_directory():
    """Get the base directory for the application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return base_dir

def setup_project_paths():
    """Set up project directory structure and return path configuration."""
    base_dir = get_base_directory()
    
    # Define main directories
    paths = {
        "base_dir": base_dir,
        "core_dir": os.path.join(base_dir, "core"),
        "gui_dir": os.path.join(base_dir, "gui"),
        "processing_dir": os.path.join(base_dir, "processing"),
        "resources_dir": os.path.join(base_dir, "resources")
    }
    
    # Create directories if they don't exist
    for dir_path in paths.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Add to system path
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    
    for dir_name in ["core", "gui", "processing"]:
        module_path = os.path.join(base_dir, dir_name)
        if module_path not in sys.path:
            sys.path.append(module_path)
    
    return paths

# ===========================================
# Module Imports
# ===========================================


def import_module(module_path, module_name):
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            logging.error(f"Could not load spec for {module_name} from {module_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add to sys.modules to avoid import errors
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logging.error(f"Error importing {module_name} from {module_path}: {str(e)}")
        return None

def load_core_modules(paths):
    """Load core modules from the core directory."""
    modules = {}
    
    # Define required modules and their file names
    core_modules = {
        "file_manager": "file_manager.py",
        "translator": "translator.py",
        "repo_updater": "repo_updater.py",
        "tesseract_manager": "tesseract_manager.py"
    }
    
    for module_name, file_name in core_modules.items():
        module_path = os.path.join(paths["core_dir"], file_name)
        if os.path.exists(module_path):
            modules[module_name] = import_module(module_path, module_name)
        else:
            logging.warning(f"Module file not found: {module_path}")
    
    return modules

def load_gui_modules(paths):
    """Load GUI modules from the gui directory."""
    modules = {}
    
    # Define required modules and their file names
    gui_modules = {
        "dialog_helper": "dialog_helper.py",
        "gui_manager": "gui_manager.py" 
    }
    
    for module_name, file_name in gui_modules.items():
        module_path = os.path.join(paths["gui_dir"], file_name)
        if os.path.exists(module_path):
            modules[module_name] = import_module(module_path, module_name)
        else:
            logging.warning(f"Module file not found: {module_path}")
    
    return modules

def load_processing_modules(paths):
    """Load processing modules from the processing directory."""
    modules = {}
    
    # Define required modules and their file names
    processing_modules = {
        "blur_detector": "blur_detector.py",
        "image_extractor": "image_extractor.py",
        "marker_detection": "marker_detection.py",
        "metadata_extractor": "metadata_extractor.py"
    }
    
    for module_name, file_name in processing_modules.items():
        module_path = os.path.join(paths["processing_dir"], file_name)
        if os.path.exists(module_path):
            modules[module_name] = import_module(module_path, module_name)
        else:
            logging.warning(f"Module file not found: {module_path}")
    
    return modules

# ===========================================
# Main Application Class
# ===========================================

class ChipTrayExtractor:
    """Main application class for the Chip Tray Extractor."""
    
    def __init__(self, paths):
        """Initialize the chip tray extractor with paths and modules."""
        self.paths = paths
        self.root = None
        self.progress_queue = queue.Queue()
        
        # Load modules
        self.modules = {
            "core": load_core_modules(paths),
            "gui": load_gui_modules(paths),
            "processing": load_processing_modules(paths)
        }
        
        # Initialize core components
        self.initialize_core_components()
        
        # Set up GUI
        self.setup_gui()
    
    def initialize_core_components(self):
        """Initialize core components of the application."""
        try:
            # Initialize FileManager
            if "file_manager" in self.modules["core"]:
                self.file_manager = self.modules["core"]["file_manager"].FileManager()
            else:
                logging.error("FileManager module not loaded")
                raise ImportError("FileManager module not loaded")
            
            # Initialize configuration from config.json
            config_path = os.path.join(self.paths["base_dir"], "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    logging.info(f"Loaded configuration from {config_path}")
                except Exception as e:
                    logging.error(f"Error loading config.json: {str(e)}")
                    self.config = self._load_default_config()
            else:
                logging.warning(f"Config file not found at {config_path}, using defaults")
                self.config = self._load_default_config()
                
            # Initialize TranslationManager with file_manager
            if "translator" in self.modules["core"]:
                self.translator = self.modules["core"]["translator"].TranslationManager(file_manager=self.file_manager)
            else:
                logging.error("TranslationManager module not loaded")
                raise ImportError("TranslationManager module not loaded")
            # Set up DialogHelper with translator
            if "dialog_helper" in self.modules["gui"]:
                self.modules["gui"]["dialog_helper"].DialogHelper.set_translator(self.translator)
            
            # Initialize GUIManager
            if "gui_manager" in self.modules["gui"]:
                self.gui_manager = self.modules["gui"]["gui_manager"].GUIManager(self.file_manager)
            else:
                logging.error("GUIManager module not loaded")
                raise ImportError("GUIManager module not loaded")
            
            # Initialize UpdateChecker
            if "repo_updater" in self.modules["core"]:
                self.update_checker = self.modules["core"]["repo_updater"].RepoUpdater(
                    config_manager=self.config_manager,
                    dialog_helper=self.modules["gui"]["dialog_helper"].DialogHelper
                )
            else:
                raise ImportError("Module 'repo_updater' is required but not found in core modules.")


            # Initialize TesseractManager
            if "tesseract_manager" in self.modules["core"]:
                self.tesseract_manager = self.modules["core"]["tesseract_manager"].TesseractManager()
                self.tesseract_manager.extractor = self
                self.tesseract_manager.file_manager = self.file_manager
            
            # Initialize visualization cache
            self.visualization_cache = {}
            
            # Initialize configuration
            self.config = self._load_default_config()
            
            # Update tesseract manager config
            if hasattr(self, 'tesseract_manager'):
                self.tesseract_manager.config = self.config
            
        except Exception as e:
            logging.error(f"Error initializing core components: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    # TODO - ensure that we're loading the config from the .json file in the main.py script
    def _load_default_config(self):
        """Load default configuration settings."""
        return {
            # Output settings
            'output_folder': 'output',
            'save_debug_images': False,
            'output_format': 'tiff',
            'jpeg_quality': 100,

            # Blur detection settings
            'enable_blur_detection': True,
            'blur_threshold': 207.24,
            'blur_roi_ratio': 0.8,
            'flag_blurry_images': False,
            'blurry_threshold_percentage': 10.0,
            'save_blur_visualizations': True,
            
            # ArUco marker settings
            'aruco_dict_type': cv2.aruco.DICT_4X4_1000,
            'corner_marker_ids': [0, 1, 2, 3],
            'compartment_marker_ids': list(range(4, 24)),
            'metadata_marker_ids': [24],
            
            # Processing settings
            'compartment_count': 20,
            'compartment_interval': 1,
            
            # OCR settings
            'enable_ocr': False,  # Will be updated after tesseract check
            'ocr_confidence_threshold': 70.0,
            'ocr_config_options': [
                '--psm 11 --oem 3',
                '--psm 6 --oem 3',
                '--psm 3 --oem 3',
            ],
            'use_metadata_for_filenames': True,
            'metadata_filename_pattern': '{hole_id}_CC_{depth_from}-{depth_to}m',
            'prompt_for_metadata': True,
            'valid_hole_prefixes': ['BA', 'NB', 'SB', 'KM'],
            'enable_prefix_validation': True,
        }
    
    def setup_gui(self):
        """Set up the GUI components."""
        # This will be implemented in create_gui method
        pass
        
    def create_gui(self):
        """Create the main GUI window."""
        self.root = tk.Tk()
        
        # Create the main window using GUIManager
        self.gui_manager.create_main_window(
            self.root, 
            self.translator.translate("Chip Tray Extractor"),
            width=1200,
            height=900
        )
        
        # Additional GUI setup will go here
        
        # Set up timer to check for progress updates
        self.root.after(100, self.check_progress)
        
        # Check for updates at startup if enabled
        if self.config.get('check_for_updates', True):
            self.root.after(2000, self._check_updates_at_startup)
    
    def check_progress(self):
        """Check for progress updates from the processing queue."""
        # Will be implemented based on your existing code
        pass
    
    def show_update_dialog(self):
        """Show the update dialog."""
        try:
            # Import and show the update dialog
            from gui.update_dialog import UpdateDialog
            UpdateDialog.show(self.root, self.config_manager)
        except ImportError:
            # If import fails, show error message
            if hasattr(self, 'modules') and 'gui' in self.modules and 'dialog_helper' in self.modules['gui']:
                self.modules["gui"]["dialog_helper"].DialogHelper.show_message(
                    self.root,
                    "Update Dialog Not Available",
                    "The update dialog module is not available.",
                    message_type="error"
                )
            else:
                logging.error("Update dialog module not available and dialog helper not found")
        except Exception as e:
            logging.error(f"Error showing update dialog: {str(e)}")
            if hasattr(self, 'modules') and 'gui' in self.modules and 'dialog_helper' in self.modules['gui']:
                self.modules["gui"]["dialog_helper"].DialogHelper.show_message(
                    self.root, 
                    "Error", 
                    f"An error occurred while showing the update dialog: {str(e)}",
                    message_type="error"
                )

    def _check_updates_at_startup(self):
        """Check for updates at startup."""
        try:
            if hasattr(self, 'update_checker'):
                # Only show if auto-updates are enabled
                if self.config.get('check_for_updates', True):
                    # Import update dialog
                    from gui.update_dialog import UpdateDialog
                    
                    # Compare versions silently
                    result = self.update_checker.compare_versions()
                    
                    if result.get('update_available', False):
                        # Only show dialog if an update is available
                        self.root.after(500, lambda: UpdateDialog.show(self.root, self.config_manager))
        except ImportError:
            logging.error("Update dialog module not available")
        except Exception as e:
            logging.error(f"Error checking for updates: {str(e)}")
    
    def start_processing(self):
        """Start processing images."""
        # Will be implemented based on your existing code
        pass
    
    def process_folder(self, folder_path):
        """Process all images in a folder."""
        # Will be implemented based on your existing code
        pass
    
    def process_image(self, image_path):
        """Process a single chip tray image."""
        # Will be implemented based on your existing code
        pass
    
    def browse_folder(self):
        """Open folder browser dialog and update the folder entry."""
        # Will be implemented based on your existing code
        pass
    
    def quit_app(self):
        """Close the application."""
        if self.root:
            self.root.destroy()

# ===========================================
# Main Function
# ===========================================

def main():
    """Main entry point of the application."""
    logging.info("Starting Chip Tray Processor")
    
    try:
        # Set up project paths
        paths = setup_project_paths()
        logging.info(f"Project paths set up: {paths}")

        # Create and initialize the main application
        app = ChipTrayExtractor(paths)
        
        # Create and show the GUI
        app.create_gui()
        
        # Start the main event loop
        if app.root:
            app.root.mainloop()
        
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Show error dialog if possible
        try:
            tk.messagebox.showerror("Error", f"An unhandled error occurred:\n{str(e)}")
        except:
            print(f"ERROR: {str(e)}")
        
        sys.exit(1)

# ===========================================
# Entry Point
# ===========================================
if __name__ == "__main__":
    main()
