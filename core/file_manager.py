# core/file_manager.py

"""
Manages file operations for the Chip Tray Extractor.

Handles directory creation, file naming conventions, and saving operations.

File Structure:
C:\\Excel Automation Local Outputs\\Chip Tray Photo Processor\\
├── Program Resources\\translations.csv
├── Program Resources\\config.json

C:\\Excel Automation Local Outputs\\Chip Tray Photo Processor\\Processed\\
├── Blur Analysis\\[HoleID]\\
├── Chip Compartments\\[HoleID]\\With Assays
├── Debug Images\\[HoleID]\\
├── Drill Traces\\
├── Processed Originals\\[HoleID]\\
└── Failed and Skipped Originals\\[HoleID]\\
"""

import os
import re
import cv2
import numpy as np
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union


# TODO: Check any file path construction that might rely on relative paths
# TODO: Ensure logging is properly configured for the module (use global logging settings)
# TODO: Verify any references to config attributes that might need updating


class FileManager:
    """
    Manages file operations for the Chip Tray Extractor.
    
    Handles directory creation, file naming conventions, and saving operations.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the file manager with a base directory.
        
        Args:
            base_dir: Base directory for all outputs, defaults to "C:\\Excel Automation Local Outputs\\Chip Tray Photo Processor"
        """
        # Logger setup first
        self.logger = logging.getLogger(__name__)
        
        # Set up base directory before using it
        self.base_dir = base_dir or Path("C:/Excel Automation Local Outputs/Chip Tray Photo Processor")

        self.processed_dir = os.path.join(self.base_dir, "Processed")
        
        # Now that processed_dir exists, we can define directory structure
        self.dir_structure = {
            "blur_analysis": os.path.join(self.processed_dir, "Blur Analysis"),
            "chip_compartments": os.path.join(self.processed_dir, "Chip Compartments"),
            "debug_images": os.path.join(self.processed_dir, "Debug Images"),
            "drill_traces": os.path.join(self.processed_dir, "Drill Traces"),
            "processed_originals": os.path.join(self.processed_dir, "Processed Originals"),
            "failed_originals": os.path.join(self.processed_dir, "Failed and Skipped Originals")
        }
        
        # Create base directories after everything is set up
        self.create_base_directories()


    # in FileManager Class    
    def check_original_file_processed(self, original_filename: str) -> Optional[Dict[str, Any]]:
        """
        Check if an original file has already been processed by extracting metadata from filename.
        
        Args:
            original_filename: Original input filename
            
        Returns:
            Dictionary with extracted metadata if found, None otherwise
        """
        try:
            # Get the base filename without path and extension
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            
            # Define a pattern to extract metadata from filename
            # Pattern matches formats like "BB1234_40-60_Original" or "BB1234_40-60_Original_Skipped"
            pattern = r'([A-Z]{2}\d{4})_(\d+)-(\d+)_Original(?:_Skipped)?'
            
            # Try to extract metadata directly from filename
            match = re.search(pattern, base_name)
            if match:
                hole_id = match.group(1)
                depth_from = float(match.group(2))
                depth_to = float(match.group(3))
                
                # Check against valid prefixes if the configuration exists and prefix validation is enabled
                valid_prefixes = getattr(self, 'config', {}).get('valid_hole_prefixes', [])
                enable_prefix_validation = getattr(self, 'config', {}).get('enable_prefix_validation', False)
                
                # If prefix validation is enabled, check against valid prefixes
                if enable_prefix_validation and valid_prefixes:
                    prefix = hole_id[:2].upper()
                    if prefix not in valid_prefixes:
                        self.logger.warning(f"Hole ID prefix {prefix} not in valid prefixes: {valid_prefixes}")
                        return None
                
                # Determine if the file was previously skipped
                is_skipped = '_Skipped' in base_name
                
                self.logger.info(f"Found metadata in filename: {hole_id}, {depth_from}-{depth_to}")
                return {
                    'hole_id': hole_id,
                    'depth_from': depth_from,
                    'depth_to': depth_to,
                    'confidence': 100.0 if not is_skipped else 90.0,  # Slightly lower confidence for skipped files
                    'from_filename': True,
                    'previously_skipped': is_skipped
                }
            
            # No match found
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking for previously processed file: {str(e)}")
            return None
    
    def save_compartment(self, 
                        image: np.ndarray, 
                        hole_id: str, 
                        compartment_num: int,
                        has_data: bool = False,
                        output_format: str = "tiff") -> str:
        """
        Save a compartment image.
        
        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number
            has_data: Whether the image has data columns
            output_format: Output image format
            
        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.get_hole_dir("chip_compartments", hole_id)
            
            # Create filename with 3-digit compartment number (001, 002, etc.)
            if has_data:
                filename = f"{hole_id}_CC_{compartment_num:03d}_Data.{output_format}"
            else:
                filename = f"{hole_id}_CC_{compartment_num:03d}.{output_format}"
            
            # Full path
            file_path = os.path.join(save_dir, filename)
            
            # Save the image
            if output_format.lower() == "jpg":
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                cv2.imwrite(file_path, image)
            
            self.logger.info(f"Saved compartment image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving compartment image: {str(e)}")
            return None

    def save_compartment_with_data(self, 
                                image: np.ndarray, 
                                hole_id: str, 
                                compartment_num: int,
                                output_format: str = "tiff") -> str:
        """
        Save a compartment image with data columns to a 'With_Data' subfolder.
        
        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number
            output_format: Output image format
            
        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory and create a "With_Data" subfolder
            base_dir = self.get_hole_dir("chip_compartments", hole_id)
            with_data_dir = os.path.join(base_dir, "With_Data")
            os.makedirs(with_data_dir, exist_ok=True)
            
            # Create filename with 3-digit compartment number (001, 002, etc.)
            filename = f"{hole_id}_CC_{compartment_num:03d}_Data.{output_format}"
            
            # Full path
            file_path = os.path.join(with_data_dir, filename)
            
            # Save the image
            if output_format.lower() == "jpg":
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                cv2.imwrite(file_path, image)
            
            self.logger.info(f"Saved compartment with data image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving compartment with data image: {str(e)}")
            return None
    
    
    
    def rename_debug_files(self, 
                        original_filename: str, 
                        hole_id: Optional[str], 
                        depth_from: Optional[float], 
                        depth_to: Optional[float]) -> None:
        """
        Rename debug files for a specific image after successful metadata extraction.
        
        Args:
            original_filename: Original input image filename
            hole_id: Extracted hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
        """
        try:
            if not (hole_id and depth_from is not None and depth_to is not None):
                return
            
            # Get the debug directory for this hole ID
            debug_dir = self.get_hole_dir("debug_images", hole_id)
            
            # Find all debug files for this image
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            
            try:
                debug_files = [
                    f for f in os.listdir(debug_dir) 
                    if f.startswith(base_name) and f.endswith('.jpg')
                ]
            except FileNotFoundError:
                # Directory might not exist yet
                os.makedirs(debug_dir, exist_ok=True)
                debug_files = []
                self.logger.info(f"Created debug directory for {hole_id}")
            
            # Also look for temp debug images in the Unidentified folder
            temp_debug_dir = os.path.join(self.dir_structure["debug_images"], "Unidentified")
            if os.path.exists(temp_debug_dir):
                try:
                    temp_files = [
                        f for f in os.listdir(temp_debug_dir)
                        if f.startswith(base_name) and f.endswith('.jpg')
                    ]
                    
                    # Move and rename temp files
                    for old_filename in temp_files:
                        # Extract the step name from the old filename
                        if '_' in old_filename:
                            step_name = old_filename.split('_', 1)[1].replace('.jpg', '')
                            
                            # Generate new filename with metadata
                            new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{step_name}.jpg"
                            
                            old_path = os.path.join(temp_debug_dir, old_filename)
                            new_path = os.path.join(debug_dir, new_filename)
                            
                            try:
                                # Ensure debug directory exists
                                os.makedirs(debug_dir, exist_ok=True)
                                
                                # Copy the file (use copy instead of move for safety)
                                shutil.copy2(old_path, new_path)
                                
                                # Only remove original after successful copy
                                os.remove(old_path)
                                
                                self.logger.info(f"Moved debug file from temp location: {old_filename} -> {new_filename}")
                            except Exception as e:
                                self.logger.error(f"Error moving temp debug file {old_filename}: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing temp debug files: {e}")
            
            # Rename files in the debug directory
            for old_filename in debug_files:
                try:
                    # Extract the step name from the old filename
                    step_parts = old_filename.split('_')
                    if len(step_parts) >= 2:
                        step_name = step_parts[-1].replace('.jpg', '')
                        
                        # Generate new filename with metadata
                        new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{step_name}.jpg"
                        
                        old_path = os.path.join(debug_dir, old_filename)
                        new_path = os.path.join(debug_dir, new_filename)
                        
                        # Rename the file
                        os.rename(old_path, new_path)
                        self.logger.info(f"Renamed debug file: {old_filename} -> {new_filename}")
                except Exception as e:
                    self.logger.error(f"Error renaming debug file {old_filename}: {e}")
        except Exception as e:
            self.logger.error(f"Error in rename_debug_files: {e}")

    def save_temp_debug_image(self, image: np.ndarray, original_filename: str, debug_type: str) -> str:
        """
        Save a debug image without proper hole/depth metadata.
        
        Args:
            image: Image to save
            original_filename: Original input filename used to generate base name
            debug_type: Type of debug image
            
        Returns:
            Full path to the saved image
        """
        try:
            # Use the centralized directory structure
            temp_dir = os.path.join(self.dir_structure["debug_images"], "Unidentified")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate filename from original file
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            filename = f"{base_name}_{debug_type}.jpg"
            full_path = os.path.join(temp_dir, filename)
            
            # Save the image with moderate compression
            cv2.imwrite(full_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            self.logger.info(f"Saved temporary debug image: {full_path}")
            return full_path
            
        except Exception as e:
            self.logger.error(f"Error saving temporary debug image: {str(e)}")
            
            # Create a fallback directory if needed
            fallback_dir = os.path.join(self.base_dir, "Temp_Debug")
            os.makedirs(fallback_dir, exist_ok=True)
            
            # Generate a unique fallback filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_filename = f"debug_{timestamp}_{debug_type}.jpg"
            fallback_path = os.path.join(fallback_dir, fallback_filename)
            
            # Try to save to fallback location
            try:
                cv2.imwrite(fallback_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.logger.info(f"Saved debug image to fallback location: {fallback_path}")
                return fallback_path
            except Exception as fallback_error:
                self.logger.error(f"Failed to save debug image to fallback location: {str(fallback_error)}")
                return ""
    
    # TODO - let the user select the base directory during the first time run on the machine and save it in a config file  in 'Program Resources' - use the TODOs in the script updater to build this...
    def create_base_directories(self) -> None:
        """Create the base directory structure."""
        try:
            # Create the processed directory
            os.makedirs(self.processed_dir, exist_ok=True)
            
            # Create each subdirectory
            for dir_path in self.dir_structure.values():
                os.makedirs(dir_path, exist_ok=True)
                
            self.logger.info(f"Base directory structure: {self.base_dir}")
        except Exception as e:
            self.logger.error(f"Error creating directory structure: {str(e)}")
            raise
    
    def get_hole_dir(self, dir_type: str, hole_id: str) -> str:
        """
        Get the directory path for a specific hole ID.
        
        Args:
            dir_type: Directory type (must be one of the keys in dir_structure)
            hole_id: Hole ID
            
        Returns:
            Full path to the hole directory
        """
        if dir_type not in self.dir_structure:
            raise ValueError(f"Invalid directory type: {dir_type}")
        
        # Create hole-specific directory
        hole_dir = os.path.join(self.dir_structure[dir_type], hole_id)
        os.makedirs(hole_dir, exist_ok=True)
        
        return hole_dir
    
    
    def save_blur_analysis(self, 
                        image: np.ndarray, 
                        hole_id: str, 
                        compartment_num: int) -> str:
        """
        Save a blur analysis image.
        
        Args:
            image: Image to save
            hole_id: Hole ID
            compartment_num: Compartment number
            
        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.get_hole_dir("blur_analysis", hole_id)
            
            # Create filename
            filename = f"{hole_id}_{compartment_num}_blur_analysis.jpg"
            
            # Full path
            file_path = os.path.join(save_dir, filename)
            
            # Save the image
            cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            self.logger.info(f"Saved blur analysis image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving blur analysis image: {str(e)}")
            return None

    def save_debug_image(self, 
                       image: np.ndarray, 
                       hole_id: str, 
                       depth_from: float,
                       depth_to: float,
                       image_type: str) -> str:
        """
        Save a debug image.
        
        Args:
            image: Image to save
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            image_type: Type of debug image
            
        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.get_hole_dir("debug_images", hole_id)
            
            # Create filename
            filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Debug_{image_type}.jpg"
            
            # Full path
            file_path = os.path.join(save_dir, filename)
            
            # Save the image
            cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            self.logger.info(f"Saved debug image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving debug image: {str(e)}")
            return None
    
    def save_drill_trace(self, 
                       image: np.ndarray, 
                       hole_id: str) -> str:
        """
        Save a drill trace image.
        
        Args:
            image: Image to save
            hole_id: Hole ID
            
        Returns:
            Path to the saved file
        """
        try:
            # Get the appropriate directory
            save_dir = self.dir_structure["drill_traces"]
            
            # Create filename
            filename = f"{hole_id}_Trace.tiff"
            
            # Full path
            file_path = os.path.join(save_dir, filename)
            
            # Save the image
            cv2.imwrite(file_path, image)
            
            self.logger.info(f"Saved drill trace image: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving drill trace image: {str(e)}")
            return None
    
    # TODO - try to move a copy to the onedrive folder from the GUI (self.onedrive_manager._processed_originals_path) before moving it to the local processed folder
    def move_original_file(self, 
                        source_path: str, 
                        hole_id: str, 
                        depth_from: float,
                        depth_to: float,
                        is_processed: bool = True) -> str:
        """
        Move an original file to the appropriate directory.
        
        Args:
            source_path: Path to the source file
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            is_processed: Whether the file was successfully processed
            
        Returns:
            Path to the new file location
        """
        try:
            # Check if source file exists
            if not os.path.exists(source_path):
                self.logger.warning(f"Source file does not exist: {source_path}")
                return None

            # Determine the target directory
            if is_processed:
                target_dir = self.get_hole_dir("processed_originals", hole_id)
            else:
                target_dir = self.get_hole_dir("failed_originals", hole_id)
            
            # Get original file extension
            _, ext = os.path.splitext(source_path)
            
            # Create new filename with appropriate suffix
            status_suffix = "" if is_processed else "_Skipped"
            new_filename = f"{hole_id}_{int(depth_from)}-{int(depth_to)}_Original{status_suffix}{ext}"
            
            # Full target path
            target_path = os.path.join(target_dir, new_filename)
            
            # Check if file already exists, add number if needed
            counter = 1
            base_name = new_filename
            while os.path.exists(target_path):
                name_parts = os.path.splitext(base_name)
                new_filename = f"{name_parts[0]}_{counter}{name_parts[1]}"
                target_path = os.path.join(target_dir, new_filename)
                counter += 1
            
            # Copy the file first, then delete original after successful copy
            try:
                # Make sure target directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, target_path)
                
                # Delete original file only after successful copy
                if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                    os.remove(source_path)
                    self.logger.info(f"Moved original file to: {target_path}")
                else:
                    self.logger.error(f"Failed to copy original file: {source_path} -> {target_path}")
                    return None
                    
            except (PermissionError, OSError) as e:
                self.logger.error(f"Permission or OS error moving file: {str(e)}")
                # Try alternative method with shutil.move
                try:
                    shutil.move(source_path, target_path)
                    self.logger.info(f"Moved original file with shutil.move: {target_path}")
                except Exception as move_e:
                    self.logger.error(f"Failed to move with shutil.move: {str(move_e)}")
                    return None
            
            return target_path
            
        except Exception as e:
            self.logger.error(f"Error moving original file: {str(e)}")
            return None
