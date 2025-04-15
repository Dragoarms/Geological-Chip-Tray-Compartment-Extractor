# core/tesseract_manager.py

"""
    Manages Tesseract OCR detection, installation guidance, and OCR functionality.
    Designed to gracefully handle cases where Tesseract is not installed.
"""
    

import os
import platform
import re
import math
import importlib.util
import webbrowser
import traceback
import numpy as np
import cv2
from typing import Dict, Optional, Any
import logging
logger = logging.getLogger(__name__)
import tkinter as tk
from tkinter import ttk
import queue
from gui.dialog_helper import DialogHelper



class TesseractManager:

    def __init__(self):
        """Initialize the Tesseract manager with default paths and settings."""
        self.is_available = False
        self.version = None
        self.pytesseract = None
        self.install_instructions = {
            'Windows': 'https://github.com/UB-Mannheim/tesseract/wiki',
            }
        
        # Try to detect Tesseract
        self.detect_tesseract()
    
    def detect_tesseract(self) -> bool:
        """
        Detect if Tesseract OCR is installed and available in the system.
        
        Returns:
            bool: True if Tesseract is available, False otherwise
        """
        # First check if pytesseract can be imported
        try:
            # Try to import pytesseract
            if importlib.util.find_spec("pytesseract") is None:
                logger.warning("pytesseract package not found")
                return False
            
            # Import pytesseract for OCR functionality
            import pytesseract
            from pytesseract import Output
            self.pytesseract = pytesseract
            
            # Try various methods to find Tesseract executable
            if platform.system() == 'Windows':
                # Common installation locations on Windows
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR',
                    r'C:\Program Files (x86)\Tesseract-OCR',
                    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Tesseract-OCR'),
                    os.path.join(os.environ.get('APPDATA', ''), 'Tesseract-OCR')
                ]
                
                # Find the first valid path
                tesseract_path = None
                for path in possible_paths:
                    exe_path = os.path.join(path, 'tesseract.exe')
                    if os.path.exists(exe_path):
                        tesseract_path = path
                        break
                
                if tesseract_path:
                    # Add to system PATH if it exists
                    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + tesseract_path
                    pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_path, 'tesseract.exe')
                    logger.info(f"Found Tesseract at: {tesseract_path}")
            
            # Test if Tesseract works by getting version
            try:
                self.version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract OCR version {self.version} detected")
                self.is_available = True
                return True
            except Exception as e:
                logger.warning(f"Tesseract is installed but not working correctly: {str(e)}")
                return False
                
        except ImportError:
            logger.warning("Failed to import pytesseract module")
            return False
        except Exception as e:
            logger.warning(f"Error detecting Tesseract: {str(e)}")
            return False
    
    def get_installation_instructions(self) -> str:
        """
        Get installation instructions for the current platform.
        
        Returns:
            str: Installation instructions for Tesseract OCR
        """
        system = platform.system()
        base_instructions = (
            "To use OCR features, you need to:\n\n"
            "1. Install Tesseract OCR for your platform\n"
            "2. Install the pytesseract Python package: pip install pytesseract\n\n"
        )
        
        if system in self.install_instructions:
            platform_instructions = f"For {system} systems: {self.install_instructions[system]}"
        else:
            platform_instructions = "Visit https://github.com/tesseract-ocr/tesseract for installation instructions."
        
        return base_instructions + platform_instructions
    
    def show_installation_dialog(self, parent: Optional[tk.Tk] = None) -> None:
        """
        Show a dialog with installation instructions for Tesseract OCR.
        
        Args:
            parent: Optional parent Tkinter window
        """
        instructions = self.get_installation_instructions()
        system = platform.system()
        
        # If no parent window, just log the instructions
        if parent is None:
            logger.info(DialogHelper.t(f"Tesseract OCR not available. {instructions}"))
            return
        
        # Create a custom dialog
        dialog = tk.Toplevel(parent)
        dialog.title(DialogHelper.t("Tesseract OCR Required"))
        dialog.grab_set()  # Make dialog modal
        
        # Add icon and header
        header_frame = ttk.Frame(dialog, padding="10")
        header_frame.pack(fill=tk.X)
        
        # Use system-specific icons
        if system == 'Windows':
            try:
                dialog.iconbitmap(default='')  # Default Windows icon
            except:
                pass
                
        # Header label
        header_label = ttk.Label(
            header_frame, 
            text=DialogHelper.t("Tesseract OCR Required for Text Recognition"),
            font=("Arial", 12, "bold")
        )
        header_label.pack(pady=10)
        
        # Instructions text
        text_frame = ttk.Frame(dialog, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        instructions_text = tk.Text(text_frame, wrap=tk.WORD, height=10)
        instructions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        instructions_text.insert(tk.END, instructions)
        instructions_text.config(state=tk.DISABLED)
        
        # Scrollbar for text
        scrollbar = ttk.Scrollbar(instructions_text, command=instructions_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        instructions_text.config(yscrollcommand=scrollbar.set)
        
        # Action buttons
        button_frame = ttk.Frame(dialog, padding="10")
        button_frame.pack(fill=tk.X)
        
        # Platform-specific install button
        if system in self.install_instructions:
            install_url = self.install_instructions[system].split(' ')[0]
            install_button = ttk.Button(
                button_frame, 
                text=DialogHelper.t(f"Download for {system}"), 
                command=lambda: webbrowser.open(install_url)
            )
            install_button.pack(side=tk.LEFT, padx=5)
        
        close_button = ttk.Button(
            button_frame, 
            text=DialogHelper.t("Continue without OCR"), 
            command=dialog.destroy
        )
        close_button.pack(side=tk.RIGHT, padx=5)
    
    # Function to extract patterns from OCR
    def extract_metadata_with_composite(self, 
                                    image: np.ndarray, 
                                    markers: Dict[int, np.ndarray], 
                                    original_filename: Optional[str] = None,
                                    progress_queue: Optional[queue.Queue] = None) -> Dict[str, Any]:
        """
        Extract metadata using a composite approach - combining multiple preprocessing methods
        on the extracted label regions and voting on the most likely correct values.
        
        Args:
            image: Input image
            markers: Detected ArUco marker positions
            original_filename: Original image filename for debug purposes
            progress_queue: Optional queue for reporting progress
        
        Returns:
            Metadata dictionary with extracted information
        """

        
        try:
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Check if we have necessary markers
            if not markers:
                logger.warning("No markers found for metadata region extraction")
                return {
                    'hole_id': None,
                    'depth_from': None,
                    'depth_to': None,
                    'confidence': 0.0,
                    'metadata_region': None
                }
            
            # Define metadata region using markers
            metadata_region = None  # Initialize the variable
            hole_id_region = None
            depth_region = None
            
            # Look for marker ID 24 which is between hole ID and depth labels
            if 24 in markers:
                # Use marker 24 as a reference point
                marker24_corners = markers[24]
                marker24_center_x = int(np.mean(marker24_corners[:, 0]))
                marker24_center_y = int(np.mean(marker24_corners[:, 1]))
                
                # The hole ID label is likely above marker 24
                hole_id_region_y1 = max(0, marker24_center_y - 130)  # 130px above marker center
                hole_id_region_y2 = max(0, marker24_center_y - 20)   # 20px above marker center
                
                # The depth label is likely below marker 24
                depth_region_y1 = min(h, marker24_center_y + 20)    # 20px below marker center
                depth_region_y2 = min(h, marker24_center_y + 150)   # 150px below marker center
                
                # Horizontal range for both regions (centered on marker 24)
                region_x1 = max(0, marker24_center_x - 100)
                region_x2 = min(w, marker24_center_x + 100)
                
                # Extract the hole ID and depth regions
                hole_id_region = image[hole_id_region_y1:hole_id_region_y2, region_x1:region_x2].copy()
                depth_region = image[depth_region_y1:depth_region_y2, region_x1:region_x2].copy()
                
                # Create visualization
                viz_image = image.copy()
                # Draw hole ID region in blue
                cv2.rectangle(viz_image, 
                            (region_x1, hole_id_region_y1), 
                            (region_x2, hole_id_region_y2), 
                            (255, 0, 0), 2)
                # Draw marker 24 in purple
                cv2.circle(viz_image, (marker24_center_x, marker24_center_y), 10, (255, 0, 255), -1)
                # Draw depth region in green
                cv2.rectangle(viz_image, 
                            (region_x1, depth_region_y1), 
                            (region_x2, depth_region_y2), 
                            (0, 255, 0), 2)
                            
                # Create a metadata region by combining both regions
                metadata_region = image[hole_id_region_y1:depth_region_y2, region_x1:region_x2].copy()
                metadata_region_viz = viz_image
                
            else:
                # Fallback: use the bottom portion of the image
                logger.warning("Marker 24 not found, using fallback region detection")
                
                # Define a larger metadata region at the bottom of the image
                metadata_top = int(h * 0.65)  # Start at 65% down the image
                metadata_left = int(w * 0.03)  # 3% from left edge
                metadata_right = int(w * 0.5)  # Use left half of the image
                metadata_bottom = h
                
                # Extract the whole metadata region
                metadata_region = image[
                    metadata_top:metadata_bottom, 
                    metadata_left:metadata_right
                ].copy()
                
                # Split this region for hole ID (top) and depth (bottom)
                middle_y = (metadata_bottom - metadata_top) // 2 + metadata_top
                
                hole_id_region = image[metadata_top:middle_y, metadata_left:metadata_right].copy()
                depth_region = image[middle_y:metadata_bottom, metadata_left:metadata_right].copy()
                
                # Create visualization
                viz_image = image.copy()
                cv2.rectangle(viz_image, 
                            (metadata_left, metadata_top), 
                            (metadata_right, metadata_bottom), 
                            (0, 255, 0), 2)
                cv2.line(viz_image,
                        (metadata_left, middle_y),
                        (metadata_right, middle_y),
                        (255, 0, 0), 2)
                        
                metadata_region_viz = viz_image
            
            # ALWAYS save visualization to memory cache regardless of debug settings
            # This is crucial for duplicate detection dialog
            if hasattr(self, 'extractor') and hasattr(self.extractor, 'visualization_cache'):
                if 'current_processing' not in self.extractor.visualization_cache:
                    self.extractor.visualization_cache['current_processing'] = {}

                self.extractor.visualization_cache['current_processing'].update({
                    'metadata_region': metadata_region,
                    'metadata_region_viz': metadata_region_viz,
                    'hole_id_region': hole_id_region,
                    'depth_region': depth_region
                })

            
            # Process hole ID and depth regions separately
            hole_id_results = []
            depth_results = []
            
            if hole_id_region is not None and depth_region is not None:
                # Create preprocessing methods specifically tailored for label text
                # From your images, we can see both original color and various binary versions
                preprocessing_methods = [
                    ("original", lambda img: img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                    ("threshold1", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        127, 255, cv2.THRESH_BINARY)[1]),
                    ("threshold2", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        100, 255, cv2.THRESH_BINARY)[1]),
                    ("threshold3", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        150, 255, cv2.THRESH_BINARY)[1]),
                    ("adaptive", lambda img: cv2.adaptiveThreshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("otsu", lambda img: cv2.threshold(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("canny", lambda img: cv2.Canny(
                        img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                        50, 150))
                ]
                
                # Create composite images for hole ID and depth
                hole_id_composite_images = []
                depth_composite_images = []
                
                # Process hole ID region
                for method_name, method_func in preprocessing_methods:
                    try:
                        # Process image
                        processed_img = method_func(hole_id_region)
                        
                        # Run OCR specifically for hole ID pattern
                        # Use different PSM modes that are optimal for single line text (7) or single word (8)
                        for psm in [7, 8, 6]:
                            config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                            text = self.pytesseract.image_to_string(processed_img, config=config).strip()
                            
                            # Try to extract hole ID pattern
                            match = re.search(r'([A-Z]{2}\d{4})', text.replace(" ", ""))
                            hole_id = match.group(1) if match else None
                            
                            if hole_id:
                                hole_id_results.append({
                                    'hole_id': hole_id,
                                    'method': method_name,
                                    'psm': psm,
                                    'text': text,
                                    'processed_image': processed_img
                                })
                                
                                # Add image to composite with text overlay
                                labeled_img = processed_img.copy()
                                if len(labeled_img.shape) == 2:
                                    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
                                
                                # Add text overlay
                                cv2.putText(
                                    labeled_img,
                                    f"{method_name} PSM{psm}: {hole_id}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2
                                )
                                
                                hole_id_composite_images.append(labeled_img)
                                
                                # If we found a valid hole ID, no need to try other PSM modes
                                break
                        
                    except Exception as e:
                        logger.error(f"Error processing hole ID with {method_name}: {str(e)}")
                
                # Process depth region
                for method_name, method_func in preprocessing_methods:
                    try:
                        # Process image
                        processed_img = method_func(depth_region)
                        
                        # Run OCR specifically for depth pattern
                        # Try different PSM modes
                        for psm in [7, 6, 3]:
                            config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789-."
                            text = self.pytesseract.image_to_string(processed_img, config=config).strip()
                            
                            # Try to extract depth pattern
                            match = re.search(r'(\d+)[\s\-–—to]+(\d+)', text)
                            if match:
                                depth_from = float(match.group(1))
                                depth_to = float(match.group(2))
                                
                                depth_results.append({
                                    'depth_from': depth_from,
                                    'depth_to': depth_to,
                                    'method': method_name,
                                    'psm': psm,
                                    'text': text,
                                    'processed_image': processed_img
                                })
                                
                                # Add image to composite with text overlay
                                labeled_img = processed_img.copy()
                                if len(labeled_img.shape) == 2:
                                    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_GRAY2BGR)
                                
                                # Add text overlay
                                cv2.putText(
                                    labeled_img,
                                    f"{method_name} PSM{psm}: {depth_from}-{depth_to}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2
                                )
                                
                                depth_composite_images.append(labeled_img)
                                
                                # If we found valid depths, no need to try other PSM modes
                                break
                        
                    except Exception as e:
                        logger.error(f"Error processing depth with {method_name}: {str(e)}")
                
                # Create and save composite images
                if hole_id_composite_images and hasattr(self, 'config') and self.config.get('save_debug_images', False):
                    file_manager = getattr(self, 'file_manager', None)
                    if file_manager is not None:
                        # Create a grid layout for hole ID images
                        grid_size = math.ceil(math.sqrt(len(hole_id_composite_images)))
                        grid_h = grid_size
                        grid_w = grid_size
                        
                        # Find max dimensions
                        max_h = max(img.shape[0] for img in hole_id_composite_images)
                        max_w = max(img.shape[1] for img in hole_id_composite_images)
                        
                        # Create grid
                        grid = np.ones((max_h * grid_h, max_w * grid_w, 3), dtype=np.uint8) * 255
                        
                        # Place images
                        for i, img in enumerate(hole_id_composite_images):
                            if i >= grid_h * grid_w:
                                break
                                
                            row = i // grid_w
                            col = i % grid_w
                            
                            y = row * max_h
                            x = col * max_w
                            
                            h, w = img.shape[:2]
                            grid[y:y+h, x:x+w] = img
                        
                        # Save composite
                        file_manager.save_temp_debug_image(
                            grid,
                            original_filename,
                            "hole_id_composite"
                        )
                
                # Same for depth composite
                if depth_composite_images and hasattr(self, 'config') and self.config.get('save_debug_images', False):
                    file_manager = getattr(self, 'file_manager', None)
                    if file_manager is not None:
                        # Create a grid layout for depth images
                        grid_size = math.ceil(math.sqrt(len(depth_composite_images)))
                        grid_h = grid_size
                        grid_w = grid_size
                        
                        # Find max dimensions
                        max_h = max(img.shape[0] for img in depth_composite_images)
                        max_w = max(img.shape[1] for img in depth_composite_images)
                        
                        # Create grid
                        grid = np.ones((max_h * grid_h, max_w * grid_w, 3), dtype=np.uint8) * 255
                        
                        # Place images
                        for i, img in enumerate(depth_composite_images):
                            if i >= grid_h * grid_w:
                                break
                                
                            row = i // grid_w
                            col = i % grid_w
                            
                            y = row * max_h
                            x = col * max_w
                            
                            h, w = img.shape[:2]
                            grid[y:y+h, x:x+w] = img
                        
                        # Save composite
                        file_manager.save_temp_debug_image(
                            grid,
                            original_filename,
                            "depth_composite"
                        )
            
            # Vote on results
            # For hole ID
            hole_id_votes = {}
            for result in hole_id_results:
                hole_id = result.get('hole_id')
                if hole_id:
                    hole_id_votes[hole_id] = hole_id_votes.get(hole_id, 0) + 1
            
            # For depths
            depth_from_votes = {}
            depth_to_votes = {}
            for result in depth_results:
                depth_from = result.get('depth_from')
                depth_to = result.get('depth_to')
                
                if depth_from is not None:
                    # Round to nearest integer for voting
                    depth_from_int = int(round(depth_from))
                    depth_from_votes[depth_from_int] = depth_from_votes.get(depth_from_int, 0) + 1
                
                if depth_to is not None:
                    # Round to nearest integer for voting
                    depth_to_int = int(round(depth_to))
                    depth_to_votes[depth_to_int] = depth_to_votes.get(depth_to_int, 0) + 1
            
            # Get the most voted values
            final_hole_id = None
            if hole_id_votes:
                final_hole_id = max(hole_id_votes.items(), key=lambda x: x[1])[0]
            
            final_depth_from = None
            if depth_from_votes:
                final_depth_from = max(depth_from_votes.items(), key=lambda x: x[1])[0]
            
            final_depth_to = None
            if depth_to_votes:
                final_depth_to = max(depth_to_votes.items(), key=lambda x: x[1])[0]
            
            # Additional validation for depths
            if final_depth_from is not None and final_depth_to is not None:
                # Make sure depth_to > depth_from
                if final_depth_to <= final_depth_from:
                    logger.warning(f"Invalid depth range detected: {final_depth_from}-{final_depth_to}")
                    
                    # Try to correct: standard pattern is 20m intervals
                    if abs(final_depth_to - final_depth_from) < 5:
                        # Likely a small OCR error, set depth_to to depth_from + 20
                        final_depth_to = final_depth_from + 20
                        logger.info(f"Corrected depth range to: {final_depth_from}-{final_depth_to}")
            
            # Calculate confidence based on vote consistency
            confidence = 0.0
            
            # Hole ID confidence
            if final_hole_id and hole_id_votes:
                # Percentage of agreement
                agreement = min(100.0, hole_id_votes[final_hole_id] / len(hole_id_results) * 100 if hole_id_results else 0)
                confidence += min(agreement, 50.0)  # Max 50 points from hole ID

            # Depth confidence
            if final_depth_from is not None and final_depth_to is not None and depth_from_votes and depth_to_votes:
                # Percentage of agreement for each value
                # Make sure the keys exist in the dictionaries
                from_agreement = min(100.0, depth_from_votes.get(final_depth_from, 0) / len(depth_results) * 100 if depth_results else 0)
                to_agreement = min(100.0, depth_to_votes.get(final_depth_to, 0) / len(depth_results) * 100 if depth_results else 0)
                
                # Average the agreements
                avg_agreement = (from_agreement + to_agreement) / 2
                confidence += min(avg_agreement, 50.0)  # Max 50 points from depths

            # Ensure final confidence never exceeds 100%
            confidence = min(confidence, 100.0)
            

            
            # Final result
            result = {
                'hole_id': final_hole_id,
                'depth_from': float(final_depth_from) if final_depth_from is not None else None,
                'depth_to': float(final_depth_to) if final_depth_to is not None else None,
                'confidence': confidence,
                'metadata_region': metadata_region,
                'metadata_region_viz': metadata_region_viz,
                'ocr_text': f"Composite OCR: ID={final_hole_id}, Depth={final_depth_from}-{final_depth_to}"
            }
            
            if (
                hasattr(self, 'extractor') and 
                hasattr(self.extractor, 'visualization_cache') and
                'current_processing' in self.extractor.visualization_cache
            ):
                boundaries_viz = self.extractor.visualization_cache['current_processing'].get('compartment_boundaries_viz')
                if boundaries_viz is not None:
                    result['compartment_boundaries_viz'] = boundaries_viz

            # Log final results
            logger.info("------------------------------------------------")
            logger.info("OCR results:")
            if final_hole_id:
                logger.info(f"Hole ID: {final_hole_id} (votes: {hole_id_votes.get(final_hole_id, 0)}/{len(hole_id_results) or 1})")
            else:
                logger.info("Hole ID: None")
                
            if final_depth_from is not None and final_depth_to is not None:
                logger.info(f"Depth range: {final_depth_from}-{final_depth_to} "
                        f"(votes: {depth_from_votes.get(final_depth_from, 0)}/{len(depth_results) or 1}, "
                        f"{depth_to_votes.get(final_depth_to, 0)}/{len(depth_results) or 1})")
            else:
                logger.info("Depth range: None")
                
            logger.info(f"Confidence: {confidence:.1f}%")
            logger.info("------------------------------------------------")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in composite OCR extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'hole_id': None,
                'depth_from': None,
                'depth_to': None,
                'confidence': 0.0,
                'metadata_region': None
            }
    
    def _remove_borders(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and remove black borders around chip tray labels.
        
        Args:
            gray_image: Grayscale input image
        
        Returns:
            Border-removed image or None if no borders detected
        """
        try:
            # Create binary image to identify potential borders
            _, border_thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(border_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we have contours, find the largest one (likely the label boundary)
            if contours:
                # Sort contours by area, largest first
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Get the largest contour
                largest_contour = contours[0]
                
                # Check if this contour is sufficiently large and looks like a border
                # (covers at least 70% of the perimeter of the image)
                image_perimeter = 2 * (gray_image.shape[0] + gray_image.shape[1])
                contour_perimeter = cv2.arcLength(largest_contour, True)
                perimeter_ratio = contour_perimeter / image_perimeter
                
                if perimeter_ratio > 0.7:
                    logger.info(f"Detected potential border (perimeter ratio: {perimeter_ratio:.2f})")
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Ensure bounding rect covers most of the image (indicating a border)
                    image_area = gray_image.shape[0] * gray_image.shape[1]
                    rect_area = w * h
                    area_ratio = rect_area / image_area
                    
                    if area_ratio > 0.8:
                        logger.info(f"Confirmed border detection (area ratio: {area_ratio:.2f})")
                        
                        # Create a mask for everything inside the contour
                        mask = np.zeros_like(gray_image)
                        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
                        
                        # Create a smaller mask to exclude the border itself
                        # Use adaptive kernel size based on image dimensions
                        border_width = min(max(3, min(w, h) // 30), 10)  # Between 3 and 10 pixels
                        kernel = np.ones((border_width, border_width), np.uint8)
                        eroded_mask = cv2.erode(mask, kernel, iterations=1)
                        
                        # Extract the label content without the border
                        inner_content = np.ones_like(gray_image) * 255  # White background
                        inner_content[eroded_mask > 0] = gray_image[eroded_mask > 0]
                        
                        return inner_content
            
            # No borders detected or removed
            return None
            
        except Exception as e:
            logger.error(f"Error during border removal: {str(e)}")
            return None

