
"""
Chip Tray Extractor with OCR capabilities

A tool to extract individual compartment images from panoramic chip tray photos 
using ArUco markers for alignment and compartment boundary detection.
Additionally extracts text metadata using Tesseract OCR when available.

Features:
- Simple GUI for folder selection and processing options
- ArUco marker detection for precise compartment extraction
- OCR metadata extraction with Tesseract (optional)
- High-quality image processing to preserve detail
- Visual debug output showing detection steps

Author: Claude
"""

import os
import sys
import logging
import threading
import queue
import traceback
import importlib.util
import subprocess
import platform
import webbrowser
import re
import shutil
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Then handle potentially problematic imports
try:
    from PIL import Image, ImageTk
except ImportError:
    logger.warning("PIL/Pillow library not found, some image operations may fail")

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog

try:
    import pytesseract
except ImportError:
    # Import failed — call fallback directly if class is defined below
    class _TesseractImportHelper:
        @staticmethod
        def get_installation_instructions():
            from platform import system
            instructions = {
                'Windows': 'https://github.com/UB-Mannheim/tesseract/wiki'
            }
            base = (
                "To use OCR features, you need to:\n\n"
                "1. Install Tesseract OCR for your platform\n"
                "2. Install the pytesseract Python package: pip install pytesseract\n\n"
            )
            sys_type = system()
            specific = f"For {sys_type} systems: {instructions.get(sys_type, 'See https://github.com/tesseract-ocr/tesseract')}"
            return base + specific

    print(_TesseractImportHelper.get_installation_instructions())

# function to maintain a consistent naming convention for debug images
def generate_debug_filename(
    base_filename: str, 
    step_name: str, 
    hole_id: Optional[str] = None, 
    depth_from: Optional[float] = None, 
    depth_to: Optional[float] = None
) -> str:
    """
    Generate a consistent debug filename.
    
    Args:
        base_filename: Original image filename
        step_name: Processing step name
        hole_id: Optional hole identifier
        depth_from: Optional depth range start
        depth_to: Optional depth range end
    
    Returns:
        Formatted debug filename
    """
    # Remove file extension from base filename
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    
    # If hole_id and depths are provided, use them
    if hole_id and depth_from is not None and depth_to is not None:
        filename = f"{hole_id}_depth_{depth_from:.1f}-{depth_to:.1f}_{base_name}_{step_name}.jpg"
    else:
        # Fallback to original naming strategy
        filename = f"{base_name}_{step_name}.jpg"
    
    return filename

# Function to correctly rename all the Debug images with the information from the OCR method
def rename_debug_files(
        original_filename: str, 
        hole_id: Optional[str], 
        depth_from: Optional[float], 
        depth_to: Optional[float]
    ) -> None:
        """
        Rename debug files for a specific image after successful metadata extraction.
        
        Args:
            original_filename: Original input image filename
            hole_id: Extracted hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
        """
        if not (hole_id and depth_from is not None and depth_to is not None):
            return
        
        # Get path to debug directory
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(original_filename)), 'debug')
        
        # Create the debug directory if it doesn't exist
        os.makedirs(debug_dir, exist_ok=True)
        
        # Find all debug files for this image
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        debug_files = [
            f for f in os.listdir(debug_dir) 
            if f.startswith(base_name) and f.endswith('.jpg')
        ]
        
        for old_filename in debug_files:
            # Extract the step name from the old filename
            step_parts = old_filename.split('_')
            if len(step_parts) >= 2:
                step_name = step_parts[-1].replace('.jpg', '')
                
                # Generate new filename with metadata
                new_filename = generate_debug_filename(
                    original_filename, 
                    step_name,
                    hole_id, 
                    depth_from, 
                    depth_to
                )
                
                old_path = os.path.join(debug_dir, old_filename)
                new_path = os.path.join(debug_dir, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    logger.info(f"Renamed debug file: {old_filename} -> {new_filename}")
                except Exception as e:
                    logger.error(f"Error renaming debug file {old_filename}: {e}")



class TesseractManager:
    """
    Manages Tesseract OCR detection, installation guidance, and OCR functionality.
    Designed to gracefully handle cases where Tesseract is not installed.
    """
    
    def __init__(self):
        """Initialize the Tesseract manager with default paths and settings."""
        self.is_available = False
        self.version = None
        self.pytesseract = None
        self.install_instructions = {
            'Windows': 'https://github.com/UB-Mannheim/tesseract/wiki',
            'Darwin': 'https://brew.sh/ (then run: brew install tesseract)',
            'Linux': 'sudo apt-get install tesseract-ocr libtesseract-dev'
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
            logger.info(f"Tesseract OCR not available. {instructions}")
            return
        
        # Create a custom dialog
        dialog = tk.Toplevel(parent)
        dialog.title("Tesseract OCR Required")
        dialog.geometry("500x400")
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
            text="Tesseract OCR Required for Text Recognition",
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
                text=f"Download for {system}", 
                command=lambda: webbrowser.open(install_url)
            )
            install_button.pack(side=tk.LEFT, padx=5)
        
        close_button = ttk.Button(
            button_frame, 
            text="Continue without OCR", 
            command=dialog.destroy
        )
        close_button.pack(side=tk.RIGHT, padx=5)
    
    def _calculate_avg_confidence(self, data: Dict[str, List[Any]]) -> float:
        """
        Calculate the average confidence of OCR results.
        
        Args:
            data: OCR data dictionary from pytesseract.image_to_data
        
        Returns:
            Average confidence percentage, or 0.0 if no confidence data available
        """
        try:
            # Extract confidence values, filtering out invalid entries
            confidences = [
                float(conf) for conf in data.get('conf', []) 
                if conf not in [-1, '-1'] and str(conf).replace('.', '').isdigit()
            ]
            
            # Calculate average confidence
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                return max(0.0, min(avg_confidence, 100.0))  # Clamp between 0 and 100
            
            logger.warning("No valid confidence values found in OCR data")
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating OCR confidence: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    # Function to extract patterns from OCR
    def extract_metadata_patterns(self, ocr_text: str) -> Dict[str, Any]:
        """
        Extract metadata from OCR text using pattern matching.
        
        Args:
            ocr_text: OCR-extracted text
            
        Returns:
            Dictionary containing extracted metadata
        """
        result = {
            'hole_id': None,
            'depth_from': None,
            'depth_to': None
        }
        
        # Clean and normalize text
        ocr_text = re.sub(r'\s+', ' ', ocr_text).strip()
        
        # Log the cleaned OCR text for debugging
        logger.info(f"Cleaned OCR text: '{ocr_text}'")
        
        # Split the text by lines - handle different text on different lines
        lines = ocr_text.split('\n')
        
        # Process each line separately
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Try to extract hole ID - should be on its own line
            km_match = re.search(r'(KM\s*\d{4}|K\s*M\s*\d{4})', line, re.IGNORECASE)
            if km_match:
                # Extract and clean up hole ID
                hole_id = re.sub(r'\s+', '', km_match.group(0)).upper()
                result['hole_id'] = hole_id
                logger.info(f"Found hole ID: {hole_id}")
                continue
                
            # Try to extract depth range - typically has a dash or hyphen
            depth_match = re.search(r'(\d+)[\s\-–—to]+(\d+)', line)
            if depth_match:
                try:
                    result['depth_from'] = float(depth_match.group(1))
                    result['depth_to'] = float(depth_match.group(2))
                    logger.info(f"Found depth range: {result['depth_from']}-{result['depth_to']}")
                except ValueError:
                    logger.warning(f"Found depth-like pattern but couldn't convert to float: {depth_match.group(0)}")
        
        # If we still don't have results, try more aggressive patterns
        if result['hole_id'] is None:
            # Try to find any pattern that looks like KMXXXX anywhere in the text
            all_text = ' '.join(lines)
            km_match = re.search(r'[KMLPRkmlpr]{2}\s*\d{4}', all_text)
            if km_match:
                result['hole_id'] = re.sub(r'\s+', '', km_match.group(0)).upper()
                logger.info(f"Found hole ID (fallback method): {result['hole_id']}")
        
        # If we still don't have depth, try to find any two numbers with separators
        if result['depth_from'] is None or result['depth_to'] is None:
            all_text = ' '.join(lines)
            depth_match = re.search(r'(\d+)[^\d]+(\d+)', all_text)
            if depth_match:
                try:
                    result['depth_from'] = float(depth_match.group(1))
                    result['depth_to'] = float(depth_match.group(2))
                    logger.info(f"Found depth range (fallback method): {result['depth_from']}-{result['depth_to']}")
                except ValueError:
                    pass
        
        return result

    def _extract_depths_from_text(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract depth range values from OCR text.
        
        Args:
            text: Text from OCR
                
        Returns:
            Tuple of (depth_from, depth_to)
        """
        depth_from = None
        depth_to = None
        
        # Log the input text for debugging
        logger.info(f"Extracting depths from text: '{text}'")
        
        # Remove all non-digit, non-dash, non-space characters
        cleaned_text = re.sub(r'[^0-9\s\-–—]', '', text)
        logger.info(f"Cleaned text for depth extraction: '{cleaned_text}'")
        
        # Try pattern matching for "060 - 080" format (three digits, separator, three digits)
        match = re.search(r'(\d{3})\s*[\-–—]\s*(\d{3})', cleaned_text)
        if match:
            try:
                depth_from = float(match.group(1))
                depth_to = float(match.group(2))
                logger.info(f"Three-digit format matched: {depth_from}-{depth_to}")
                return depth_from, depth_to
            except (ValueError, IndexError):
                logger.warning(f"Format match failed conversion: {match.groups()}")
        
        # Try general pattern for any digits with separator
        match = re.search(r'(\d+)\s*[\-–—]\s*(\d+)', cleaned_text)
        if match:
            try:
                depth_from = float(match.group(1))
                depth_to = float(match.group(2))
                logger.info(f"General format matched: {depth_from}-{depth_to}")
                return depth_from, depth_to
            except (ValueError, IndexError):
                pass
        
        # If we just have digits without clear separator, try to extract them
        digits = re.findall(r'\d+', cleaned_text)
        if len(digits) >= 2:
            try:
                depth_from = float(digits[0])
                depth_to = float(digits[1])
                logger.info(f"Found separate digits: {depth_from}-{depth_to}")
                return depth_from, depth_to
            except (ValueError, IndexError):
                pass
        
        return depth_from, depth_to
    
    def _preprocess_for_ocr(self, input_image: np.ndarray, original_filename: Optional[str] = None) -> np.ndarray:
        """
        Specialized image preprocessing for chip tray labels.
        Optimized for black text on white background with borders.
        
        Args:
            input_image: Input image to preprocess
            original_filename: Path to original image file for debug image saving
        
        Returns:
            np.ndarray: Preprocessed image optimized for OCR
        """
        try:
            # Check if image is valid
            if input_image is None or input_image.size == 0:
                logger.error("Invalid image provided for OCR preprocessing")
                raise ValueError("Invalid input image")
                
            # Convert to grayscale if needed
            if len(input_image.shape) == 3:
                gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_image.copy()
            
            # Store intermediate results for debugging
            debug_images = {"gray": gray}
            
            # Detect and remove black borders
            # Create binary image to identify potential borders
            _, border_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            debug_images["border_thresh"] = border_thresh

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
                image_perimeter = 2 * (gray.shape[0] + gray.shape[1])
                contour_perimeter = cv2.arcLength(largest_contour, True)
                perimeter_ratio = contour_perimeter / image_perimeter
                
                if perimeter_ratio > 0.7:
                    logger.info(f"Detected potential border (perimeter ratio: {perimeter_ratio:.2f})")
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Ensure bounding rect covers most of the image (indicating a border)
                    image_area = gray.shape[0] * gray.shape[1]
                    rect_area = w * h
                    area_ratio = rect_area / image_area
                    
                    if area_ratio > 0.8:
                        logger.info(f"Confirmed border detection (area ratio: {area_ratio:.2f})")
                        
                        # Create a mask for everything inside the contour
                        mask = np.zeros_like(gray)
                        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
                        debug_images["border_mask"] = mask
                        
                        # Create a smaller mask to exclude the border itself
                        # Use adaptive kernel size based on image dimensions
                        border_width = min(max(3, min(w, h) // 30), 10)  # Between 3 and 10 pixels
                        kernel = np.ones((border_width, border_width), np.uint8)
                        eroded_mask = cv2.erode(mask, kernel, iterations=1)
                        debug_images["eroded_mask"] = eroded_mask
                        
                        # Extract the label content without the border
                        inner_content = np.ones_like(gray) * 255  # White background
                        inner_content[eroded_mask > 0] = gray[eroded_mask > 0]
                        
                        # Replace the original grayscale image with the border-removed version
                        gray = inner_content
                        debug_images["border_removed"] = gray
                        logger.info("Border removal completed")
            
            # 1. Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            debug_images["filtered"] = filtered
            
            # 2. Apply adaptive thresholding with a larger block size for label text
            binary = cv2.adaptiveThreshold(
                filtered, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 
                15, 2  # Larger block size for more stable thresholding
            )
            debug_images["binary"] = binary
            
            # 3. Invert if needed - assuming dark text on light background
            # Count white vs black pixels to determine if inversion is needed
            white_count = np.sum(binary == 255)
            black_count = np.sum(binary == 0)
            
            # If there are more black pixels than white, invert the image
            if black_count > white_count:
                binary = cv2.bitwise_not(binary)
                debug_images["binary_inverted"] = binary
            
            # 4. Morphological operations to clean up noise
            # Create a small kernel for morphological operations
            kernel = np.ones((2, 2), np.uint8)
            
            # Opening to remove small noise
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            debug_images["opened"] = opened
            
            # Closing to fill small holes in characters
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            debug_images["final"] = closed
            
            # Save debug images if enabled
            if hasattr(self, 'config') and self.config.get('save_debug_images', False) and original_filename:
                debug_dir = os.path.join(os.path.dirname(os.path.abspath(original_filename)), 
                                        self.config.get('debug_folder', 'debug'))
                os.makedirs(debug_dir, exist_ok=True)
                
                for name, img in debug_images.items():
                    debug_path = os.path.join(debug_dir, 
                                            generate_debug_filename(original_filename, f'ocr_{name}'))
                    cv2.imwrite(debug_path, img)
            
            return closed
            
        except Exception as e:
            logger.error(f"Error during OCR preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            # Return original grayscale as fallback
            if 'gray' in locals():
                return gray
            elif input_image is not None:
                if len(input_image.shape) == 3:
                    return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                return input_image
            return np.zeros((10, 10), dtype=np.uint8)  # Return empty image as last resort

    def extract_labels_with_templates(
        self, 
        metadata_region: np.ndarray, 
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract label text using template matching and targeted OCR.
        
        Args:
            metadata_region: Image region containing the metadata labels
            original_filename: Original image filename for debug purposes
            
        Returns:
            Dictionary with extracted metadata
        """
        try:
            # Convert to grayscale if needed
            if len(metadata_region.shape) == 3:
                gray = cv2.cvtColor(metadata_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = metadata_region.copy()
                
            # Create debug images dictionary
            debug_images = {"original": metadata_region.copy(), "gray": gray}
            
            # Apply adaptive thresholding to find rectangular labels
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
            debug_images["threshold"] = thresh
            
            # Find contours - looking specifically for rectangular label regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by shape, size, and aspect ratio to find label rectangles
            label_regions = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio and area
                aspect_ratio = w / h
                area = w * h
                
                # Filter for typical label rectangles (wider than tall, reasonable size)
                if 2.0 < aspect_ratio < 7.0 and area > 500:
                    # Check if it's rectangular enough (compare contour area to bounding rect area)
                    rect_area = w * h
                    contour_area = cv2.contourArea(contour)
                    fill_ratio = contour_area / rect_area
                    
                    # Labels are typically rectangular with high fill ratio
                    if fill_ratio > 0.75:
                        label_regions.append((x, y, w, h))
            
            # Sort regions by Y-coordinate (top to bottom) to get the correct label order
            label_regions.sort(key=lambda r: r[1])
            
            # Visualization
            viz_image = metadata_region.copy()
            for i, (x, y, w, h) in enumerate(label_regions):
                cv2.rectangle(viz_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(viz_image, f"Label {i+1}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            debug_images["labels_detected"] = viz_image
            
            # Initialize metadata
            hole_id = None
            depth_from = None
            depth_to = None
            
            # Process each detected label separately
            for i, (x, y, w, h) in enumerate(label_regions[:2]):  # Process at most 2 labels
                # Extract the label region
                label_img = gray[y:y+h, x:x+w]
                
                # Skip very small regions
                if label_img.shape[0] < 10 or label_img.shape[1] < 10:
                    continue
                    
                # Enlarge for better OCR results
                resized_label = cv2.resize(label_img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
                
                # Apply both regular and inverted thresholding for better OCR reliability
                _, label_bin = cv2.threshold(resized_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                label_bin_inv = cv2.bitwise_not(label_bin)  # Inverted version
                
                # Save debug images
                debug_images[f"label_{i+1}"] = resized_label
                debug_images[f"label_{i+1}_bin"] = label_bin
                debug_images[f"label_{i+1}_bin_inv"] = label_bin_inv
                
                # Process first label (usually hole ID)
                if i == 0 and hole_id is None:
                    for binary_img in [label_bin, label_bin_inv]:
                        # Try PSM 7 (single line) and PSM 8 (single word) modes
                        for psm_mode in [7, 8]:
                            try:
                                config = f"--psm {psm_mode} --oem 3"
                                if psm_mode == 8:
                                    config += " -c tessedit_char_whitelist=KM0123456789"
                                    
                                text = self.pytesseract.image_to_string(binary_img, config=config).strip()
                                logger.info(f"Label {i+1} OCR text (PSM {psm_mode}): '{text}'")
                                
                                # Check for KM pattern
                                km_pattern = re.search(r'[KM]{2}\d{4}', text.upper().replace(" ", ""))
                                if km_pattern:
                                    hole_id = km_pattern.group(0)
                                    logger.info(f"Found hole ID: {hole_id}")
                                    break
                            except Exception as e:
                                logger.warning(f"OCR error for hole ID: {str(e)}")
                        
                        if hole_id:
                            break
                
                # Process second label (usually depth range)
                elif i == 1 and (depth_from is None or depth_to is None):
                    for binary_img in [label_bin, label_bin_inv]:
                        # Try different PSM modes
                        for psm_mode in [7, 6, 3]:
                            try:
                                config = f"--psm {psm_mode} --oem 3"
                                if psm_mode == 7:
                                    config += " -c tessedit_char_whitelist=0123456789- "
                                    
                                text = self.pytesseract.image_to_string(binary_img, config=config).strip()
                                logger.info(f"Label {i+1} OCR text (PSM {psm_mode}): '{text}'")
                                
                                # Extract depth range
                                depths = self._extract_depths_from_text(text)
                                if depths[0] is not None and depths[1] is not None:
                                    depth_from, depth_to = depths
                                    logger.info(f"Found depth range: {depth_from}-{depth_to}")
                                    break
                            except Exception as e:
                                logger.warning(f"OCR error for depth: {str(e)}")
                        
                        if depth_from is not None and depth_to is not None:
                            break
            
            # Save debug images
            if hasattr(self, 'config') and self.config.get('save_debug_images', False) and original_filename:
                debug_dir = os.path.join(os.path.dirname(os.path.abspath(original_filename)), 
                                    self.config.get('debug_folder', 'debug'))
                os.makedirs(debug_dir, exist_ok=True)
                
                for name, img in debug_images.items():
                    debug_path = os.path.join(debug_dir, 
                                            generate_debug_filename(original_filename, f'template_{name}'))
                    cv2.imwrite(debug_path, img)
            
            # Calculate confidence
            confidence = 0.0
            
            # Add confidence for hole ID
            if hole_id:
                if re.match(r'^[KM]{2}\d{4}$', hole_id):  # Exact match for standard format
                    confidence += 45.0
                else:
                    confidence += 25.0
            
            # Add confidence for depth range
            if depth_from is not None and depth_to is not None:
                if depth_from < depth_to:  # Valid range
                    confidence += 40.0
                else:
                    confidence += 20.0
            
            logger.info(f"Template extraction results: ID={hole_id}, Depth={depth_from}-{depth_to}, Confidence={confidence:.1f}%")
            
            return {
                'hole_id': hole_id,
                'depth_from': depth_from,
                'depth_to': depth_to,
                'confidence': confidence,
                'label_regions': label_regions,
                'debug_images': debug_images
            }
            
        except Exception as e:
            logger.error(f"Error in template-based label extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'hole_id': None,
                'depth_from': None,
                'depth_to': None,
                'confidence': 0.0
            }

    def _detect_numbers_in_label(self, label_image: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect numbers in a depth label using digit detection techniques.
        Useful when OCR fails to recognize the dash.
        
        Args:
            label_image: Binary image of the label
            
        Returns:
            Tuple of (depth_from, depth_to)
        """
        try:
            # Find all contours - these might be individual digits
            contours, _ = cv2.findContours(label_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by x-coordinate
            digit_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                # Filter by aspect ratio and size to find digits
                if 0.3 < aspect_ratio < 1.2 and h > 20:
                    digit_contours.append((x, y, w, h, contour))
            
            # Sort by x-coordinate
            digit_contours.sort(key=lambda c: c[0])
            
            # Skip if too few contours
            if len(digit_contours) < 2:
                return None, None
            
            # Extract and recognize each digit individually
            extracted_digits = []
            
            for x, y, w, h, contour in digit_contours:
                # Extract the digit
                digit_img = label_image[y:y+h, x:x+w]
                
                # Resize for better recognition
                digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_CUBIC)
                
                # Apply OCR with digit-specific settings
                try:
                    digit = self.pytesseract.image_to_string(
                        digit_img, 
                        config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
                    ).strip()
                    
                    if digit and digit.isdigit():
                        extracted_digits.append((x, digit))
                except Exception:
                    continue
            
            # Sort digits by position
            extracted_digits.sort(key=lambda d: d[0])
            
            # Process the recognized digits
            if len(extracted_digits) >= 2:
                # Special case for "0-20": typically 2 or 3 digits
                if len(extracted_digits) == 2:
                    # Assume "0" and "20"
                    return 0.0, float(extracted_digits[1][1])
                elif len(extracted_digits) == 3:
                    # Typically "0", "-", "20" (but - might be recognized as a digit)
                    return float(extracted_digits[0][1]), float(extracted_digits[2][1])
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error in digit detection: {str(e)}")
            return None, None

    def extract_metadata_from_image(
            self, 
            image: np.ndarray, 
            markers: Dict[int, np.ndarray], 
            original_filename: Optional[str] = None,
            progress_queue: Optional[queue.Queue] = None
        ) -> Dict[str, Any]:
        """
        Extract metadata using ArUco marker positions for precise ROI selection.
        
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
            
            # Define marker groups
            corner_marker_ids = [0, 1, 2, 3]
            compartment_marker_ids = [id for id in range(4, 24) if id in markers]
            
            if not compartment_marker_ids or not any(id in markers for id in corner_marker_ids):
                logger.warning("Insufficient markers found for metadata region extraction")
                return {
                    'hole_id': None,
                    'depth_from': None,
                    'depth_to': None,
                    'confidence': 0.0,
                    'metadata_region': None
                }
            
            # 1. Find the BOTTOM of corner markers
            detected_corner_ids = [id for id in corner_marker_ids if id in markers]
            corner_bottoms = []
            for id in detected_corner_ids:
                marker_corners = markers[id]
                # Get bottom-most point (max y-value)
                bottom_y = np.max(marker_corners[:, 1])
                corner_bottoms.append(bottom_y)
            
            # Get the lowest (maximum y-value) corner marker
            corner_bottom = max(corner_bottoms) if corner_bottoms else 0
            
            # 2. Find the BOTTOM of compartment markers 
            # Use the bottom-most point of all compartment markers
            compartment_bottoms = []
            for marker_id in compartment_marker_ids:
                marker_corners = markers[marker_id]
                # Get bottom-most point (max y-value)
                bottom_y = np.max(marker_corners[:, 1])
                compartment_bottoms.append(bottom_y)
            
            # Use the bottom of the lowest compartment marker
            compartment_bottom = min(compartment_bottoms) if compartment_bottoms else h
            
            # 3. Find the LEFTMOST and RIGHTMOST compartment markers
            compartment_lefts = []
            compartment_rights = []
            for marker_id in compartment_marker_ids:
                marker_corners = markers[marker_id]
                # Get leftmost point (min x-value)
                left_x = np.min(marker_corners[:, 0])
                # Get rightmost point (max x-value)
                right_x = np.max(marker_corners[:, 0])
                
                compartment_lefts.append(left_x)
                compartment_rights.append(right_x)
            
            leftmost_compartment_x = min(compartment_lefts) if compartment_lefts else 0
            rightmost_compartment_x = max(compartment_rights) if compartment_rights else w
            
            # Define metadata region boundaries
            metadata_top = corner_bottom  # Below the lowest corner marker
            metadata_bottom = compartment_bottom  # At the bottom of compartment markers
            metadata_left = 0  # From the left edge
            metadata_right = leftmost_compartment_x  # To the leftmost compartment marker
            
            # Use more generous margins for better region capture
            margin_top = int(h * 0.05)  # 5% of image height instead of 2%
            margin_side = int(w * 0.03)  # 3% of image width instead of 1%
            
            metadata_top = max(0, metadata_top - margin_top)
            metadata_bottom = min(h, metadata_bottom + margin_top)
            metadata_left = max(0, metadata_left - margin_side)
            metadata_right = min(w, metadata_right + margin_side)
            
            # Validate the region (ensure it's not too small or inverted)
            if metadata_bottom <= metadata_top or metadata_right <= metadata_left:
                logger.warning("Invalid metadata region dimensions calculated")
                return {
                    'hole_id': None,
                    'depth_from': None,
                    'depth_to': None,
                    'confidence': 0.0,
                    'metadata_region': None
                }
            
            # Extract the metadata region
            metadata_region = image[
                int(metadata_top):int(metadata_bottom), 
                int(metadata_left):int(metadata_right)
            ].copy()

            # Create visualization image
            viz_image = image.copy()
            cv2.rectangle(viz_image, 
                        (int(metadata_left), int(metadata_top)), 
                        (int(metadata_right), int(metadata_bottom)), 
                        (0, 255, 0), 2)

            # Save both images for debug if enabled
            if original_filename and hasattr(self, 'config') and self.config.get('save_debug_images', False):
                debug_dir = os.path.join(os.path.dirname(os.path.abspath(original_filename)), 
                                        self.config.get('debug_folder', 'debug'))
                os.makedirs(debug_dir, exist_ok=True)
                
                debug_filename = generate_debug_filename(
                    original_filename, 
                    'metadata_region'
                )
                debug_path = os.path.join(debug_dir, debug_filename)
                cv2.imwrite(debug_path, metadata_region)
                
                # Also save a visualization showing the region on the original image
                viz_filename = generate_debug_filename(original_filename, 'metadata_region_viz')
                viz_path = os.path.join(debug_dir, viz_filename)
                cv2.imwrite(viz_path, viz_image)
            
            # Apply enhanced preprocessing with shadow and blur handling
            preprocessed = self._preprocess_for_ocr(metadata_region, original_filename)
            
        # Store the preprocessed image directly (for debug and UI)
            extracted_metadata = {
                'hole_id': None,
                'depth_from': None,
                'depth_to': None,
                'confidence': 0.0,
                'metadata_region': metadata_region,
                'metadata_region_viz': viz_image,
                'preprocessed_ocr': preprocessed,  # Store for debugging
                'ocr_text': ""
            }
            
            # Test OCR configuration
            ocr_config_attempts = [
                ('Default', '--psm 6 --oem 3'),          # Single uniform block of text
                ('Single Word', '--psm 8 --oem 3'),      # Treat as a single word - good for hole IDs
                ('Single Line', '--psm 7 --oem 3'),      # Treat as a single text line - good for depths
                ('Single Char', '--psm 10 --oem 3'),     # Treat as a single character
                ('Sparse Text', '--psm 11 --oem 3 -c tessedit_char_whitelist=KM0123456789-')  # Sparse text with char whitelist
            ]
            
            best_confidence = 0.0
            best_result = ""
            
            for attempt_name, config_options in ocr_config_attempts:
                if progress_queue is not None:
                    progress_queue.put((f"OCR Attempt: {attempt_name}", None))
                
                try:
                    # Check if Tesseract is available
                    if not self.is_available or self.pytesseract is None:
                        logger.warning("Tesseract not available for OCR")
                        break
                    
                    # Debug: Save the exact image going to Tesseract
                    if original_filename and hasattr(self, 'config') and self.config.get('save_debug_images', False):
                        debug_dir = os.path.join(os.path.dirname(os.path.abspath(original_filename)), 
                                            self.config.get('debug_folder', 'debug'))
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        debug_path = os.path.join(debug_dir, 
                                            generate_debug_filename(original_filename, f'ocr_to_tesseract_{attempt_name}'))
                        cv2.imwrite(debug_path, preprocessed)
                    
                    # Perform OCR with current configuration
                    result = self.pytesseract.image_to_string(
                        preprocessed, 
                        config=config_options
                    ).strip()
                    
                    # Get confidence data
                    data = self.pytesseract.image_to_data(
                        preprocessed,
                        output_type=self.pytesseract.Output.DICT,
                        config=config_options
                    )
                    
                    # Calculate confidence
                    confidence = self._calculate_avg_confidence(data)
                    
                    # Log OCR attempt details
                    logger.info(f"OCR {attempt_name} attempt: Confidence={confidence:.1f}%, Text='{result}'")
                    
                    # Move this function inside the class
                    extracted = self.extract_metadata_patterns(result)
                    
                    # Update if better confidence or found metadata
                    if (confidence > best_confidence or
                        (extracted_metadata['hole_id'] is None and extracted['hole_id'] is not None) or
                        (extracted_metadata['depth_from'] is None and extracted['depth_from'] is not None)):
                        best_result = result
                        best_confidence = confidence
                        
                        # Update metadata with extracted values
                        extracted_metadata['hole_id'] = extracted['hole_id']
                        extracted_metadata['depth_from'] = extracted['depth_from']
                        extracted_metadata['depth_to'] = extracted['depth_to']
                        extracted_metadata['confidence'] = confidence
                        extracted_metadata['ocr_text'] = result
                    
                    # Break early if we have good results
                    if (confidence > self.config.get('ocr_confidence_threshold', 70.0) and 
                        extracted['hole_id'] is not None and 
                        extracted['depth_from'] is not None and
                        extracted['depth_to'] is not None):
                        break
                        
                except Exception as e:
                    logger.warning(f"OCR processing error with config {config_options}: {e}")
                    logger.error(traceback.format_exc())
            
            if (extracted_metadata['hole_id'] is None or 
                extracted_metadata['depth_from'] is None or 
                extracted_metadata['depth_to'] is None):
                
                logger.info("------------------------------------------------")
                logger.info("Standard OCR detection results:")
                logger.info(f"Hole ID: {extracted_metadata['hole_id']}")
                logger.info(f"Depth range: {extracted_metadata['depth_from']} - {extracted_metadata['depth_to']}")
                logger.info(f"Confidence: {extracted_metadata['confidence']:.1f}%")
                logger.info(f"OCR text: '{extracted_metadata['ocr_text']}'")
                logger.info("------------------------------------------------")
                logger.info("Trying template-based approach as fallback...")
                
                if progress_queue is not None:
                    progress_queue.put(("Trying template-based OCR fallback...", None))
                
                # Call the template-based extraction as fallback
                try:
                    template_results = self.extract_labels_with_templates(metadata_region, original_filename)
                    
                    # Log the template results
                    logger.info(f"Template OCR results: {template_results}")
                    
                    # Only use template results if they're better than what we already have
                    if template_results.get('confidence', 0) > extracted_metadata.get('confidence', 0):
                        # Update the metadata with template results
                        if template_results.get('hole_id'):
                            extracted_metadata['hole_id'] = template_results['hole_id']
                        
                        if template_results.get('depth_from') is not None:
                            extracted_metadata['depth_from'] = template_results['depth_from']
                        
                        if template_results.get('depth_to') is not None:
                            extracted_metadata['depth_to'] = template_results['depth_to']
                        
                        if template_results.get('confidence'):
                            extracted_metadata['confidence'] = template_results['confidence']
                        
                        if progress_queue is not None:
                            progress_queue.put((f"Template OCR found: {extracted_metadata['hole_id']} "
                                            f"({extracted_metadata['depth_from']}-{extracted_metadata['depth_to']})", None))
                
                except Exception as e:
                    logger.error(f"Error in template OCR fallback: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue with whatever metadata we had from standard OCR
            
            logger.info("------------------------------------------------")
            logger.info("Final metadata extraction results:")
            logger.info(f"Hole ID: {extracted_metadata['hole_id']}")
            logger.info(f"Depth range: {extracted_metadata['depth_from']} - {extracted_metadata['depth_to']}")
            logger.info(f"Confidence: {extracted_metadata['confidence']:.1f}%")
            logger.info("------------------------------------------------")
            

            # Return final result
            return extracted_metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'hole_id': None,
                'depth_from': None,
                'depth_to': None,
                'confidence': 0.0,
                'metadata_region': None
            }


class MetadataInputDialog:
    """
    Dialog for collecting metadata when OCR fails or needs confirmation.
    Shows the metadata region of the image and provides fields for entering hole ID and depth range.
    """
    
    def __init__(self, parent: Optional[tk.Tk], image: Optional[np.ndarray] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata input dialog.
        
        Args:
            parent: Parent Tkinter window
            image: Optional image to display (metadata region)
            metadata: Optional pre-filled metadata from OCR
        """
        self.parent = parent
        self.image = image
        self.metadata = metadata or {}
        
        # Result values
        self.hole_id = tk.StringVar(value=self.metadata.get('hole_id', ''))
        self.depth_from = tk.StringVar(value=str(self.metadata.get('depth_from', '')))
        self.depth_to = tk.StringVar(value=str(self.metadata.get('depth_to', '')))
        self.result = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Enter Chip Tray Metadata")
        self.dialog.geometry("800x700")  # Increased width and height
        self.dialog.grab_set()  # Make dialog modal
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """Create all widgets for the dialog."""
        # Main frame with padding
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_label = ttk.Label(
            main_frame, 
            text="Enter Chip Tray Metadata",
            font=("Arial", 12, "bold")
        )
        header_label.pack(pady=(0, 10))
        
        # Image display (if provided)
        if self.image is not None:
            self._add_image_display(main_frame)
        
        # OCR result display (if available)
        if self.metadata.get('ocr_text'):
            self._add_ocr_result_display(main_frame)
        
        # Input fields
        self._add_input_fields(main_frame)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ok_button = ttk.Button(
            button_frame, 
            text="OK", 
            command=self._on_ok
        )
        ok_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        cancel_button = ttk.Button(
            button_frame, 
            text="Cancel", 
            command=self._on_cancel
        )
        cancel_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
    
    
    
    def _add_image_display(self, parent: ttk.Frame) -> None:
        """
        Add image display widget to the dialog with specialized layout for chip tray labels.
        
        Args:
            parent: Parent frame to add the widget to
        """
        # Create a frame to hold all images
        image_frame = ttk.LabelFrame(parent, text="Image Analysis", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        try:
            # Create a canvas for scrollable content
            canvas = tk.Canvas(image_frame, borderwidth=0)
            scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Display order and captions
            display_items = [
                ('metadata_region_viz', 'Full Image with Detected Region'),
                ('metadata_region', 'Extracted Metadata Region'),
                ('preprocessed_ocr', 'OCR Processing Result')
            ]
            
            for item_key, item_title in display_items:
                if item_key in self.metadata and self.metadata[item_key] is not None:
                    # Create container for this image
                    item_frame = ttk.Frame(scrollable_frame)
                    item_frame.pack(pady=10, fill=tk.X)
                    
                    # Add title above image
                    title_label = ttk.Label(
                        item_frame, 
                        text=item_title,
                        font=("Arial", 10, "bold")
                    )
                    title_label.pack(pady=(0, 5))
                    
                    # Prepare image
                    img = self.metadata[item_key]
                    h, w = img.shape[:2]
                    
                    # Calculate display dimensions - preserve aspect ratio
                    max_width = 750  # Maximum width for display
                    max_height = 400  # Maximum height for display
                    
                    # Calculate scale factor to fit within limits
                    width_scale = max_width / w if w > max_width else 1
                    height_scale = max_height / h if h > max_height else 1
                    scale = min(width_scale, height_scale)
                    
                    # Apply scaling
                    if scale < 1:
                        new_width = int(w * scale)
                        new_height = int(h * scale)
                        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    else:
                        resized = img.copy()
                    
                    # Convert to RGB if needed
                    if len(resized.shape) == 3:
                        if resized.shape[2] == 3:
                            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        else:
                            img_rgb = resized
                    else:
                        img_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                    
                    # Convert to Tkinter compatible image
                    from PIL import Image, ImageTk
                    pil_img = Image.fromarray(img_rgb)
                    tk_img = ImageTk.PhotoImage(image=pil_img)
                    
                    # Create image display
                    img_label = ttk.Label(item_frame, image=tk_img)
                    img_label.image = tk_img  # Keep reference
                    img_label.pack()
                    
                    # Add dimensions info
                    size_label = ttk.Label(
                        item_frame,
                        text=f"Original size: {w}x{h} pixels",
                        font=("Arial", 8),
                        foreground="gray"
                    )
                    size_label.pack(pady=(5, 0))
            
            # Add help text
            help_text = ttk.Label(
                scrollable_frame, 
                text="Examine these images to verify hole ID and depth range.",
                font=("Arial", 9)
            )
            help_text.pack(pady=10)
            
        except Exception as e:
            # If image display fails, show error message
            logger.error(f"Error displaying metadata images: {str(e)}")
            logger.error(traceback.format_exc())
            error_label = ttk.Label(
                image_frame, 
                text=f"Error displaying images: {str(e)}",
                foreground="red"
            )
            error_label.pack(pady=10)
    
    def _add_ocr_result_display(self, parent: ttk.Frame) -> None:
        """
        Add OCR result display widget to the dialog.
        
        Args:
            parent: Parent frame to add the widget to
        """
        ocr_frame = ttk.LabelFrame(parent, text="OCR Result", padding="10")
        ocr_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Display OCR text
        ocr_text = tk.Text(ocr_frame, height=3, wrap=tk.WORD)
        ocr_text.pack(fill=tk.X, expand=True)
        ocr_text.insert(tk.END, self.metadata.get('ocr_text', ''))
        ocr_text.config(state=tk.DISABLED)
        
        # Add confidence info
        confidence = self.metadata.get('confidence', 0)
        confidence_label = ttk.Label(
            ocr_frame, 
            text=f"OCR Confidence: {confidence:.1f}%",
            font=("Arial", 9)
        )
        confidence_label.pack(anchor='e', pady=(5, 0))
    
    def _add_input_fields(self, parent: ttk.Frame) -> None:
        """
        Add input fields for metadata.
        
        Args:
            parent: Parent frame to add the widgets to
        """
        fields_frame = ttk.LabelFrame(parent, text="Enter Metadata", padding="10")
        fields_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Hole ID field
        hole_id_frame = ttk.Frame(fields_frame)
        hole_id_frame.pack(fill=tk.X, pady=5)
        
        hole_id_label = ttk.Label(
            hole_id_frame, 
            text="Hole ID:",
            width=10
        )
        hole_id_label.pack(side=tk.LEFT)
        
        hole_id_entry = ttk.Entry(
            hole_id_frame,
            textvariable=self.hole_id
        )
        hole_id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Format help
        hole_id_help = ttk.Label(
            hole_id_frame,
            text="Format: XX0000",
            font=("Arial", 8),
            foreground="gray"
        )
        hole_id_help.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Depth range fields
        depth_frame = ttk.Frame(fields_frame)
        depth_frame.pack(fill=tk.X, pady=5)
        
        depth_label = ttk.Label(
            depth_frame, 
            text="Depth:",
            width=10
        )
        depth_label.pack(side=tk.LEFT)
        
        depth_from_entry = ttk.Entry(
            depth_frame,
            textvariable=self.depth_from,
            width=8
        )
        depth_from_entry.pack(side=tk.LEFT)
        
        depth_separator = ttk.Label(
            depth_frame, 
            text="-"
        )
        depth_separator.pack(side=tk.LEFT, padx=5)
        
        depth_to_entry = ttk.Entry(
            depth_frame,
            textvariable=self.depth_to,
            width=8
        )
        depth_to_entry.pack(side=tk.LEFT)
        
        # Format help
        depth_help = ttk.Label(
            depth_frame,
            text="Format: 0.0-0.0",
            font=("Arial", 8),
            foreground="gray"
        )
        depth_help.pack(side=tk.RIGHT, padx=(5, 0))
    
    def _on_ok(self) -> None:
        """Handle OK button click."""
        # Validate input
        try:
            hole_id = self.hole_id.get().strip()
            depth_from_str = self.depth_from.get().strip()
            depth_to_str = self.depth_to.get().strip()
            
            # Validate hole ID
            if not hole_id:
                messagebox.showerror("Validation Error", "Hole ID is required")
                return
            
            # Validate depth range if provided
            depth_from = None
            depth_to = None
            
            if depth_from_str:
                try:
                    depth_from = float(depth_from_str)
                except ValueError:
                    messagebox.showerror("Validation Error", "Depth From must be a number")
                    return
            
            if depth_to_str:
                try:
                    depth_to = float(depth_to_str)
                except ValueError:
                    messagebox.showerror("Validation Error", "Depth To must be a number")
                    return
            
            # Set result
            self.result = {
                'hole_id': hole_id,
                'depth_from': depth_from,
                'depth_to': depth_to
            }
            
            # Close dialog
            self.dialog.destroy()
            
        except Exception as e:
            logger.error(f"Error validating metadata input: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()
    
    def show(self) -> Optional[Dict[str, Any]]:
        """
        Show the dialog and wait for user input.
        
        Returns:
            Dictionary with entered metadata or None if canceled
        """
        # Wait for dialog to be closed
        self.dialog.wait_window()
        return self.result


class DuplicateHandler:
    """
    Manages detection and handling of duplicate image processing entries.
    
    Tracks processed entries to prevent unintentional duplicate processing.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the duplicate handler.
        
        Args:
            output_dir: Directory where processed images are saved
        """
        self.output_dir = output_dir
        self.processed_entries = self._load_existing_entries()
    
    def _generate_entry_key(self, hole_id: str, depth_from: float, depth_to: float) -> str:
        """
        Generate a unique key for a processed entry.
        
        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
        
        Returns:
            Standardized key representing the entry
        """
        return f"{hole_id.upper()}_{depth_from:.1f}-{depth_to:.1f}"
    
    def _load_existing_entries(self) -> Dict[str, List[str]]:
        """
        Load existing processed entries from saved files in the output directory.
        
        Returns:
            Dictionary of processed entries
        """
        entries = {}
        
        # Safety check for output directory
        if not os.path.exists(self.output_dir):
            return entries
        
        # Find all files in output directory TODO - The files will not always start with KM...
        for filename in os.listdir(self.output_dir):
            # Try to extract metadata from filename
            match = re.match(r'(KM\d{4})_.*_(\d+\.\d)-(\d+\.\d)m', filename)
            if match:
                hole_id, depth_from, depth_to = match.groups()
                key = self._generate_entry_key(hole_id, float(depth_from), float(depth_to))
                
                # Track files for this key
                if key not in entries:
                    entries[key] = []
                entries[key].append(filename)
        
        return entries
    
    def check_duplicate(
        self, 
        hole_id: str, 
        depth_from: float, 
        depth_to: float,
        small_image: np.ndarray,
        full_filename: str
    ) -> bool:
        """
        Check if an entry is a potential duplicate and prompt user.
        
        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            small_image: Downsampled image for comparison
            full_filename: Full path to the original image file
        
        Returns:
            bool: True if processing should continue, False if skipped
        """
        # Generate key for current entry
        entry_key = self._generate_entry_key(hole_id, depth_from, depth_to)
        
        # Check if this exact entry exists
        if entry_key in self.processed_entries:
            # Create duplicate resolution dialog
            result = self._show_duplicate_dialog(
                hole_id, 
                depth_from, 
                depth_to, 
                small_image, 
                self.processed_entries[entry_key]
            )
            
            return result
        
        return True
    
    def _show_duplicate_dialog(
        self, 
        hole_id: str, 
        depth_from: float, 
        depth_to: float, 
        small_image: np.ndarray, 
        existing_files: List[str]
    ) -> bool:
        """
        Show dialog for duplicate resolution.
        
        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            small_image: Downsampled image for comparison
            existing_files: List of existing files for this entry
        
        Returns:
            bool: True to continue processing, False to skip
        """
        # Create a Tkinter dialog for duplicate resolution
        dialog = tk.Toplevel()
        dialog.title("Duplicate Entry Detected")
        dialog.geometry("800x600")
        
        # Variables to track user decision
        decision = tk.StringVar(value="skip")
        
        # Create frames
        top_frame = ttk.Frame(dialog, padding=10)
        top_frame.pack(fill=tk.X)
        
        image_frame = ttk.Frame(dialog, padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        button_frame = ttk.Frame(dialog, padding=10)
        button_frame.pack(fill=tk.X)
        
        # Warning message
        warning_label = ttk.Label(
            top_frame, 
            text=f"Duplicate detected for Hole {hole_id}, Depth {depth_from}-{depth_to}m",
            font=("Arial", 12, "bold")
        )
        warning_label.pack(pady=10)
        
        # Existing files list
        existing_label = ttk.Label(
            top_frame, 
            text="Existing Files:"
        )
        existing_label.pack()
        
        existing_listbox = tk.Listbox(top_frame, height=3)
        existing_listbox.pack(fill=tk.X, padx=10)
        for file in existing_files:
            existing_listbox.insert(tk.END, file)
        
        # Image display (convert to Tkinter-compatible image)
        from PIL import Image, ImageTk
        
        # Resize image
        max_height = 400
        h, w = small_image.shape[:2]
        scale = max_height / h
        resized = cv2.resize(small_image, (int(w * scale), max_height))
        
        # Convert to RGB
        if len(resized.shape) == 3:
            if resized.shape[2] == 3:
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = resized
        else:
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        image_label = ttk.Label(image_frame, image=tk_img)
        image_label.image = tk_img  # Keep a reference
        image_label.pack(expand=True)
        
        # Radio button options
        ttk.Radiobutton(
            button_frame, 
            text="Skip Processing This Image", 
            variable=decision, 
            value="skip"
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(
            button_frame, 
            text="Edit Metadata for This Image", 
            variable=decision, 
            value="edit"
        ).pack(side=tk.LEFT, padx=10)
        
        # OK button
        ok_button = ttk.Button(
            button_frame, 
            text="Continue", 
            command=dialog.destroy
        )
        ok_button.pack(side=tk.RIGHT, padx=10)
        
        # Wait for dialog
        dialog.wait_window()
        
        # Return decision
        return decision.get() == "edit"
    
    def register_processed_entry(
        self, 
        hole_id: str, 
        depth_from: float, 
        depth_to: float, 
        output_files: List[str]
    ):
        """
        Register a successfully processed entry.
        
        Args:
            hole_id: Unique hole identifier
            depth_from: Starting depth
            depth_to: Ending depth
            output_files: List of files generated for this entry
        """
        entry_key = self._generate_entry_key(hole_id, depth_from, depth_to)
        self.processed_entries[entry_key] = output_files


class ChipTrayExtractor:
    """
    Main class for extracting chip tray compartments from panoramic images.
    """

    def __init__(self):
        """Initialize the chip tray extractor with default settings."""
        self.progress_queue = queue.Queue()
        self.processing_complete = False
        self.root = None
        
        # Initialize Tesseract manager
        self.tesseract_manager = TesseractManager()
        
        # Configuration settings (these can be modified via the GUI)
        self.config = {
            # Output settings
            'output_folder': 'extracted_compartments',
            'debug_folder': 'debug',  # Always use 'debug' subfolder
            'save_debug_images': True,
            'output_format': 'jpg',
            'jpeg_quality': 100,

                # Add blur detection settings to config
            'enable_blur_detection': True,
            'blur_threshold': 100.0,  # Default threshold for Laplacian variance
            'blur_roi_ratio': 0.8,    # Use 80% of center image area for blur detection
            'flag_blurry_images': True,  # Whether to visually flag blurry images
            'blurry_threshold_percentage': 30.0,  # Percentage of blurry compartments to flag the tray
            'save_blur_visualizations': True,  # Whether to save blur detection visualizations
            
            # ArUco marker settings
            'aruco_dict_type': cv2.aruco.DICT_4X4_1000,
            'corner_marker_ids': [0, 1, 2, 3],  # Top-left, top-right, bottom-right, bottom-left
            'compartment_marker_ids': list(range(4, 24)),  # 20 markers for compartments
            
            # Processing settings
            'compartment_count': 20,
            
            # OCR settings
            'enable_ocr': self.tesseract_manager.is_available,
            'ocr_confidence_threshold': 70.0,  # Minimum confidence to accept OCR results
            'ocr_config_options': [
                '--psm 11 --oem 3',  # Sparse text, LSTM engine
                '--psm 6 --oem 3',   # Assume a single block of text
                '--psm 3 --oem 3',   # Fully automatic page segmentation
            ],
            'use_metadata_for_filenames': True,
            'metadata_filename_pattern': '{hole_id}_CC_{depth_from}-{depth_to}m',
            'prompt_for_metadata': True,  # Ask user to confirm/correct OCR results
        }
        
        # Pass config to tesseract manager
        self.tesseract_manager.config = self.config

        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.config['aruco_dict_type'])
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

            # Initialize blur detector
        self.blur_detector = BlurDetector(
            threshold=self.config['blur_threshold'],
            roi_ratio=self.config['blur_roi_ratio']
        )

        # Initialise Drill Trace Generator
        self.trace_generator = DrillholeTraceGenerator(config=self.config, progress_queue=self.progress_queue, root=self.root)


    def select_folder(self) -> Optional[str]:
        """Open folder picker dialog and return selected folder path."""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        folder_path = filedialog.askdirectory(title="Select folder with chip tray photos")
        return folder_path if folder_path else None

    
    def get_debug_dir(self, base_path: str) -> str:
        """
        Get the path to the debug directory based on output settings.
        
        Args:
            base_path: Base path of the input file or directory
            
        Returns:
            Path to the debug directory
        """
        # Get the parent directory of the base path
        parent_dir = os.path.dirname(os.path.abspath(base_path))
        
        # Create path to debug directory in the same location as output folder
        debug_dir = os.path.join(parent_dir, self.config['debug_folder'])
        
        # Ensure the directory exists
        os.makedirs(debug_dir, exist_ok=True)
        
        return debug_dir

    def _handle_metadata_dialog_on_main_thread(self, ocr_metadata: Dict[str, Any], 
                                            result_queue: queue.Queue) -> None:
        """
        Helper method to show metadata dialog on the main thread and put result in queue.
        
        Args:
            ocr_metadata: OCR extracted metadata
            result_queue: Queue to put the result in
        """
        try:
            # Ensure we have a root window for the dialog
            if not hasattr(self, 'root') or self.root is None:
                logger.warning("No GUI root window available for metadata dialog")
                result_queue.put(None)
                return

            # Create dialog with both metadata_region and metadata_region_viz
            dialog = MetadataInputDialog(
                parent=self.root,
                image=ocr_metadata.get('metadata_region_viz'),  # Pass the visualization image
                metadata=ocr_metadata
            )
            
            # Show dialog and get result
            result = dialog.show()
            
            # If the user cancels, confirm cancellation
            if result is None:
                confirm = messagebox.askyesno(
                    "Confirm Cancellation",
                    "Are you sure you want to skip processing this image?\n\n"
                    "This will be logged as a failed processing attempt.",
                    icon='warning'
                )
                
                if confirm:
                    # User confirmed cancellation
                    logger.warning("User cancelled metadata input, skipping image processing")
                    result_queue.put(None)  # Signal that processing should be aborted
                else:
                    # User changed their mind, show the dialog again
                    self._handle_metadata_dialog_on_main_thread(ocr_metadata, result_queue)
                    return
            else:
                # User provided metadata
                result_queue.put(result)
            
        except Exception as e:
            logger.error(f"Error creating metadata dialog: {str(e)}")
            logger.error(traceback.format_exc())
            result_queue.put(None)

    def detect_blur_in_compartments(self, 
                                compartments: List[np.ndarray], 
                                save_dir: Optional[str] = None,
                                base_filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect blur in extracted compartment images.
        
        Args:
            compartments: List of compartment images
            save_dir: Optional directory to save blur visualizations
            base_filename: Original filename for generating visualization filenames
            
        Returns:
            List of dictionaries with blur analysis results
        """
        if not self.config['enable_blur_detection']:
            # Return empty results if blur detection is disabled
            return [{'index': i, 'is_blurry': False, 'variance': 0.0} for i in range(len(compartments))]
        
        # Configure blur detector with current settings
        self.blur_detector.threshold = self.config['blur_threshold']
        self.blur_detector.roi_ratio = self.config['blur_roi_ratio']
        
        # Analyze all compartments
        generate_viz = self.config['save_blur_visualizations'] and save_dir and base_filename
        blur_results = self.blur_detector.batch_analyze_images(compartments, generate_viz)
        
        # Save visualizations if enabled
        if generate_viz:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(base_filename))[0]
            
            for result in blur_results:
                if 'visualization' in result:
                    viz_filename = f"{base_name}_compartment_{result['index']+1}_blur_analysis.jpg"
                    viz_path = os.path.join(save_dir, viz_filename)
                    cv2.imwrite(viz_path, result['visualization'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Log summary
        blurry_count = sum(1 for result in blur_results if result.get('is_blurry', False))
        logger.info(f"Blur detection: {blurry_count}/{len(compartments)} compartments are blurry")
        
        # Update progress queue if available
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((f"Blur detection: {blurry_count}/{len(compartments)} compartments are blurry", None))
        
        return blur_results

    # Add this method to ChipTrayExtractor class
    def add_blur_indicators(self, 
                        compartment_images: List[np.ndarray], 
                        blur_results: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Add visual indicators to blurry compartment images.
        
        Args:
            compartment_images: List of compartment images
            blur_results: List of blur analysis results
            
        Returns:
            List of compartment images with blur indicators added
        """
        if not self.config['flag_blurry_images']:
            return compartment_images
        
        result_images = []
        
        for i, image in enumerate(compartment_images):
            # Find the blur result for this image
            result = next((r for r in blur_results if r['index'] == i), None)
            
            if result and result.get('is_blurry', False):
                # Create a copy for modification
                marked_image = image.copy()
                
                # Get image dimensions
                h, w = marked_image.shape[:2]
                
                # Add a red border
                border_thickness = max(3, min(h, w) // 50)  # Scale with image size
                cv2.rectangle(
                    marked_image, 
                    (0, 0), 
                    (w - 1, h - 1), 
                    (0, 0, 255),  # Red in BGR
                    border_thickness
                )
                
                # Add "BLURRY" text
                font_scale = max(0.5, min(h, w) / 500)  # Scale with image size
                text_size, _ = cv2.getTextSize(
                    "BLURRY", 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    2
                )
                
                # Position text in top-right corner
                text_x = w - text_size[0] - 10
                text_y = text_size[1] + 10
                
                # Add background rectangle for better visibility
                cv2.rectangle(
                    marked_image,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Add text
                cv2.putText(
                    marked_image,
                    "BLURRY",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),  # Red in BGR
                    2
                )
                
                # Add variance value in smaller text
                variance_text = f"Var: {result.get('variance', 0):.1f}"
                cv2.putText(
                    marked_image,
                    variance_text,
                    (text_x, text_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.7,
                    (0, 200, 255),  # Orange in BGR
                    1
                )
                
                result_images.append(marked_image)
            else:
                # Keep original image
                result_images.append(image)
        
        return result_images


    def improve_aruco_detection(self, image: np.ndarray) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Attempt to improve ArUco marker detection using various image preprocessing techniques.
        
        Args:
            image: Input image as numpy array
                
        Returns:
            Tuple of (dict mapping marker IDs to corner coordinates, visualization image)
        """
        # Try original detection first
        markers, viz_image = self.detect_aruco_markers(image)
        initial_marker_count = len(markers)
        
        # Log initial detection
        logger.info(f"Initial detection found {initial_marker_count} ArUco markers")
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((f"Initial detection found {initial_marker_count} ArUco markers", None))
        
        # Check if we need to improve detection
        expected_markers = len(self.config['corner_marker_ids']) + len(self.config['compartment_marker_ids'])
        if initial_marker_count >= expected_markers:
            # All markers detected, no need for improvement
            return markers, viz_image
        
        # Store the best result
        best_markers = markers
        best_viz = viz_image
        best_count = initial_marker_count
        
        # Initialize a combined marker dictionary
        combined_markers = markers.copy()
        
        # Try different preprocessing methods
        preprocessing_methods = [
            ("Adaptive thresholding", self._preprocess_adaptive_threshold),
            ("Histogram equalization", self._preprocess_histogram_equalization),
            ("Contrast enhancement", self._preprocess_contrast_enhancement),
            ("Edge enhancement", self._preprocess_edge_enhancement)
        ]
        
        for method_name, preprocess_func in preprocessing_methods:
            # Update status
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((f"Applying {method_name} to improve ArUco detection...", None))
            
            # Apply preprocessing
            processed_image = preprocess_func(image)
            
            # Detect markers on processed image
            new_markers, new_viz = self.detect_aruco_markers(processed_image)
            
            # Log results
            logger.info(f"{method_name}: detected {len(new_markers)} ArUco markers")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((f"{method_name}: detected {len(new_markers)} ArUco markers", None))
            
            # Check if this method found any new markers
            new_marker_ids = set(new_markers.keys()) - set(combined_markers.keys())
            if new_marker_ids:
                logger.info(f"{method_name} found {len(new_marker_ids)} new markers: {sorted(new_marker_ids)}")
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((f"{method_name} found {len(new_marker_ids)} new markers: {sorted(new_marker_ids)}", None))
                
                # Add new markers to combined set
                for marker_id in new_marker_ids:
                    combined_markers[marker_id] = new_markers[marker_id]
            
            # Update best result if this method found more markers
            if len(new_markers) > best_count:
                best_markers = new_markers
                best_viz = new_viz
                best_count = len(new_markers)
        
        # Log final results
        logger.info(f"Best single method found {best_count} markers")
        logger.info(f"Combined methods found {len(combined_markers)} markers")
        
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((f"Best single method found {best_count} markers", None))
            self.progress_queue.put((f"Combined methods found {len(combined_markers)} markers", None))
        
        # Use combined markers if better than any single method
        if len(combined_markers) > best_count:
            # We need to create a visualization for the combined markers
            combined_viz = image.copy()
            
            # Draw all the combined markers
            for marker_id, corners in combined_markers.items():
                # Draw marker outline
                cv2.polylines(combined_viz, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                
                # Add marker ID label
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))
                cv2.putText(combined_viz, f"ID:{marker_id}", (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return combined_markers, combined_viz
        else:
            return best_markers, best_viz

    def _preprocess_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to improve marker detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def _preprocess_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to improve contrast."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return cv2.equalizeHist(gray)

    def _preprocess_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _preprocess_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges to improve marker detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect potential markers
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return dilated

        """Apply enhanced preprocessing to improve OCR results."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive thresholding to handle different lighting conditions
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilation to make text more prominent
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Save debug images if enabled and filename provided
        if self.config.get('save_debug_images', False) and original_filename:
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(original_filename)), 
                                    self.config.get('debug_folder', 'debug'))
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save enhanced image
            debug_enhanced = generate_debug_filename(original_filename, 'enhanced_for_ocr')
            cv2.imwrite(os.path.join(debug_dir, debug_enhanced), enhanced)
            
            # Save binary image
            debug_binary = generate_debug_filename(original_filename, 'binary_for_ocr')
            cv2.imwrite(os.path.join(debug_dir, debug_binary), binary)
            
            # Save dilated image
            debug_dilated = generate_debug_filename(original_filename, 'dilated_for_ocr')
            cv2.imwrite(os.path.join(debug_dir, debug_dilated), dilated)
        
        return dilated


    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Detect ArUco markers in the image with improved feedback.
        
        Args:
            image: Input image as numpy array
                
        Returns:
            Tuple of (dict mapping marker IDs to corner coordinates, visualization image)
        """
        # Create a copy for visualization
        viz_image = image.copy()
        
        # Convert to grayscale for marker detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        # Create a dictionary to store detected markers
        markers = {}
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers on visualization image
            cv2.aruco.drawDetectedMarkers(viz_image, corners, ids)
            
            # Store markers in dictionary with ID as key
            for i, marker_id in enumerate(ids.flatten()):
                markers[marker_id] = corners[i][0]  # Each corner is a 4x2 array
                
                # Add marker ID label
                center_x = int(np.mean(corners[i][0][:, 0]))
                center_y = int(np.mean(corners[i][0][:, 1]))
                cv2.putText(viz_image, f"ID:{marker_id}", (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Log detected markers and missing markers
            expected_markers = set(self.config['corner_marker_ids'] + self.config['compartment_marker_ids'])
            detected_markers = set(markers.keys())
            missing_markers = expected_markers - detected_markers
            
            logger.info(f"Detected {len(ids)} ArUco markers out of {len(expected_markers)} expected")
            logger.info(f"Missing markers: {sorted(missing_markers) if missing_markers else 'None'}")
            
            # Update status in GUI if available
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((f"Detected {len(ids)}/{len(expected_markers)} ArUco markers", None))
                if missing_markers:
                    self.progress_queue.put((f"Missing markers: {sorted(missing_markers)}", None))
        else:
            logger.warning("No ArUco markers detected in the image")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("No ArUco markers detected in the image", None))
        
        return markers, viz_image
    
    def extract_compartment_boundaries(self, 
                                    image: np.ndarray, 
                                    markers: Dict[int, np.ndarray]) -> Tuple[Optional[List[Tuple[int, int, int, int]]], np.ndarray]:
        """
        Extract compartment boundaries using detected ArUco markers.
        Each marker defines a single compartment directly beneath it.
        
        Args:
            image: Input image
            markers: Dictionary mapping marker IDs to corner coordinates
            
        Returns:
            Tuple of (list of compartment boundaries as (x1, y1, x2, y2), visualization image)
        """
        # Create a copy for visualization
        viz_image = image.copy()
        
        # Log marker detection summary
        logger.info(f"Working with {len(markers)} markers: {sorted(markers.keys())}")
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((f"Working with {len(markers)} markers: {sorted(markers.keys())}", None))
        
        # Check if we have enough markers
        corner_markers = [markers.get(id) for id in self.config['corner_marker_ids']]
        corner_markers = [m for m in corner_markers if m is not None]
        
        if len(corner_markers) < 2:
            logger.error("Not enough corner markers detected to establish tray boundaries")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("Not enough corner markers detected", None))
            return None, viz_image
        
        # Find top and bottom boundaries using corner markers (same as before)
        corner_marker_ids = [id for id in self.config['corner_marker_ids'] if id in markers]
        
        # Calculate top boundary (y-min of top markers)
        top_marker_ids = [0, 1]  # IDs of markers that should be at top
        top_markers = [markers[id] for id in corner_marker_ids if id in top_marker_ids]
        
        if not top_markers:
            logger.warning("No top markers found, using top-most detected marker")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("No top markers found, using top-most detected marker", None))
            # Use the top-most marker(s) instead
            marker_y_means = {id: np.mean(markers[id][:, 1]) for id in corner_marker_ids}
            sorted_by_y = sorted(marker_y_means.items(), key=lambda x: x[1])
            top_markers = [markers[sorted_by_y[0][0]]]
        
        # Find lowest y-coordinate of top markers (highest in image)
        top_y = min([np.min(marker[:, 1]) for marker in top_markers])
        
        # Calculate bottom boundary (y-max of bottom markers)
        bottom_marker_ids = [2, 3]  # IDs of markers that should be at bottom
        bottom_markers = [markers[id] for id in corner_marker_ids if id in bottom_marker_ids]
        
        if not bottom_markers:
            logger.warning("No bottom markers found, using bottom-most detected marker")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("No bottom markers found, using bottom-most detected marker", None))
            # Use the bottom-most marker(s) instead
            marker_y_means = {id: np.mean(markers[id][:, 1]) for id in corner_marker_ids}
            sorted_by_y = sorted(marker_y_means.items(), key=lambda x: x[1], reverse=True)
            bottom_markers = [markers[sorted_by_y[0][0]]]
        
        # Find highest y-coordinate of bottom markers (lowest in image)
        bottom_y = max([np.max(marker[:, 1]) for marker in bottom_markers])
        
        # Draw horizontal boundary lines
        cv2.line(viz_image, (0, int(top_y)), (image.shape[1], int(top_y)), (0, 255, 0), 2)
        cv2.line(viz_image, (0, int(bottom_y)), (image.shape[1], int(bottom_y)), (0, 255, 0), 2)
        
        # Get compartment marker IDs - ensure they are sorted by ID
        compartment_marker_ids = sorted([id for id in self.config['compartment_marker_ids'] if id in markers])
        
        if len(compartment_marker_ids) < 1:
            logger.error("Not enough compartment markers detected")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("Not enough compartment markers detected", None))
            return None, viz_image
        
        # Calculate marker information - now we need left/right edges
        marker_info = {}
        
        for marker_id in compartment_marker_ids:
            marker_corners = markers[marker_id]
            
            # Calculate marker center
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # Calculate marker width - direct from marker
            left_x = int(np.min(marker_corners[:, 0]))
            right_x = int(np.max(marker_corners[:, 0]))
            width = right_x - left_x
            
            marker_info[marker_id] = {
                'center_x': center_x,
                'center_y': center_y,
                'left_x': left_x,
                'right_x': right_x,
                'width': width
            }
            
            # Draw marker centers and boundaries on visualization
            cv2.circle(viz_image, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(viz_image, f"{marker_id}", (center_x - 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Draw the marker's boundaries
            cv2.line(viz_image, (left_x, int(top_y)), (left_x, int(bottom_y)), (0, 0, 255), 1)
            cv2.line(viz_image, (right_x, int(top_y)), (right_x, int(bottom_y)), (0, 0, 255), 1)
        
        # Report detected markers
        detected_msg = f"Detected {len(compartment_marker_ids)}/{len(self.config['compartment_marker_ids'])} compartment markers"
        logger.info(detected_msg)
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((detected_msg, None))
        
        # Find average marker width for estimating missing markers
        avg_marker_width = sum(info['width'] for info in marker_info.values()) / len(marker_info)
        logger.info(f"Average marker width: {avg_marker_width:.2f} pixels")
        
        # Calculate spacings between consecutive markers to understand the pattern
        spacings = []
        for i in range(len(compartment_marker_ids) - 1):
            curr_id = compartment_marker_ids[i]
            next_id = compartment_marker_ids[i + 1]
            if next_id - curr_id == 1:  # Only consider consecutive markers
                curr_center = marker_info[curr_id]['center_x']
                next_center = marker_info[next_id]['center_x']
                spacing = next_center - curr_center
                spacings.append(spacing)
        
        avg_spacing = np.median(spacings) if spacings else avg_marker_width * 1.5
        logger.info(f"Average spacing between consecutive markers: {avg_spacing:.2f} pixels")
        
        # Identify missing markers and estimate their positions
        expected_marker_ids = self.config['compartment_marker_ids']
        missing_marker_ids = [id for id in expected_marker_ids if id not in marker_info]
        
        # Estimate positions of missing markers
        estimated_marker_info = {}
        
        if missing_marker_ids:
            logger.info(f"Estimating positions for {len(missing_marker_ids)} missing markers")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((f"Estimating positions for {len(missing_marker_ids)} missing markers", None))
            
            # First, copy all detected marker info
            for marker_id, info in marker_info.items():
                estimated_marker_info[marker_id] = info.copy()
            
            # Then estimate missing markers
            for marker_id in missing_marker_ids:
                # Find the closest lower and higher marker IDs that were detected
                lower_ids = [id for id in compartment_marker_ids if id < marker_id]
                higher_ids = [id for id in compartment_marker_ids if id > marker_id]
                
                if lower_ids and higher_ids:
                    # Interpolate between the closest markers
                    lower_id = max(lower_ids)
                    higher_id = min(higher_ids)
                    lower_center = marker_info[lower_id]['center_x']
                    higher_center = marker_info[higher_id]['center_x']
                    
                    # Linear interpolation for center position
                    weight = (marker_id - lower_id) / (higher_id - lower_id)
                    estimated_center = lower_center + weight * (higher_center - lower_center)
                    
                    # Use average width for the missing marker
                    estimated_width = avg_marker_width
                    
                    estimated_marker_info[marker_id] = {
                        'center_x': int(estimated_center),
                        'center_y': int((top_y + bottom_y) / 2),  # Middle of the tray
                        'left_x': int(estimated_center - estimated_width / 2),
                        'right_x': int(estimated_center + estimated_width / 2),
                        'width': estimated_width,
                        'estimated': True
                    }
                elif lower_ids:
                    # Extrapolate forward
                    lower_id = max(lower_ids)
                    lower_center = marker_info[lower_id]['center_x']
                    estimated_center = lower_center + (marker_id - lower_id) * avg_spacing
                    
                    estimated_marker_info[marker_id] = {
                        'center_x': int(estimated_center),
                        'center_y': int((top_y + bottom_y) / 2),
                        'left_x': int(estimated_center - avg_marker_width / 2),
                        'right_x': int(estimated_center + avg_marker_width / 2),
                        'width': avg_marker_width,
                        'estimated': True
                    }
                elif higher_ids:
                    # Extrapolate backward
                    higher_id = min(higher_ids)
                    higher_center = marker_info[higher_id]['center_x']
                    estimated_center = higher_center - (higher_id - marker_id) * avg_spacing
                    
                    estimated_marker_info[marker_id] = {
                        'center_x': int(estimated_center),
                        'center_y': int((top_y + bottom_y) / 2),
                        'left_x': int(estimated_center - avg_marker_width / 2),
                        'right_x': int(estimated_center + avg_marker_width / 2),
                        'width': avg_marker_width,
                        'estimated': True
                    }
                else:
                    logger.warning(f"Cannot estimate position for marker {marker_id}")
                    continue
                
                # Draw estimated marker position
                info = estimated_marker_info[marker_id]
                cv2.circle(viz_image, (info['center_x'], info['center_y']), 5, (0, 165, 255), -1)
                cv2.putText(viz_image, f"E{marker_id}", (info['center_x'] - 10, info['center_y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                # Draw estimated compartment boundaries
                cv2.line(viz_image, (info['left_x'], int(top_y)), (info['left_x'], int(bottom_y)), (0, 165, 255), 1)
                cv2.line(viz_image, (info['right_x'], int(top_y)), (info['right_x'], int(bottom_y)), (0, 165, 255), 1)
        else:
            # Just use the detected markers if none are missing
            estimated_marker_info = marker_info
        
        # Now create the compartment boundaries directly from the markers
        compartment_boundaries = []
        
        # The expected compartment count
        expected_count = self.config['compartment_count']
        
        # Sort marker IDs to ensure we get the correct order
        all_marker_ids = sorted(estimated_marker_info.keys())
        
        # Only use the first expected_count markers
        used_marker_ids = all_marker_ids[:expected_count] if len(all_marker_ids) > expected_count else all_marker_ids
        
        for marker_id in used_marker_ids:
            info = estimated_marker_info[marker_id]
            
            # Use the marker's dimensions directly for the compartment
            x1 = info['left_x']
            x2 = info['right_x']
            y1 = int(top_y)
            y2 = int(bottom_y)
            
            compartment_boundaries.append((x1, y1, x2, y2))
            
            # Draw compartment boundary
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw marker ID
            marker_idx = used_marker_ids.index(marker_id) + 1  # 1-based indexing for display
            cv2.putText(viz_image, f"C{marker_idx}", ((x1 + x2) // 2, (y1 + y2) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Check if we found all expected compartments
        if len(compartment_boundaries) < expected_count:
            logger.warning(f"Only found {len(compartment_boundaries)}/{expected_count} compartments")
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((f"Only found {len(compartment_boundaries)}/{expected_count} compartments", "warning"))
        
        # Log summary of compartment detection
        logger.info(f"Successfully identified {len(compartment_boundaries)} compartment boundaries")
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((f"Successfully identified {len(compartment_boundaries)} compartment boundaries", None))
                
        return compartment_boundaries, viz_image

    def correct_image_skew(self, image: np.ndarray, markers: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Correct image skew using corner ArUco markers.
        
        Args:
            image: Input image as numpy array
            markers: Dictionary mapping marker IDs to corner coordinates
            
        Returns:
            Deskewed image as numpy array
        """
        # Get available corner markers
        corner_ids = [id for id in self.config['corner_marker_ids'] if id in markers]
        
        if len(corner_ids) < 2:
            logger.warning("Not enough corner markers for skew correction")
            return image
        
        # Get marker centers
        corner_centers = []
        for id in corner_ids:
            corners = markers[id]
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            corner_centers.append((id, center_x, center_y))
        
        # Check if we have top markers for angle calculation
        top_markers = [m for m in corner_centers if m[0] in [0, 1]]
        if len(top_markers) >= 2:
            # Sort by x-coordinate
            top_markers.sort(key=lambda m: m[1])
            
            # Calculate angle
            _, x1, y1 = top_markers[0]
            _, x2, y2 = top_markers[1]
            
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            
            if abs(angle_deg) > 0.5:  # Only correct if angle is significant
                logger.info(f"Correcting image skew of {angle_deg:.2f} degrees")
                
                # Get image center for rotation
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                
                # Get rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
                
                # Apply rotation
                corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                            flags=cv2.INTER_LINEAR, 
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(255, 255, 255))
                return corrected_image
        
        # If we can't correct or angle is minimal, return original
        return image

    def move_processed_image(self, source_path: str, output_dir: str) -> Optional[str]:
        """
        Move a successfully processed image to a 'Processed Originals' subfolder.

        Args:
            source_path (str): Path to the original image file
            output_dir (str): Base output directory for the current processing run

        Returns:
            Optional[str]: Path to the new image location, or None if move fails
        """
        try:
            # Create 'Processed Originals' directory if it doesn't exist
            processed_dir = os.path.join(output_dir, 'Processed Originals')
            os.makedirs(processed_dir, exist_ok=True)

            # Generate a unique filename to prevent overwriting
            base_name = os.path.basename(source_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{os.path.splitext(base_name)[0]}_{timestamp}{os.path.splitext(base_name)[1]}"
            
            # Full path for the new location
            destination_path = os.path.join(processed_dir, new_filename)

            # Move the file
            shutil.move(source_path, destination_path)
            
            logger.info(f"Moved processed image to: {destination_path}")
            return destination_path

        except Exception as e:
            logger.error(f"Error moving processed image {source_path}: {str(e)}")
            return None

    def process_image(self, image_path: str) -> bool:
        """
        Process a single chip tray image with enhanced feedback and improved detection.
        Uses a two-phase approach: first detects markers and metadata on a downsized image
        for speed, then applies the results to the full-resolution image for quality output.
        
        Args:
            image_path (str): Path to the image file to be processed
        
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        # Initialize duplicate handler with output directory
        output_dir = os.path.join(os.path.dirname(image_path), self.config['output_folder'])
        duplicate_handler = DuplicateHandler(output_dir)

        try:
            # Try to read the image using PIL first
            try:
                from PIL import Image as PILImage
                import numpy as np
                
                # Open with PIL
                pil_img = PILImage.open(image_path)
                
                # Convert to numpy array for OpenCV processing
                original_image = np.array(pil_img)
                
                # Convert RGB to BGR for OpenCV compatibility if needed
                if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                    
                logger.info(f"Successfully read image with PIL: {image_path}")
                
            except Exception as e:
                logger.warning(f"Failed to read image with PIL, trying OpenCV: {str(e)}")
                
                # Fallback to OpenCV if PIL fails
                original_image = cv2.imread(image_path)
                if original_image is None:
                    error_msg = f"Failed to read image with both PIL and OpenCV: {image_path}"
                    logger.error(error_msg)
                    if hasattr(self, 'progress_queue'):
                        self.progress_queue.put((error_msg, None))
                    return False
            
            # Create output directories
            output_dir = os.path.join(os.path.dirname(image_path), self.config['output_folder'])
            os.makedirs(output_dir, exist_ok=True)
            
            debug_dir = None
            if self.config['save_debug_images']:
                debug_dir = self.get_debug_dir(image_path)
                os.makedirs(debug_dir, exist_ok=True)
            
            # Update status
            base_name = os.path.basename(image_path)
            status_msg = f"Processing Image: {base_name}"
            logger.info(status_msg)
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((status_msg, None))
            
            # Calculate scale based on original size - aim for ~1-2MB file size
            h, w = original_image.shape[:2]
            original_pixels = h * w
            target_pixels = 2000000  # Target ~2 million pixels (e.g., 1414x1414)
            
            if original_pixels > target_pixels:
                scale = (target_pixels / original_pixels) ** 0.5
                new_width = int(w * scale)
                new_height = int(h * scale)
                
                # Create downsampled image
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((f"Downsizing image from {w}x{h} to {new_width}x{new_height} for processing", None))
                    
                small_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Save downsized debug image if enabled
                if debug_dir:
                    debug_filename = generate_debug_filename(image_path, 'original_small')
                    debug_path = os.path.join(debug_dir, debug_filename)
                    cv2.imwrite(debug_path, small_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    logger.info(f"Saved downsized image for processing: {debug_path}")
            else:
                # Image is already small enough
                small_image = original_image.copy()
                logger.info(f"Image already small ({w}x{h}), using as is for processing")
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((f"Image already small ({w}x{h}), using as is for processing", None))
            
            # Store visualization steps (all on the small image)
            viz_steps = []
            viz_steps.append(("Original Image (Small)", small_image.copy()))
            
            # Step A: Detect ArUco markers with improved methods - using SMALL image
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("Detecting ArUco markers...", None))
            
            markers, markers_viz = self.improve_aruco_detection(small_image)
            viz_steps.append(("ArUco Marker Detection", markers_viz))
            
            if not markers:
                error_msg = "No ArUco markers detected"
                logger.error(error_msg)
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((error_msg, None))
                return False
            
            # Initialize metadata with an empty dictionary
            metadata: Dict[str, Any] = {}
            
            # Extract metadata with OCR if enabled - using SMALL image
            if self.config['enable_ocr'] and self.tesseract_manager.is_available:
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put(("Extracting metadata with OCR...", None))
                
                try:
                    # Use markers to precisely locate metadata
                    config = {
                        'ocr_config_options': self.config['ocr_config_options'],
                        'confidence_threshold': self.config['ocr_confidence_threshold'],
                        'save_debug_images': self.config['save_debug_images']
                    }
                    ocr_metadata = self.tesseract_manager.extract_metadata_from_image(small_image, markers, original_filename=image_path) # TODO - is this running on the small image or on the images which have been processed to improve the OCR???
                    
                    # Log OCR results
                    ocr_log_msg = f"OCR Results: Confidence={ocr_metadata.get('confidence', 0):.1f}%"
                    if ocr_metadata.get('hole_id'):
                        ocr_log_msg += f", Hole ID={ocr_metadata['hole_id']}"
                    if ocr_metadata.get('depth_from') is not None and ocr_metadata.get('depth_to') is not None:
                        ocr_log_msg += f", Depth={ocr_metadata['depth_from']}-{ocr_metadata['depth_to']}"
                    
                    logger.info(ocr_log_msg)
                    if hasattr(self, 'progress_queue'):
                        self.progress_queue.put((ocr_log_msg, None))
                    
                    # Determine if user confirmation is needed TODO - Check this is correct...
                    needs_confirmation = (
                        self.config['prompt_for_metadata'] and (
                            ocr_metadata.get('confidence', 0) < self.config['ocr_confidence_threshold'] or
                            ocr_metadata.get('hole_id') is None or 
                            ocr_metadata.get('depth_from') is None or
                            ocr_metadata.get('depth_to') is None or
                            # Add validation checks for extracted data
                            (ocr_metadata.get('depth_from') is not None and 
                            ocr_metadata.get('depth_to') is not None and
                            ocr_metadata.get('depth_from') >= ocr_metadata.get('depth_to'))
                        )
                    )
                    
                    if needs_confirmation and self.root is not None:
                        # Handle user confirmation through dialog
                        if hasattr(self, 'progress_queue'):
                            self.progress_queue.put(("OCR confidence low, prompting for metadata...", None))
                        
                        # Create a temporary queue for receiving metadata
                        metadata_queue = queue.Queue()
                        
                        # Schedule dialog creation on main thread
                        self.root.after(0, self._handle_metadata_dialog_on_main_thread, 
                                    ocr_metadata, metadata_queue)
                        
                        # Wait for dialog result
                        try:
                            result = metadata_queue.get(timeout=300)  # 5 minute timeout
                            if result:
                                metadata = result
                                logger.info(f"User entered metadata: {metadata}")
                            else:
                                # User canceled the dialog and confirmed cancellation
                                logger.warning("Image processing canceled by user")
                                if hasattr(self, 'progress_queue'):
                                    self.progress_queue.put(("Processing canceled by user", None))
                                return False  # Return early to skip further processing
                        except queue.Empty:
                            logger.warning("Metadata dialog timed out")
                            # Use OCR results as fallback if dialog timed out
                            metadata = {
                                'hole_id': ocr_metadata.get('hole_id'),
                                'depth_from': ocr_metadata.get('depth_from'),
                                'depth_to': ocr_metadata.get('depth_to')
                            }
                            logger.info(f"Using OCR metadata after timeout: {metadata}")
                        except Exception as e:
                            logger.error(f"OCR error: {str(e)}")
                            logger.error(traceback.format_exc())
                            if hasattr(self, 'progress_queue'):
                                self.progress_queue.put((f"OCR error: {str(e)}", None))
                            return False
                    else:
                        # Use OCR metadata without prompting
                        metadata = {
                            'hole_id': ocr_metadata.get('hole_id'),
                            'depth_from': ocr_metadata.get('depth_from'),
                            'depth_to': ocr_metadata.get('depth_to')
                        }
                
                    # Check for potential duplicates AFTER OCR or manual metadata entry
                    if (metadata.get('hole_id') and 
                        metadata.get('depth_from') is not None and 
                        metadata.get('depth_to') is not None):
                        
                        # Check for duplicates
                        try:
                            continue_processing = duplicate_handler.check_duplicate(
                                metadata['hole_id'], 
                                metadata['depth_from'], 
                                metadata['depth_to'], 
                                small_image,  # Use the downsampled image
                                image_path
                            )
                            
                            # If user chooses to skip, return early
                            if not continue_processing:
                                logger.info(f"Skipping duplicate image: {os.path.basename(image_path)}")
                                return False
                        except Exception as e:
                            logger.error(f"Error checking for duplicates: {str(e)}")
                            return False

                    # Update debug file names if we have metadata
                    if metadata and metadata.get('hole_id'):
                        rename_debug_files(
                            image_path, 
                            metadata.get('hole_id'), 
                            metadata.get('depth_from'), 
                            metadata.get('depth_to')
                        )
                except Exception as e:
                    logger.error(f"Error in metadata extraction: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue processing without metadata
            
            # Correct image skew on the SMALL image if possible
            try:
                if markers:
                    corrected_small_image = self.correct_image_skew(small_image, markers)
                    if corrected_small_image is not small_image:  # If correction was applied
                        # Re-detect markers on corrected image
                        markers, markers_viz = self.improve_aruco_detection(corrected_small_image)
                        small_image = corrected_small_image  # Update small image for further processing
                        viz_steps.append(("Corrected Image", markers_viz))
                        
                        logger.info(f"Re-detected {len(markers)} markers after skew correction")
                        if hasattr(self, 'progress_queue'):
                            self.progress_queue.put((f"Re-detected {len(markers)} markers after skew correction", None))
            except Exception as e:
                logger.warning(f"Skew correction failed: {str(e)}")
            
            # Report marker detection status
            expected_markers = set(self.config['corner_marker_ids'] + self.config['compartment_marker_ids'])
            detected_markers = set(markers.keys())
            missing_markers = expected_markers - detected_markers
            
            status_msg = f"Detected {len(detected_markers)}/{len(expected_markers)} ArUco markers"
            logger.info(status_msg)
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((status_msg, None))
                
                if missing_markers:
                    self.progress_queue.put((f"Missing markers: {sorted(missing_markers)}", None))
            
            # Extract compartment boundaries on the SMALL image
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("Extracting compartment boundaries...", None))
            
            try:
                boundaries_result = self.extract_compartment_boundaries(small_image, markers)
                if boundaries_result is None:
                    error_msg = "Failed to extract compartment boundaries"
                    logger.error(error_msg)
                    if hasattr(self, 'progress_queue'):
                        self.progress_queue.put((error_msg, None))
                    return False
                    
                compartment_boundaries_small, boundaries_viz = boundaries_result
            except Exception as e:
                error_msg = f"Error in compartment boundary extraction: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((error_msg, None))
                return False
            
            viz_steps.append(("Compartment Boundaries", boundaries_viz))
            
            # Report number of compartments found
            status_msg = f"Found {len(compartment_boundaries_small)}/{self.config['compartment_count']} compartments"
            logger.info(status_msg)
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((status_msg, None))
            
            # SCALED EXTRACTION: Scale up the coordinates from small image to original image
            if small_image.shape != original_image.shape:
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put(("Scaling boundaries to original image size...", None))
                    
                # Calculate scale factors
                scale_x = original_image.shape[1] / small_image.shape[1]
                scale_y = original_image.shape[0] / small_image.shape[0]
                
                # Scale up the coordinates
                compartment_boundaries = []
                for x1, y1, x2, y2 in compartment_boundaries_small:
                    # Round to integers
                    scaled_x1 = int(x1 * scale_x)
                    scaled_y1 = int(y1 * scale_y)
                    scaled_x2 = int(x2 * scale_x)
                    scaled_y2 = int(y2 * scale_y)
                    
                    compartment_boundaries.append((scaled_x1, scaled_y1, scaled_x2, scaled_y2))
                    
                logger.info(f"Scaled {len(compartment_boundaries)} compartment boundaries to original image")
            else:
                # No scaling needed
                compartment_boundaries = compartment_boundaries_small
            
            # Now extract compartments from the ORIGINAL high-resolution image
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("Extracting high-resolution compartments...", None))
                
            compartments, compartments_viz = self.extract_compartments(original_image, compartment_boundaries)
            
            # Create a visualization from the high-res extraction for debugging
            high_res_viz = original_image.copy()
            for i, (x1, y1, x2, y2) in enumerate(compartment_boundaries):
                cv2.rectangle(high_res_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(high_res_viz, f"{i+1}", (x1 + 10, y1 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Add high-res to viz_steps if needed, but resize it to match other visualizations
            if debug_dir:
                # Save high-res visualization separately
                h_viz, w_viz = high_res_viz.shape[:2]
                target_height = 600  # Same as used in create_visualization_image
                scale = target_height / h_viz
                resized_high_res = cv2.resize(high_res_viz, (int(w_viz * scale), target_height))
                viz_steps.append(("High-Res Compartments", resized_high_res))
                
                # Save the high-res compartment visualization directly
                high_res_viz_filename = generate_debug_filename(image_path, 'highres_compartments')
                high_res_viz_path = os.path.join(debug_dir, high_res_viz_filename)
                # Save with reasonable quality to avoid excessive file size
                cv2.imwrite(high_res_viz_path, cv2.resize(high_res_viz, 
                                                        (min(1920, high_res_viz.shape[1]), 
                                                        min(1080, high_res_viz.shape[0])),
                                                        interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                logger.info(f"Saved high-resolution compartment visualization to {high_res_viz_path}")
            
            # Save the extracted compartments from the original image
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(("Saving compartment images...", None))
                
            num_saved = self.save_compartments(compartments, output_dir, image_path, metadata)
            
            # Create and save visualization image if required
            if debug_dir:
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put(("Saving debug images...", None))
                    
                viz_image = self.create_visualization_image(small_image, viz_steps)  # Use small image for visualization
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                viz_path = os.path.join(debug_dir, f"{base_name}_visualization.jpg")
                cv2.imwrite(viz_path, viz_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                logger.info(f"Saved visualization to {viz_path}")
                
                # Also save individual steps
                for i, (name, img) in enumerate(viz_steps):
                    step_name = name.lower().replace(' ', '_')
                    step_path = os.path.join(debug_dir, f"{base_name}_{i+1:02d}_{step_name}.jpg")
                    cv2.imwrite(step_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Final summary
            success_msg = f"Successfully processed {base_name}: saved {num_saved}/{self.config['compartment_count']} compartments"
            if metadata.get('hole_id'):
                success_msg += f" for hole {metadata['hole_id']}"
                
            logger.info(success_msg)
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((success_msg, None))
                
                # Add details about missing compartments if any
                if num_saved < self.config['compartment_count']:
                    warning_msg = f"Warning: Expected {self.config['compartment_count']} compartments, but saved {num_saved}"
                    logger.warning(warning_msg)
                    self.progress_queue.put((warning_msg, None))
            
            # When processing is complete and successful
            if (metadata.get('hole_id') and 
                metadata.get('depth_from') is not None and 
                metadata.get('depth_to') is not None):
                # Register the processed entry
                output_files = [
                    f"{metadata['hole_id']}_CC_{metadata['depth_from']:.1f}-{metadata['depth_to']:.1f}m_compartment_{i+1}.{self.config['output_format']}"
                    for i in range(len(compartments))
                ]
                
                duplicate_handler.register_processed_entry(
                    metadata['hole_id'], 
                    metadata['depth_from'], 
                    metadata['depth_to'], 
                    output_files
                )
            
            try:
                # Determine processing success based on number of saved compartments
                num_saved = self.save_compartments(compartments, output_dir, image_path, metadata)
                is_processing_successful = num_saved > 0
                # Attempt to move the processed image if successful
                if is_processing_successful:
                    # Create output directory (if it hasn't been created already)
                    output_base_dir = os.path.join(os.path.dirname(image_path), self.config['output_folder'])
                    
                    # Create 'Processed Originals' subdirectory
                    processed_originals_dir = os.path.join(output_base_dir, 'Processed Originals')
                    os.makedirs(processed_originals_dir, exist_ok=True)
                    
                    # Generate unique filename with timestamp
                    base_name = os.path.basename(image_path)
                    name_without_ext, ext = os.path.splitext(base_name)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{name_without_ext}_{timestamp}{ext}"
                    
                    # Full paths for source and destination
                    destination_path = os.path.join(processed_originals_dir, unique_filename)
                    
                    # Move the processed image
                    shutil.move(image_path, destination_path)
                    
                    # Log successful move
                    logger.info(f"Moved processed image to: {destination_path}")
                
                return is_processing_successful
            
            except Exception as move_error:
                # Log move error but don't interrupt overall processing result
                logger.warning(f"Could not move processed image {image_path}: {str(move_error)}")
                return is_processing_successful
        
        except Exception as e:
            # Handle any unexpected errors during processing
            error_msg = f"Error processing {image_path}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Update progress queue if available
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((error_msg, None))
            
            return False

    def process_folder(self, folder_path: str) -> Tuple[int, int]:
        """
        Process all images in a folder with enhanced progress reporting.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Tuple of (number of images processed, number of images failed)
        """
        # Get all image files from the folder
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f)) and 
                    f.lower().endswith(image_extensions)]
        
        if not image_files:
            warning_msg = f"No image files found in {folder_path}"
            logger.warning(warning_msg)
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put((warning_msg, None))
            return 0, 0
        
        # Log found images
        info_msg = f"Found {len(image_files)} image files"
        logger.info(info_msg)
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((info_msg, None))
        
        # Process each image
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_files):
            try:
                # Update progress
                progress = ((i + 1) / len(image_files)) * 100
                progress_msg = f"Processing Image: {i+1}/{len(image_files)}: {os.path.basename(image_path)}"
                logger.info(progress_msg)
                
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((progress_msg, progress))
                
                # Process the image
                if self.process_image(image_path):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                error_msg = f"Error processing {image_path}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((error_msg, None))
                failed += 1
        
        # Log summary
        summary_msg = f"Processing complete: {successful} successful, {failed} failed"
        logger.info(summary_msg)
        if hasattr(self, 'progress_queue'):
            self.progress_queue.put((summary_msg, 100))  # Set progress to 100%
        
        return successful, failed

    def extract_compartments(self, 
                            image: np.ndarray, 
                            compartment_boundaries: List[Tuple[int, int, int, int]]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract individual compartment images based on detected boundaries.
        
        Args:
            image: Input image
            compartment_boundaries: List of (x1, y1, x2, y2) coordinates for each compartment
            
        Returns:
            Tuple of (list of extracted compartment images, visualization image)
        """
        # Create a copy for visualization
        viz_image = image.copy()
        
        # Extract each compartment
        compartments = []
        
        for i, (x1, y1, x2, y2) in enumerate(compartment_boundaries):
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract compartment region
            compartment = image[y1:y2, x1:x2].copy()
            
            # Add to list
            compartments.append(compartment)
            
            # Draw rectangle on visualization image
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz_image, f"{i+1}", (x1 + 10, y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return compartments, viz_image

    def on_generate_trace(self):
        # Select image folder
        compartment_dir = filedialog.askdirectory(title="Select folder with compartment images")
        if not compartment_dir:
            return

        # Select CSV file
        csv_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not csv_path:
            return

        # Initialize the trace generator
        trace_generator = DrillholeTraceGenerator(
            config=self.config, 
            progress_queue=self.progress_queue,
            root=self.root
        )

        # Let user select optional columns
        csv_columns = trace_generator.get_csv_columns(csv_path)
        if not csv_columns:
            messagebox.showerror("CSV Error", "Could not read columns from CSV.")
            return

        selected_columns = trace_generator.select_csv_columns(csv_columns)

        # Run the trace generation
        generated_paths = trace_generator.process_all_drillholes(
            compartment_dir=compartment_dir,
            csv_path=csv_path,
            selected_columns=selected_columns
        )

        if generated_paths:
            messagebox.showinfo("Success", f"Generated {len(generated_paths)} drillhole trace images.")
        else:
            messagebox.showwarning("No Output", "No drillhole trace images were generated.")

    def create_gui(self):
        """Create a GUI for chip tray extraction with enhanced status display."""
        self.root = tk.Tk()
        self.root.title("Chip Tray Extractor")
        
        # Set up the main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Chip Tray Extractor", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Input folder selection
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        folder_frame = ttk.Frame(input_frame)
        folder_frame.pack(fill=tk.X)
        
        folder_label = ttk.Label(folder_frame, text="Input Folder:", width=15, anchor='w')
        folder_label.pack(side=tk.LEFT)
        
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_button = ttk.Button(folder_frame, text="Browse", command=self.browse_folder)
        browse_button.pack(side=tk.RIGHT)
        
        # Output settings
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output folder
        output_folder_frame = ttk.Frame(output_frame)
        output_folder_frame.pack(fill=tk.X, pady=2)
        
        output_folder_label = ttk.Label(output_folder_frame, text="Output Folder:", width=15, anchor='w')
        output_folder_label.pack(side=tk.LEFT)
        
        self.output_folder_var = tk.StringVar(value=self.config['output_folder'])
        output_folder_entry = ttk.Entry(output_folder_frame, textvariable=self.output_folder_var)
        output_folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Output format
        format_frame = ttk.Frame(output_frame)
        format_frame.pack(fill=tk.X, pady=2)
        
        format_label = ttk.Label(format_frame, text="Output Format:", width=15, anchor='w')
        format_label.pack(side=tk.LEFT)
        
        self.format_var = tk.StringVar(value=self.config['output_format'])
        format_options = ['jpg', 'png', 'tiff']
        format_dropdown = ttk.OptionMenu(format_frame, self.format_var, self.config['output_format'], *format_options)
        format_dropdown.pack(side=tk.LEFT)
        
        # JPEG Quality slider
        quality_frame = ttk.Frame(output_frame)
        quality_frame.pack(fill=tk.X, pady=2)
        
        quality_label = ttk.Label(quality_frame, text="JPEG Quality:", width=15, anchor='w')
        quality_label.pack(side=tk.LEFT)
        
        self.quality_var = tk.IntVar(value=self.config['jpeg_quality'])
        quality_slider = ttk.Scale(quality_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                variable=self.quality_var)
        quality_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        quality_value = ttk.Label(quality_frame, textvariable=self.quality_var, width=3)
        quality_value.pack(side=tk.RIGHT)
        
        # Debug options
        debug_frame = ttk.LabelFrame(main_frame, text="Debug Options", padding=10)
        debug_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Save debug images
        self.debug_var = tk.BooleanVar(value=self.config['save_debug_images'])
        debug_check = ttk.Checkbutton(debug_frame, text="Save Debug Images", variable=self.debug_var)
        debug_check.pack(anchor='w')

        # Create blur detection settings frame
        blur_frame = ttk.LabelFrame(main_frame, text="Blur Detection", padding=10)
        blur_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Enable blur detection
        self.blur_enable_var = tk.BooleanVar(value=self.config['enable_blur_detection'])
        enable_check = ttk.Checkbutton(
            blur_frame, 
            text="Enable Blur Detection", 
            variable=self.blur_enable_var,
            command=self._toggle_blur_settings
        )
        enable_check.pack(anchor='w', pady=(0, 5))
        
        # Container for blur settings (enabled/disabled based on checkbox)
        self.blur_settings_frame = ttk.Frame(blur_frame)
        self.blur_settings_frame.pack(fill=tk.X)
        
        # Blur threshold slider
        threshold_frame = ttk.Frame(self.blur_settings_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        
        threshold_label = ttk.Label(threshold_frame, text="Blur Threshold:", width=20, anchor='w')
        threshold_label.pack(side=tk.LEFT)
        
        self.blur_threshold_var = tk.DoubleVar(value=self.config['blur_threshold'])
        threshold_slider = ttk.Scale(
            threshold_frame, 
            from_=10.0, 
            to=500.0, 
            orient=tk.HORIZONTAL, 
            variable=self.blur_threshold_var
        )
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        threshold_value = ttk.Label(threshold_frame, width=5)
        threshold_value.pack(side=tk.RIGHT)
        
        # Update threshold value label when slider changes
        def update_threshold_label(*args):
            threshold_value.config(text=f"{self.blur_threshold_var.get():.1f}")
        
        self.blur_threshold_var.trace_add("write", update_threshold_label)
        update_threshold_label()  # Initial update
        
        # Blur ROI ratio slider
        roi_frame = ttk.Frame(self.blur_settings_frame)
        roi_frame.pack(fill=tk.X, pady=2)
        
        roi_label = ttk.Label(roi_frame, text="ROI Ratio:", width=20, anchor='w')
        roi_label.pack(side=tk.LEFT)
        
        self.blur_roi_var = tk.DoubleVar(value=self.config['blur_roi_ratio'])
        roi_slider = ttk.Scale(
            roi_frame, 
            from_=0.1, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.blur_roi_var
        )
        roi_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        roi_value = ttk.Label(roi_frame, width=5)
        roi_value.pack(side=tk.RIGHT)
        
        # Update ROI value label when slider changes
        def update_roi_label(*args):
            roi_value.config(text=f"{self.blur_roi_var.get():.2f}")
        
        self.blur_roi_var.trace_add("write", update_roi_label)
        update_roi_label()  # Initial update
        
        # Flag blurry images checkbox
        self.flag_blurry_var = tk.BooleanVar(value=self.config['flag_blurry_images'])
        flag_check = ttk.Checkbutton(
            self.blur_settings_frame, 
            text="Flag Blurry Images", 
            variable=self.flag_blurry_var
        )
        flag_check.pack(anchor='w', pady=(5, 0))
        
        # Save blur visualizations checkbox
        self.save_blur_viz_var = tk.BooleanVar(value=self.config['save_blur_visualizations'])
        save_viz_check = ttk.Checkbutton(
            self.blur_settings_frame, 
            text="Save Blur Analysis Visualizations", 
            variable=self.save_blur_viz_var
        )
        save_viz_check.pack(anchor='w', pady=(5, 0))
        
        # Blurry threshold percentage
        threshold_pct_frame = ttk.Frame(self.blur_settings_frame)
        threshold_pct_frame.pack(fill=tk.X, pady=(5, 0))
        
        threshold_pct_label = ttk.Label(
            threshold_pct_frame, 
            text="Quality Alert Threshold:", 
            width=20, 
            anchor='w'
        )
        threshold_pct_label.pack(side=tk.LEFT)
        
        self.blur_threshold_pct_var = tk.DoubleVar(value=self.config['blurry_threshold_percentage'])
        threshold_pct_slider = ttk.Scale(
            threshold_pct_frame, 
            from_=5.0, 
            to=100.0, 
            orient=tk.HORIZONTAL, 
            variable=self.blur_threshold_pct_var
        )
        threshold_pct_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        threshold_pct_value = ttk.Label(threshold_pct_frame, width=5)
        threshold_pct_value.pack(side=tk.RIGHT)
        
        # Update threshold percentage value label when slider changes
        def update_threshold_pct_label(*args):
            threshold_pct_value.config(text=f"{self.blur_threshold_pct_var.get():.1f}%")
        
        self.blur_threshold_pct_var.trace_add("write", update_threshold_pct_label)
        update_threshold_pct_label()  # Initial update
        
        # Calibration button
        calibration_frame = ttk.Frame(self.blur_settings_frame)
        calibration_frame.pack(fill=tk.X, pady=(5, 0))
        
        calibrate_button = ttk.Button(
            calibration_frame,
            text="Calibrate Blur Detection",
            command=self._show_blur_calibration_dialog
        )
        calibrate_button.pack(side=tk.LEFT, padx=(0, 10))
        
        help_button = ttk.Button(
            calibration_frame,
            text="?",
            width=2,
            command=self._show_blur_help
        )
        help_button.pack(side=tk.RIGHT)
        
        # Initialize the UI state
        self._toggle_blur_settings()
        
        # Progress bar
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                        orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        
        # Status text - enlarged and improved for better visibility
        status_frame = ttk.LabelFrame(main_frame, text="Detailed Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add a text widget with scrollbar
        self.status_text = tk.Text(status_frame, height=15, wrap=tk.WORD, font=("Consolas", 10))
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.status_text.config(yscrollcommand=scrollbar.set)
        self.status_text.config(state=tk.DISABLED)
        
        # Add text tags for different status types (error, warning, success)
        self.status_text.tag_configure("error", foreground="red")
        self.status_text.tag_configure("warning", foreground="orange")
        self.status_text.tag_configure("success", foreground="green")
        self.status_text.tag_configure("info", foreground="blue")
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.process_button = ttk.Button(button_frame, text="Process Photos", 
                                    command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        quit_button = ttk.Button(button_frame, text="Quit", command=self.quit_app)
        quit_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Generate Drillhole Trace
        self.trace_button = ttk.Button(
            control_frame,
            text="Generate Drillhole Trace",
            command=self.on_generate_trace
        )
        self.trace_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Set up a timer to check for progress updates
        self.root.after(100, self.check_progress)
        
        # Add initial status message
        self.update_status("Ready. Select a folder and click 'Process Photos'.", "info")

    def _toggle_blur_settings(self):
        """Enable or disable blur detection settings based on checkbox state."""
        if self.blur_enable_var.get():
            # Enable all blur settings
            for child in self.blur_settings_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Scale, ttk.Checkbutton, ttk.Button, ttk.Entry)):
                        widget.configure(state='normal')
        else:
            # Disable all blur settings
            for child in self.blur_settings_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Scale, ttk.Checkbutton, ttk.Button, ttk.Entry)):
                        widget.configure(state='disabled')

    def _show_blur_help(self):
        """Show help information about blur detection."""
        help_text = """
    Blur Detection Help

    The blur detector identifies blurry images using the Laplacian variance method:

    - Blur Threshold: Lower values make the detector more sensitive (detecting more images as blurry).
    Typical values range from 50-200 depending on image content.

    - ROI Ratio: Percentage of the central image to analyze. Use higher values for more complete analysis,
    lower values to focus on the center where subjects are typically sharpest.

    - Quality Alert Threshold: Percentage of blurry compartments that will trigger a quality alert.

    - Calibration: Use the calibration tool to automatically set an optimal threshold based on example
    sharp and blurry images.
    """
        messagebox.showinfo("Blur Detection Help", help_text)

    def _show_blur_calibration_dialog(self):
        """Show a dialog for calibrating blur detection."""
        # Create a dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Calibrate Blur Detection")
        dialog.geometry("600x500")
        dialog.grab_set()  # Make dialog modal
        
        # Main frame with padding
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Select example sharp and blurry images to calibrate the blur detection threshold.",
            wraplength=580,
            justify=tk.LEFT
        )
        instructions.pack(fill=tk.X, pady=(0, 10))
        
        # Frame for sharp images
        sharp_frame = ttk.LabelFrame(main_frame, text="Sharp (Good) Images", padding=10)
        sharp_frame.pack(fill=tk.X, pady=(0, 10))
        
        sharp_path_var = tk.StringVar()
        sharp_entry = ttk.Entry(sharp_frame, textvariable=sharp_path_var)
        sharp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        sharp_button = ttk.Button(
            sharp_frame,
            text="Browse",
            command=lambda: self._select_calibration_images(sharp_path_var)
        )
        sharp_button.pack(side=tk.RIGHT)
        
        # Frame for blurry images
        blurry_frame = ttk.LabelFrame(main_frame, text="Blurry (Poor) Images", padding=10)
        blurry_frame.pack(fill=tk.X, pady=(0, 10))
        
        blurry_path_var = tk.StringVar()
        blurry_entry = ttk.Entry(blurry_frame, textvariable=blurry_path_var)
        blurry_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        blurry_button = ttk.Button(
            blurry_frame,
            text="Browse",
            command=lambda: self._select_calibration_images(blurry_path_var)
        )
        blurry_button.pack(side=tk.RIGHT)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Calibration Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Results text widget
        results_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # Calibrate button
        def calibrate():
            # Get paths
            sharp_paths = sharp_path_var.get().split(';')
            blurry_paths = blurry_path_var.get().split(';')
            
            # Validate
            if not sharp_paths or not sharp_paths[0] or not blurry_paths or not blurry_paths[0]:
                messagebox.showerror("Error", "Please select both sharp and blurry images")
                return
            
            # Load images
            sharp_images = []
            blurry_images = []
            
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, "Loading images...\n")
            
            try:
                # Load sharp images
                for path in sharp_paths:
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            sharp_images.append(img)
                            results_text.insert(tk.END, f"Loaded sharp image: {os.path.basename(path)}\n")
                
                # Load blurry images
                for path in blurry_paths:
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            blurry_images.append(img)
                            results_text.insert(tk.END, f"Loaded blurry image: {os.path.basename(path)}\n")
                
                if not sharp_images or not blurry_images:
                    results_text.insert(tk.END, "Error: Failed to load images\n")
                    return
                
                # Calibrate threshold
                results_text.insert(tk.END, "\nCalibrating...\n")
                
                old_threshold = self.blur_detector.threshold
                new_threshold = self.blur_detector.calibrate_threshold(sharp_images, blurry_images)
                
                # Update UI
                self.blur_threshold_var.set(new_threshold)
                
                # Log results
                results_text.insert(tk.END, f"\nResults:\n")
                results_text.insert(tk.END, f"Old threshold: {old_threshold:.2f}\n")
                results_text.insert(tk.END, f"New threshold: {new_threshold:.2f}\n")
                
                # Show variances
                results_text.insert(tk.END, "\nImage Variances:\n")
                
                for i, img in enumerate(sharp_images):
                    variance = self.blur_detector.get_laplacian_variance(img)
                    results_text.insert(tk.END, f"Sharp image {i+1}: {variance:.2f}\n")
                
                for i, img in enumerate(blurry_images):
                    variance = self.blur_detector.get_laplacian_variance(img)
                    results_text.insert(tk.END, f"Blurry image {i+1}: {variance:.2f}\n")
                
                # Remind user to save settings
                results_text.insert(tk.END, "\nRemember to click 'OK' to apply the new threshold.\n")
                
            except Exception as e:
                results_text.insert(tk.END, f"Error during calibration: {str(e)}\n")
                logger.error(f"Calibration error: {str(e)}")
                logger.error(traceback.format_exc())
        
        calibrate_button = ttk.Button(
            button_frame,
            text="Calibrate",
            command=calibrate
        )
        calibrate_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # OK button
        ok_button = ttk.Button(
            button_frame,
            text="OK",
            command=dialog.destroy
        )
        ok_button.pack(side=tk.RIGHT)

    
    
    def _select_calibration_images(self, path_var):
        """
        Show file dialog to select images for calibration.
        
        Args:
            path_var: StringVar to store selected paths
        """
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            # Join paths with semicolons for display
            path_var.set(';'.join(file_paths))

        # Progress bar
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                        orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        
        # Status text - enlarged and improved for better visibility
        status_frame = ttk.LabelFrame(main_frame, text="Detailed Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add a text widget with scrollbar
        self.status_text = tk.Text(status_frame, height=15, wrap=tk.WORD, font=("Consolas", 10))
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.status_text.config(yscrollcommand=scrollbar.set)
        self.status_text.config(state=tk.DISABLED)
        
        # Add text tags for different status types (error, warning, success)
        self.status_text.tag_configure("error", foreground="red")
        self.status_text.tag_configure("warning", foreground="orange")
        self.status_text.tag_configure("success", foreground="green")
        self.status_text.tag_configure("info", foreground="blue")
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))

        
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Generate Drillhole Trace
        self.trace_button = ttk.Button(
            control_frame,
            text="Generate Drillhole Trace",
            command=self.on_generate_trace
        )
        self.trace_button.pack(side=tk.LEFT, padx=5, pady=5)

        
        self.process_button = ttk.Button(button_frame, text="Process Photos", 
                                    command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        quit_button = ttk.Button(button_frame, text="Quit", command=self.quit_app)
        quit_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        
        # Set up a timer to check for progress updates
        self.root.after(100, self.check_progress)
        
        # Add initial status message
        self.update_status("Ready. Select a folder and click 'Process Photos'.", "info")

    
    
    def update_status(self, message: str, status_type: str = None) -> None:
        """
        Add a message to the status text widget with optional formatting.
        
        Args:
            message: The message to display
            status_type: Optional status type for formatting (error, warning, success, info)
        """
        try:
            # Check if status_text exists
            if not hasattr(self, 'status_text') or self.status_text is None:
                # Just log the message
                logger.info(message)
                return
                
            self.status_text.config(state=tk.NORMAL)
            
            # Add timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_text.insert(tk.END, f"[{timestamp}] ", "info")
            
            # Add message with appropriate tag if specified
            if status_type and status_type in ["error", "warning", "success", "info"]:
                self.status_text.insert(tk.END, f"{message}\n", status_type)
            else:
                self.status_text.insert(tk.END, f"{message}\n")
            
            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)
        except Exception as e:
            # Log error but don't raise it to avoid UI crashes
            logger.error(f"Error updating status: {str(e)}")
    
    
    def save_compartments(self, 
                        compartments: List[np.ndarray], 
                        output_dir: str, 
                        base_filename: str,
                        metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Save extracted compartment images to disk.
        
        Args:
            compartments: List of compartment images
            output_dir: Directory to save compartment images
            base_filename: Base filename to use for compartment images
            metadata: Optional metadata for naming compartments
            
        Returns:
            Number of successfully saved compartments
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(base_filename))[0]
        
        # Create debug directory for blur visualizations if needed
        blur_viz_dir = None
        if self.config['enable_blur_detection'] and self.config['save_blur_visualizations']:
            blur_viz_dir = os.path.join(os.path.dirname(base_filename), 'blur_analysis')
            os.makedirs(blur_viz_dir, exist_ok=True)
        
        # Detect blur in compartments
        blur_results = self.detect_blur_in_compartments(compartments, blur_viz_dir, base_filename)
        
        # Add blur indicators to images if enabled
        if self.config['flag_blurry_images']:
            compartments = self.add_blur_indicators(compartments, blur_results)
        
        # Save each compartment
        saved_count = 0
        blurry_indices = set()
        
        for i, compartment in enumerate(compartments):
            try:
                # Generate output filename based on metadata if available
                if metadata and all(key in metadata for key in ['hole_id', 'depth_from', 'depth_to']) and self.config['use_metadata_for_filenames']:
                    # Calculate depth for this compartment based on total range divided by compartments
                    if metadata['depth_from'] is not None and metadata['depth_to'] is not None:
                        total_depth = float(metadata['depth_to']) - float(metadata['depth_from'])
                        compartment_depth = total_depth / len(compartments)
                        
                        # Calculate this compartment's depth range
                        comp_depth_from = float(metadata['depth_from']) + (i * compartment_depth)
                        comp_depth_to = comp_depth_from + compartment_depth
                        
                        # Format the filename according to the pattern
                        filename = self.config['metadata_filename_pattern'].format(
                            hole_id=metadata['hole_id'],
                            depth_from=f"{comp_depth_from:.1f}",
                            depth_to=f"{comp_depth_to:.1f}",
                            compartment_number=i+1
                        )
                        
                        # Add blur indicator to filename if blurry
                        is_blurry = next((r.get('is_blurry', False) for r in blur_results if r['index'] == i), False)
                        if is_blurry:
                            blurry_indices.add(i)
                            if self.config['flag_blurry_images']:
                                filename = f"{filename}_BLURRY"
                        
                        # Add extension
                        filename = f"{filename}.{self.config['output_format']}"
                    else:
                        # Fallback if depths not available
                        filename = f"{metadata['hole_id']}_compartment_{i+1:02d}.{self.config['output_format']}"
                else:
                    # Standard naming if metadata not available
                    filename = f"{base_name}_compartment_{i+1:02d}.{self.config['output_format']}"
                
                output_path = os.path.join(output_dir, filename)
                
                # Save with appropriate quality settings
                if self.config['output_format'].lower() == 'jpg':
                    cv2.imwrite(output_path, compartment, 
                            [cv2.IMWRITE_JPEG_QUALITY, self.config['jpeg_quality']])
                else:
                    cv2.imwrite(output_path, compartment)
                
                saved_count += 1
                logger.info(f"Saved compartment {i+1} to {filename}")
                
            except Exception as e:
                logger.error(f"Error saving compartment {i+1}: {str(e)}")
        
        # Log blur detection summary
        blurry_count = len(blurry_indices)
        if blurry_count > 0:
            logger.warning(f"Detected {blurry_count}/{len(compartments)} blurry compartments")
            
            # Calculate percentage of blurry compartments
            blurry_percentage = (blurry_count / len(compartments)) * 100
            threshold_percentage = self.config['blurry_threshold_percentage']
            
            if blurry_percentage >= threshold_percentage:
                logger.warning(f"QUALITY ALERT: {blurry_percentage:.1f}% of compartments are blurry (threshold: {threshold_percentage}%)")
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put((f"QUALITY ALERT: {blurry_percentage:.1f}% of compartments are blurry!", "warning"))
        
        return saved_count, blurry_indices

    def process_folder_with_progress(self, folder_path: str):
        """
        Process folder with progress updates for GUI.
        
        Args:
            folder_path: Path to folder containing images
        """
        try:
            # Update progress
            self.progress_queue.put(("Starting processing...", 0))
            
            # Process the folder
            successful, failed = self.process_folder(folder_path)
            
            # Update progress
            self.progress_queue.put((f"Processing complete: {successful} successful, {failed} failed", 100))
            
        except Exception as e:
            self.progress_queue.put((f"Error: {str(e)}", None))
            logger.error(f"Error processing folder: {str(e)}")
        finally:
            self.processing_complete = True

    def check_progress(self):
        """Check for progress updates from the processing thread with enhanced status handling."""
        while not self.progress_queue.empty():
            try:
                message, progress = self.progress_queue.get_nowait()
                
                # Update progress bar if progress value provided
                if progress is not None:
                    self.progress_var.set(progress)
                
                # Update status message with appropriate status type
                if message:
                    # Determine message type based on content
                    if any(error_term in message.lower() for error_term in ["error", "failed", "not enough"]):
                        self.update_status(message, "error")
                    elif any(warning_term in message.lower() for warning_term in ["warning", "missing"]):
                        self.update_status(message, "warning")
                    elif any(success_term in message.lower() for success_term in ["success", "complete", "saved"]):
                        self.update_status(message, "success")
                    else:
                        self.update_status(message, "info")
                        
                # If processing is complete, re-enable the process button
                if self.processing_complete:
                    self.process_button.config(state=tk.NORMAL)
                    
            except queue.Empty:
                pass
            
        # Schedule the next check
        self.root.after(100, self.check_progress)

    def create_visualization_image(self, 
                                    original_image: np.ndarray, 
                                    processing_images: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Create a collage showing the original image alongside processing steps.
        
        Args:
            original_image: The original input image
            processing_images: List of (step_name, image) tuples
            
        Returns:
            Visualization collage as numpy array
        """
        # Helper function to ensure 3-channel image
        def ensure_3channel(img: np.ndarray) -> np.ndarray:
            if len(img.shape) == 2:  # Grayscale
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3:
                if img.shape[2] == 1:  # Single channel
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 3:  # Already color
                    return img
                elif img.shape[2] == 4:  # RGBA
                    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Fallback - create a blank color image
            return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        # Determine the number of images in the collage
        num_images = 1 + len(processing_images)
        
        # Determine layout (1 row if ≤ 3 images, 2 rows otherwise)
        if num_images <= 3:
            rows, cols = 1, num_images
        else:
            rows = 2
            cols = (num_images + 1) // 2  # Ceiling division
        
        # Resize images to a standard height
        target_height = 600
        resized_images = []
        
        # Resize original image - ensure 3 channels
        original_image = ensure_3channel(original_image)
        h, w = original_image.shape[:2]
        scale = target_height / h
        resized_original = cv2.resize(original_image, (int(w * scale), target_height))
        resized_images.append(("Original Image", resized_original))
        
        # Resize processing images
        for name, img in processing_images:
            # Ensure 3 channels before resizing
            img = ensure_3channel(img)
            h, w = img.shape[:2]
            scale = target_height / h
            resized = cv2.resize(img, (int(w * scale), target_height))
            resized_images.append((name, resized))
        
        # Calculate canvas size
        max_width = max(img[1].shape[1] for img in resized_images)
        canvas_width = max_width * cols + 20 * (cols + 1)  # Add padding
        canvas_height = target_height * rows + 60 * (rows + 1)  # Add space for titles
        
        # Create canvas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Place images on canvas
        for i, (name, img) in enumerate(resized_images):
            row = i // cols
            col = i % cols
            
            # Calculate position
            x = 20 + col * (max_width + 20)
            y = 60 + row * (target_height + 60)
            
            # Place image
            h, w = img.shape[:2]
            canvas[y:y+h, x:x+w] = img
            
            # Add title
            cv2.putText(canvas, name, (x, y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Add main title
        cv2.putText(canvas, "Chip Tray Extraction Process", 
                (canvas_width // 2 - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        return canvas

    
    def browse_folder(self):
        """Open folder browser dialog and update the folder entry."""
        folder_path = filedialog.askdirectory(title="Select folder with chip tray photos")
        if folder_path:
            self.folder_var.set(folder_path)
            # Check if status_text exists before updating
            if hasattr(self, 'status_text') and self.status_text:
                self.update_status(f"Selected folder: {folder_path}")

    def start_processing(self):
        """Start processing in a separate thread."""
        folder_path = self.folder_var.get()
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder")
            return
        
        if not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Selected path is not a valid folder")
            return
        
        
        # Update config with current GUI settings
        self.config['output_folder'] = self.output_folder_var.get()
        self.config['output_format'] = self.format_var.get()
        self.config['jpeg_quality'] = self.quality_var.get()
        self.config['save_debug_images'] = self.debug_var.get()
        
        # Update blur detection settings
        self.config['enable_blur_detection'] = self.blur_enable_var.get()
        self.config['blur_threshold'] = self.blur_threshold_var.get()
        self.config['blur_roi_ratio'] = self.blur_roi_var.get()
        self.config['flag_blurry_images'] = self.flag_blurry_var.get()
        self.config['save_blur_visualizations'] = self.save_blur_viz_var.get()
        self.config['blurry_threshold_percentage'] = self.blur_threshold_pct_var.get()
        
        # Update blur detector with new settings
        self.blur_detector.threshold = self.config['blur_threshold']
        self.blur_detector.roi_ratio = self.config['blur_roi_ratio']
        
        # Clear status
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.config(state=tk.DISABLED)
        
        self.progress_var.set(0)
        self.processing_complete = False
        self.process_button.config(state=tk.DISABLED)
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=self.process_folder_with_progress, 
            args=(folder_path,)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        self.update_status(f"Started processing folder: {folder_path}")

    def quit_app(self):
        """Close the application."""
        if self.root:
            self.root.destroy()



class DrillholeTraceGenerator:
    """
    A class to generate drillhole trace images by stitching together chip tray compartment images.
    Integrates with ChipTrayExtractor to create complete drillhole visualization.
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None, 
                 progress_queue: Optional[Any] = None,
                 root: Optional[tk.Tk] = None):
        """
        Initialize the Drillhole Trace Generator.
        
        Args:
            config: Configuration dictionary
            progress_queue: Optional queue for reporting progress
            root: Optional Tkinter root for dialog windows
        """
        self.progress_queue = progress_queue
        self.root = root
        
        # Default configuration
        self.config = {
            'output_folder': 'drillhole_traces',
            'metadata_box_color': (200, 200, 200, 150),  # BGRA (light gray with transparency)
            'metadata_text_color': (0, 0, 0),  # BGR (black)
            'metadata_font_scale': 0.7,
            'metadata_font_thickness': 1,
            'metadata_font_face': cv2.FONT_HERSHEY_SIMPLEX,
            'metadata_box_padding': 10,
            'metadata_pattern': r'(.+)_CC_(\d+\.?\d*)-(\d+\.?\d*)m',  # Pattern to extract metadata from filename
            'max_width': 2000,  # Maximum width for output image
            'min_width': 800,   # Minimum width for output image
            'box_alpha': 0.7,    # Transparency of metadata box (0-1)
            'additional_columns': []  # Additional columns from CSV to include
        }
        
        # Update with provided config if any
        if config:
            self.config.update(config)
            
        # Logger
        self.logger = logging.getLogger(__name__)

    def get_csv_columns(self, csv_path: str) -> List[str]:
        """
        Get column names from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of column names
        """
        try:
            # Read just the header row for efficiency
            df = pd.read_csv(csv_path, nrows=0)
            columns = df.columns.tolist()
            
            # Check for required columns
            required_columns = ['holeid', 'from', 'to']
            missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in columns]]
            
            if missing_columns:
                msg = f"CSV missing required columns: {', '.join(missing_columns)}"
                self.logger.error(msg)
                if self.progress_queue:
                    self.progress_queue.put((msg, None))
                return []
            
            return columns
        except Exception as e:
            self.logger.error(f"Error reading CSV columns: {str(e)}")
            if self.progress_queue:
                self.progress_queue.put((f"Error reading CSV: {str(e)}", None))
            return []

    def select_csv_columns(self, columns: List[str]) -> List[str]:
        """
        Open a dialog to let the user select columns from a CSV.
        
        Args:
            columns: List of column names from the CSV
            
        Returns:
            List of selected column names
        """
        if not self.root:
            self.logger.warning("No Tkinter root available for column selection dialog")
            return []
            
        # Create a dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select CSV Columns")
        dialog.geometry("500x400")
        dialog.grab_set()  # Make dialog modal
        
        # Explanatory text
        header_label = ttk.Label(
            dialog, 
            text="Select additional columns to display in the metadata box (max 5):",
            wraplength=480,
            justify=tk.LEFT,
            padding=(10, 10)
        )
        header_label.pack(fill=tk.X)
        
        # Required columns notice
        required_label = ttk.Label(
            dialog, 
            text="Note: 'holeid', 'from', and 'to' are always included.",
            font=("Arial", 9, "italic"),
            foreground="gray",
            padding=(10, 0, 10, 10)
        )
        required_label.pack(fill=tk.X)
        
        # Create a frame for the column checkboxes
        column_frame = ttk.Frame(dialog, padding=10)
        column_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas with scrollbar for many columns
        canvas = tk.Canvas(column_frame)
        scrollbar = ttk.Scrollbar(column_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Dictionary to track which columns are selected
        selected_columns = {}
        
        # Exclude required columns from the selection list
        required_columns = ['holeid', 'from', 'to']
        selectable_columns = [col for col in columns if col.lower() not in [c.lower() for c in required_columns]]
        
        # Add checkboxes for each column
        for column in selectable_columns:
            var = tk.BooleanVar(value=False)
            selected_columns[column] = var
            
            checkbox = ttk.Checkbutton(
                scrollable_frame,
                text=column,
                variable=var,
                command=lambda: self._update_selection_count(selected_columns, selection_label)
            )
            checkbox.pack(anchor="w", padx=10, pady=5)
        
        # Label to show how many columns are selected
        selection_label = ttk.Label(
            dialog,
            text="0 columns selected (max 5)",
            padding=(10, 10)
        )
        selection_label.pack()
        
        # Buttons frame
        button_frame = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        button_frame.pack(fill=tk.X)
        
        # Result container
        result = []
        
        # OK button handler
        def on_ok():
            nonlocal result
            result = [col for col, var in selected_columns.items() if var.get()]
            if len(result) > 5:
                messagebox.showwarning("Too Many Columns", "Please select at most 5 columns.")
                return
            dialog.destroy()
        
        # Cancel button handler
        def on_cancel():
            nonlocal result
            result = []
            dialog.destroy()
        
        # Add buttons
        ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
        ok_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side=tk.RIGHT)
        
        # Wait for dialog to close
        dialog.wait_window()
        return result
        
    def _update_selection_count(self, selected_columns, label_widget):
        """Helper method to update the selection count label."""
        count = sum(var.get() for var in selected_columns.values())
        color = "red" if count > 5 else "black"
        label_widget.config(text=f"{count} columns selected (max 5)", foreground=color)

    def load_csv_data(self, csv_path: str, selected_columns: List[str]) -> pd.DataFrame:
        """
        Load CSV data for metadata integration.
        
        Args:
            csv_path: Path to CSV file
            selected_columns: Columns to include
            
        Returns:
            DataFrame with the CSV data
        """
        try:
            # Determine which columns to load - ensure required columns are included
            required_columns = ['holeid', 'from', 'to']
            columns_to_load = list(set(required_columns + selected_columns))
            
            # Load the CSV
            df = pd.read_csv(csv_path, usecols=columns_to_load)
            
            # Normalize column names to lowercase for case-insensitive matching
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure numeric columns are properly typed
            if 'from' in df.columns:
                df['from'] = pd.to_numeric(df['from'], errors='coerce')
            if 'to' in df.columns:
                df['to'] = pd.to_numeric(df['to'], errors='coerce')
                
            # Drop rows with missing required values
            df = df.dropna(subset=['holeid', 'from', 'to'])
            
            self.logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            if self.progress_queue:
                self.progress_queue.put((f"Error loading CSV data: {str(e)}", None))
            return pd.DataFrame()

    def parse_filename_metadata(self, filename: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Parse metadata from a filename based on configured pattern.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Tuple of (hole_id, depth_from, depth_to)
        """
        try:
            match = re.match(self.config['metadata_pattern'], filename)
            if match:
                hole_id = match.group(1)
                depth_from = float(match.group(2))
                depth_to = float(match.group(3))
                return hole_id, depth_from, depth_to
            
            # Try a more generic pattern if the configured one fails
            match = re.search(r'([A-Za-z]{2}\d{4}).*?(\d+\.?\d*)-(\d+\.?\d*)m', filename)
            if match:
                hole_id = match.group(1)
                depth_from = float(match.group(2))
                depth_to = float(match.group(3))
                return hole_id, depth_from, depth_to
                
            # No match found
            self.logger.warning(f"Could not parse metadata from filename: {filename}")
            return None, None, None
            
        except Exception as e:
            self.logger.error(f"Error parsing filename metadata: {str(e)}")
            return None, None, None

    def get_csv_data_for_interval(self, 
                                 df: pd.DataFrame, 
                                 hole_id: str, 
                                 depth_from: float, 
                                 depth_to: float) -> Dict[str, Any]:
        """
        Get relevant CSV data for a specific interval.
        
        Args:
            df: DataFrame with CSV data
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            
        Returns:
            Dictionary with column values for the interval
        """
        if df.empty:
            return {}
            
        try:
            # Normalize hole ID for matching
            hole_id_norm = hole_id.upper()
            
            # Filter for matching hole ID
            hole_data = df[df['holeid'].str.upper() == hole_id_norm]
            
            if hole_data.empty:
                return {}
                
            # Find intervals that overlap with our target interval
            matching_intervals = hole_data[
                ((hole_data['from'] <= depth_from) & (hole_data['to'] > depth_from)) |
                ((hole_data['from'] >= depth_from) & (hole_data['from'] < depth_to)) |
                ((hole_data['from'] <= depth_from) & (hole_data['to'] >= depth_to))
            ]
            
            if matching_intervals.empty:
                return {}
                
            # If multiple matches, take the one with the highest overlap
            if len(matching_intervals) > 1:
                # Calculate overlap for each interval
                def calculate_overlap(row):
                    overlap_start = max(row['from'], depth_from)
                    overlap_end = min(row['to'], depth_to)
                    return max(0, overlap_end - overlap_start)
                    
                matching_intervals['overlap'] = matching_intervals.apply(calculate_overlap, axis=1)
                best_match = matching_intervals.loc[matching_intervals['overlap'].idxmax()]
                
                # Convert the Series to a dictionary
                result = best_match.to_dict()
                
                # Clean up the result
                if 'overlap' in result:
                    del result['overlap']
                    
                return result
            else:
                # Just one match, convert to dictionary
                return matching_intervals.iloc[0].to_dict()
                
        except Exception as e:
            self.logger.error(f"Error getting CSV data for interval: {str(e)}")
            return {}

    def add_metadata_to_image(self, 
                             image: np.ndarray, 
                             hole_id: str, 
                             depth_from: float, 
                             depth_to: float,
                             csv_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Add metadata box to an image.
        
        Args:
            image: Input image
            hole_id: Hole ID
            depth_from: Starting depth
            depth_to: Ending depth
            csv_data: Optional CSV data to include
            
        Returns:
            Image with metadata box added
        """
        # Make a copy of the image to avoid modifying the original
        result = image.copy()
        
        # Ensure the image has an alpha channel for transparency
        if result.shape[2] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        
        # Prepare the metadata text
        text_lines = [
            f"Hole ID: {hole_id}",
            f"Depth: {depth_from:.1f}-{depth_to:.1f}m"
        ]
        
        # Add CSV data if available
        if csv_data and isinstance(csv_data, dict):
            # Skip holeid, from, and to as they're already included above
            skip_columns = ['holeid', 'from', 'to']
            for key, value in csv_data.items():
                if key.lower() not in skip_columns:
                    # Format numeric values appropriately
                    if isinstance(value, (int, float)):
                        if value == int(value):  # It's a whole number
                            formatted_value = f"{int(value)}"
                        else:
                            formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    text_lines.append(f"{key.capitalize()}: {formatted_value}")
        
        # Determine text size for box sizing
        font_scale = self.config['metadata_font_scale']
        font_thickness = self.config['metadata_font_thickness']
        font = self.config['metadata_font_face']
        padding = self.config['metadata_box_padding']
        
        # Calculate box size based on text
        max_width = 0
        total_height = 0
        
        for line in text_lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line, font, font_scale, font_thickness
            )
            max_width = max(max_width, text_width)
            total_height += text_height + 5  # 5 pixel spacing between lines
        
        # Add padding
        box_width = max_width + (padding * 2)
        box_height = total_height + (padding * 2)
        
        # Create a transparent overlay for the metadata box
        overlay = result.copy()
        
        # Draw the semi-transparent background box (with rounded corners if possible)
        box_color = self.config['metadata_box_color']
        alpha = self.config['box_alpha']
        
        # Create a solid rectangle
        cv2.rectangle(
            overlay, 
            (padding, padding), 
            (padding + box_width, padding + box_height), 
            box_color, 
            -1  # Filled rectangle
        )
        
        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        # Draw the text
        text_color = self.config['metadata_text_color']
        y_offset = padding + int(font_scale * 20)  # Start position for text
        
        for line in text_lines:
            cv2.putText(
                result, 
                line, 
                (padding + 5, y_offset), 
                font, 
                font_scale, 
                text_color, 
                font_thickness, 
                cv2.LINE_AA
            )
            # Update y position for next line
            (_, text_height), _ = cv2.getTextSize(
                line, font, font_scale, font_thickness
            )
            y_offset += text_height + 10
            
        return result

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate an image 90 degrees clockwise.
        
        Args:
            image: Input image
            
        Returns:
            Rotated image
        """
        # Rotate 90 degrees clockwise
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    def collect_compartment_images(self, compartment_dir: str) -> Dict[str, List[Tuple[str, float, float, str]]]:
        """
        Collect and organize compartment images by hole ID.
        
        Args:
            compartment_dir: Directory containing compartment images
            
        Returns:
            Dictionary mapping hole IDs to lists of (hole_id, depth_from, depth_to, file_path) tuples
        """
        # Dictionary to store compartment info by hole ID
        hole_compartments: Dict[str, List[Tuple[str, float, float, str]]] = {}
        
        # Valid image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        
        try:
            # Get all image files
            image_files = [f for f in os.listdir(compartment_dir) 
                        if os.path.isfile(os.path.join(compartment_dir, f)) and 
                        f.lower().endswith(valid_extensions)]
            
            if not image_files:
                self.logger.warning(f"No image files found in {compartment_dir}")
                return hole_compartments
                
            self.logger.info(f"Found {len(image_files)} image files to process")
            
            # Process each file
            for filename in image_files:
                # Parse metadata from filename
                hole_id, depth_from, depth_to = self.parse_filename_metadata(filename)
                
                if hole_id and depth_from is not None and depth_to is not None:
                    # Add to appropriate hole ID list
                    if hole_id not in hole_compartments:
                        hole_compartments[hole_id] = []
                        
                    file_path = os.path.join(compartment_dir, filename)
                    hole_compartments[hole_id].append((hole_id, depth_from, depth_to, file_path))
            
            # Sort each hole's compartments by depth
            for hole_id, compartments in hole_compartments.items():
                hole_compartments[hole_id] = sorted(compartments, key=lambda x: x[1])  # Sort by depth_from
                
            self.logger.info(f"Organized compartments for {len(hole_compartments)} holes")
            
            # Log some statistics
            for hole_id, compartments in hole_compartments.items():
                self.logger.info(f"Hole {hole_id}: {len(compartments)} compartments")
                
            return hole_compartments
            
        except Exception as e:
            self.logger.error(f"Error collecting compartment images: {str(e)}")
            if self.progress_queue:
                self.progress_queue.put((f"Error collecting images: {str(e)}", None))
            return {}

    def generate_drillhole_trace(self, 
                               hole_id: str, 
                               compartments: List[Tuple[str, float, float, str]],
                               csv_data: Optional[pd.DataFrame] = None,
                               output_dir: Optional[str] = None) -> Optional[str]:
        """
        Generate a drillhole trace image by stitching compartments.
        
        Args:
            hole_id: Hole ID
            compartments: List of (hole_id, depth_from, depth_to, file_path) tuples
            csv_data: Optional DataFrame with CSV data
            output_dir: Output directory (uses config if None)
            
        Returns:
            Path to the generated image file, or None if failed
        """
        if not compartments:
            self.logger.warning(f"No compartments provided for hole {hole_id}")
            return None
            
        try:
            # Determine output directory
            if output_dir is None:
                # Use the directory of the first compartment + config['output_folder']
                base_dir = os.path.dirname(compartments[0][3])
                output_dir = os.path.join(base_dir, self.config['output_folder'])
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Status update
            status_msg = f"Generating drillhole trace for {hole_id} with {len(compartments)} compartments"
            self.logger.info(status_msg)
            if self.progress_queue:
                self.progress_queue.put((status_msg, None))
            
            # Load and prepare images
            processed_images = []
            
            # Initialize min/max values
            min_width = float('inf')
            
            # First pass - load, rotate, and get dimensions
            for i, (hole_id, depth_from, depth_to, file_path) in enumerate(compartments):
                # Progress update
                if self.progress_queue:
                    progress = ((i + 1) / len(compartments)) * 100
                    self.progress_queue.put((f"Processing compartment {i+1}/{len(compartments)}", progress))
                
                try:
                    # Load image
                    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    
                    if image is None:
                        self.logger.warning(f"Failed to load image: {file_path}")
                        continue
                        
                    # Rotate image 90 degrees clockwise
                    rotated = self.rotate_image(image)
                    
                    # Track minimum width
                    min_width = min(min_width, rotated.shape[1])
                    
                    # Get CSV data for this interval if available
                    csv_interval_data = {}
                    if csv_data is not None and not csv_data.empty:
                        csv_interval_data = self.get_csv_data_for_interval(
                            csv_data, hole_id, depth_from, depth_to
                        )
                    
                    # Add to processed list
                    processed_images.append((rotated, depth_from, depth_to, csv_interval_data))
                    
                except Exception as e:
                    self.logger.error(f"Error processing compartment {file_path}: {str(e)}")
                    continue
            
            if not processed_images:
                self.logger.error(f"No valid images processed for hole {hole_id}")
                return None
                
            # Determine target width - between min and max configs, close to original
            target_width = max(min(min_width, self.config['max_width']), self.config['min_width'])
            
            # Second pass - resize and add metadata
            final_images = []
            
            for i, (image, depth_from, depth_to, csv_data) in enumerate(processed_images):
                # Resize to target width if needed
                if image.shape[1] != target_width:
                    scale_factor = target_width / image.shape[1]
                    new_height = int(image.shape[0] * scale_factor)
                    image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Add metadata box
                image_with_metadata = self.add_metadata_to_image(
                    image, hole_id, depth_from, depth_to, csv_data
                )
                
                final_images.append(image_with_metadata)
            
            # Stitch images vertically
            if final_images:
                # Determine if all images have alpha channel
                has_alpha = all(img.shape[2] == 4 for img in final_images)
                
                # Convert all to the same format if needed
                if not has_alpha:
                    # Convert any BGRA to BGR
                    for i in range(len(final_images)):
                        if final_images[i].shape[2] == 4:
                            final_images[i] = cv2.cvtColor(final_images[i], cv2.COLOR_BGRA2BGR)
                else:
                    # Convert any BGR to BGRA
                    for i in range(len(final_images)):
                        if final_images[i].shape[2] == 3:
                            final_images[i] = cv2.cvtColor(final_images[i], cv2.COLOR_BGR2BGRA)
                
                # Vertically concatenate all images
                trace_image = cv2.vconcat(final_images)
                
                # Save the result
                output_filename = f"{hole_id}_drillhole_trace.png" 
                output_path = os.path.join(output_dir, output_filename)
                
                # Use PNG format to preserve transparency if present
                cv2.imwrite(output_path, trace_image)
                
                success_msg = f"Successfully created drillhole trace for {hole_id} at {output_path}"
                self.logger.info(success_msg)
                if self.progress_queue:
                    self.progress_queue.put((success_msg, 100))
                
                return output_path
            else:
                self.logger.error(f"Failed to process any valid images for hole {hole_id}")
                return None
                
        except Exception as e:
            error_msg = f"Error generating drillhole trace for {hole_id}: {str(e)}"
            self.logger.error(error_msg)
            if self.progress_queue:
                self.progress_queue.put((error_msg, None))
            return None

    def process_all_drillholes(self, 
                              compartment_dir: str,
                              csv_path: Optional[str] = None,
                              selected_columns: Optional[List[str]] = None) -> List[str]:
        """
        Process all drillholes in a directory to create trace images.
        
        Args:
            compartment_dir: Directory containing compartment images
            csv_path: Optional path to CSV file with additional data
            selected_columns: Optional list of columns to include from CSV
            
        Returns:
            List of paths to generated trace images
        """
        # Load CSV data if provided
        csv_data = None
        if csv_path and os.path.exists(csv_path):
            if not selected_columns:
                selected_columns = []
                
            csv_data = self.load_csv_data(csv_path, selected_columns)
        
        # Collect compartment images organized by hole ID
        hole_compartments = self.collect_compartment_images(compartment_dir)
        
        if not hole_compartments:
            self.logger.warning("No valid compartment images found")
            if self.progress_queue:
                self.progress_queue.put(("No valid compartment images found", None))
            return []
        
        # Create output directory
        output_dir = os.path.join(compartment_dir, self.config['output_folder'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each hole
        generated_traces = []
        
        for i, (hole_id, compartments) in enumerate(hole_compartments.items()):
            # Update progress
            if self.progress_queue:
                progress = ((i + 1) / len(hole_compartments)) * 100
                self.progress_queue.put((f"Processing hole {i+1}/{len(hole_compartments)}: {hole_id}", progress))
            
            # Generate trace for this hole
            trace_path = self.generate_drillhole_trace(
                hole_id, compartments, csv_data, output_dir
            )
            
            if trace_path:
                generated_traces.append(trace_path)
        
        # Final status update
        status_msg = f"Completed drillhole trace generation: {len(generated_traces)}/{len(hole_compartments)} successful"
        self.logger.info(status_msg)
        if self.progress_queue:
            self.progress_queue.put((status_msg, 100))
        
        return generated_traces

class BlurDetector:
    """
    A class to detect blurry images using the Laplacian variance method.
    
    The Laplacian operator is used to measure the second derivative of an image.
    The variance of the Laplacian is a simple measure of the amount of edges 
    present in an image - blurry images tend to have fewer edges and thus lower variance.
    """
    
    def __init__(self, threshold: float = 100.0, roi_ratio: float = 0.8):
        """
        Initialize the blur detector with configurable parameters.
        
        Args:
            threshold: Laplacian variance threshold below which an image is considered blurry
            roi_ratio: Ratio of central image area to use for blur detection (0.0-1.0)
        """
        self.threshold = threshold
        self.roi_ratio = max(0.1, min(1.0, roi_ratio))  # Clamp between 0.1 and 1.0
        self.logger = logging.getLogger(__name__)
    
    def get_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculate the variance of the Laplacian for an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Variance of the Laplacian as a measure of blurriness (lower = more blurry)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Extract region of interest if ratio < 1.0
            if self.roi_ratio < 1.0:
                h, w = gray.shape
                center_h, center_w = h // 2, w // 2
                roi_h, roi_w = int(h * self.roi_ratio), int(w * self.roi_ratio)
                start_h, start_w = center_h - (roi_h // 2), center_w - (roi_w // 2)
                
                # Ensure ROI is within image bounds
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h = min(h, start_h + roi_h)
                end_w = min(w, start_w + roi_w)
                
                # Extract ROI
                gray = gray[start_h:end_h, start_w:end_w]
            
            # Calculate Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate variance
            variance = laplacian.var()
            return variance
            
        except Exception as e:
            self.logger.error(f"Error calculating Laplacian variance: {str(e)}")
            return 0.0
    
    def is_blurry(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if an image is blurry.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (is_blurry, variance_score)
        """
        variance = self.get_laplacian_variance(image)
        return variance < self.threshold, variance
    
    def analyze_image_with_visualization(self, image: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Analyze an image and create a visualization of the blur detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (is_blurry, variance_score, visualization_image)
        """
        # Make a copy for visualization
        viz_image = image.copy()
        
        # Calculate blur metrics
        is_blurry, variance = self.is_blurry(image)
        
        # Add text with blur metrics
        status = "BLURRY" if is_blurry else "SHARP"
        color = (0, 0, 255) if is_blurry else (0, 255, 0)  # Red for blurry, green for sharp
        
        # Add a background box for better text visibility
        h, w = viz_image.shape[:2]
        cv2.rectangle(viz_image, (10, h - 60), (w - 10, h - 10), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(
            viz_image,
            f"Status: {status}", 
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )
        
        cv2.putText(
            viz_image,
            f"Laplacian Variance: {variance:.2f} (threshold: {self.threshold:.2f})", 
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        return is_blurry, variance, viz_image
    
    def batch_analyze_images(self, 
                           images: List[np.ndarray],
                           generate_visualizations: bool = False) -> List[Dict[str, Any]]:
        """
        Analyze a batch of images for blurriness.
        
        Args:
            images: List of input images
            generate_visualizations: Whether to create visualization images
            
        Returns:
            List of dictionaries with analysis results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                # Perform analysis
                if generate_visualizations:
                    is_blurry, variance, viz_image = self.analyze_image_with_visualization(image)
                    result = {
                        'index': i,
                        'is_blurry': is_blurry,
                        'variance': variance,
                        'visualization': viz_image
                    }
                else:
                    is_blurry, variance = self.is_blurry(image)
                    result = {
                        'index': i,
                        'is_blurry': is_blurry,
                        'variance': variance
                    }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing image {i}: {str(e)}")
                # Add a placeholder result for the failed image
                results.append({
                    'index': i,
                    'is_blurry': False,
                    'variance': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def calibrate_threshold(self, 
                           sharp_images: List[np.ndarray], 
                           blurry_images: List[np.ndarray],
                           safety_factor: float = 1.5) -> float:
        """
        Calibrate the blur threshold based on example images.
        
        Args:
            sharp_images: List of sharp (good quality) images
            blurry_images: List of blurry (poor quality) images
            safety_factor: Factor to apply to the calculated threshold
            
        Returns:
            Calibrated threshold value
        """
        if not sharp_images or not blurry_images:
            self.logger.warning("Not enough sample images for calibration, using default threshold")
            return self.threshold
        
        try:
            # Calculate variances for sharp images
            sharp_variances = [self.get_laplacian_variance(img) for img in sharp_images]
            
            # Calculate variances for blurry images
            blurry_variances = [self.get_laplacian_variance(img) for img in blurry_images]
            
            # Find the minimum variance of sharp images
            min_sharp = min(sharp_variances)
            
            # Find the maximum variance of blurry images
            max_blurry = max(blurry_variances)
            
            # Calculate the threshold
            if min_sharp > max_blurry:
                # Clear separation - use the midpoint
                threshold = (min_sharp + max_blurry) / 2
            else:
                # Overlap - use a weighted average
                threshold = (min_sharp * 0.7 + max_blurry * 0.3)
            
            # Apply safety factor (lower the threshold to err on the side of detecting blurry images)
            threshold = threshold / safety_factor
            
            self.logger.info(f"Calibrated blur threshold: {threshold:.2f}")
            
            # Update the instance threshold
            self.threshold = threshold
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Error during threshold calibration: {str(e)}")
            return self.threshold


def main():
    """Run the application."""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Chip Tray Extractor')
    parser.add_argument('--input', '-i', help='Input folder containing chip tray photos')
    parser.add_argument('--output', '-o', help='Output folder for extracted compartments')
    parser.add_argument('--no-gui', action='store_true', help='Run in command-line mode without GUI')
    parser.add_argument('--format', choices=['jpg', 'png', 'tiff'], default='jpg', 
                        help='Output image format')
    parser.add_argument('--quality', type=int, default=100, help='JPEG quality (0-100)')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    
    args = parser.parse_args()
    
    # Create the extractor
    extractor = ChipTrayExtractor()
    
    # Update config based on command-line arguments
    if args.output:
        extractor.config['output_folder'] = args.output
    if args.format:
        extractor.config['output_format'] = args.format
    if args.quality:
        extractor.config['jpeg_quality'] = args.quality
    if args.debug:
        extractor.config['save_debug_images'] = True
    
    # Run in appropriate mode
    if args.no_gui or args.input:
        # Command-line mode
        if args.input:
            successful, failed = extractor.process_folder(args.input)
            logger.info(f"Processing complete: {successful} successful, {failed} failed")
        else:
            logger.error("No input folder specified. Use --input to specify a folder to process.")
            return 1
    else:
        # GUI mode
        extractor.create_gui()
        extractor.root.mainloop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())