"""
Image processing utilities for the Intelligent Traffic System.

This module provides common image processing functions used across
different detection modules.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Union
import logging


class ImageProcessor:
    """
    Image processing utility class with common operations.
    
    This class provides standardized image processing functions
    used across different detection modules.
    """
    
    def __init__(self) -> None:
        """Initialize the image processor."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image as numpy array.
            target_size: Target size as (width, height).
            maintain_aspect_ratio: Whether to maintain aspect ratio.
            
        Returns:
            Resized image.
        """
        try:
            if maintain_aspect_ratio:
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # Calculate scaling factor
                scale = min(target_w / w, target_h / h)
                
                # Calculate new dimensions
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize image
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create canvas with target size
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # Center the resized image
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return canvas
            else:
                return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Enhance image contrast using linear transformation.
        
        Args:
            image: Input image as numpy array.
            alpha: Contrast control (1.0 = no change, >1.0 = more contrast).
            beta: Brightness control (0 = no change).
            
        Returns:
            Enhanced image.
        """
        try:
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        except Exception as e:
            logging.error(f"Error enhancing contrast: {e}")
            return image
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                           sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image as numpy array.
            kernel_size: Kernel size as (width, height).
            sigma: Standard deviation for Gaussian kernel.
            
        Returns:
            Blurred image.
        """
        try:
            return cv2.GaussianBlur(image, kernel_size, sigma)
        except Exception as e:
            logging.error(f"Error applying Gaussian blur: {e}")
            return image
    
    @staticmethod
    def apply_morphological_operations(image: np.ndarray, operation: str = 'close',
                                     kernel_size: Tuple[int, int] = (3, 3),
                                     iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations to image.
        
        Args:
            image: Input binary image as numpy array.
            operation: Morphological operation ('open', 'close', 'erode', 'dilate').
            kernel_size: Kernel size as (width, height).
            iterations: Number of iterations.
            
        Returns:
            Processed image.
        """
        try:
            kernel = np.ones(kernel_size, np.uint8)
            
            if operation == 'open':
                return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif operation == 'close':
                return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif operation == 'erode':
                return cv2.erode(image, kernel, iterations=iterations)
            elif operation == 'dilate':
                return cv2.dilate(image, kernel, iterations=iterations)
            else:
                return image
        except Exception as e:
            logging.error(f"Error applying morphological operations: {e}")
            return image
    
    @staticmethod
    def create_roi_mask(image: np.ndarray, vertices: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create region of interest mask.
        
        Args:
            image: Input image as numpy array.
            vertices: List of vertices defining the ROI polygon.
            
        Returns:
            ROI mask as numpy array.
        """
        try:
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 255)
            return mask
        except Exception as e:
            logging.error(f"Error creating ROI mask: {e}")
            return np.zeros_like(image)
    
    @staticmethod
    def apply_roi_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply ROI mask to image.
        
        Args:
            image: Input image as numpy array.
            mask: ROI mask as numpy array.
            
        Returns:
            Masked image.
        """
        try:
            return cv2.bitwise_and(image, mask)
        except Exception as e:
            logging.error(f"Error applying ROI mask: {e}")
            return image
    
    @staticmethod
    def detect_edges(image: np.ndarray, low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detection.
        
        Args:
            image: Input grayscale image as numpy array.
            low_threshold: Lower threshold for edge detection.
            high_threshold: Upper threshold for edge detection.
            
        Returns:
            Edge-detected image.
        """
        try:
            return cv2.Canny(image, low_threshold, high_threshold)
        except Exception as e:
            logging.error(f"Error detecting edges: {e}")
            return np.zeros_like(image)
    
    @staticmethod
    def find_contours(image: np.ndarray, mode: int = cv2.RETR_EXTERNAL,
                     method: int = cv2.CHAIN_APPROX_SIMPLE) -> List[np.ndarray]:
        """
        Find contours in binary image.
        
        Args:
            image: Input binary image as numpy array.
            mode: Contour retrieval mode.
            method: Contour approximation method.
            
        Returns:
            List of contours.
        """
        try:
            contours, _ = cv2.findContours(image, mode, method)
            return contours
        except Exception as e:
            logging.error(f"Error finding contours: {e}")
            return []
    
    @staticmethod
    def filter_contours_by_area(contours: List[np.ndarray], min_area: int = 100,
                               max_area: int = 10000) -> List[np.ndarray]:
        """
        Filter contours by area.
        
        Args:
            contours: List of contours.
            min_area: Minimum contour area.
            max_area: Maximum contour area.
            
        Returns:
            Filtered list of contours.
        """
        try:
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    filtered_contours.append(contour)
            return filtered_contours
        except Exception as e:
            logging.error(f"Error filtering contours: {e}")
            return []
    
    @staticmethod
    def calculate_contour_properties(contour: np.ndarray) -> dict:
        """
        Calculate various properties of a contour.
        
        Args:
            contour: Input contour.
            
        Returns:
            Dictionary containing contour properties.
        """
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            return {
                'area': area,
                'perimeter': perimeter,
                'bounding_rect': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'solidity': solidity
            }
        except Exception as e:
            logging.error(f"Error calculating contour properties: {e}")
            return {}
    
    @staticmethod
    def draw_contours(image: np.ndarray, contours: List[np.ndarray],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
        """
        Draw contours on image.
        
        Args:
            image: Input image as numpy array.
            contours: List of contours to draw.
            color: Color for drawing contours.
            thickness: Thickness of contour lines.
            
        Returns:
            Image with drawn contours.
        """
        try:
            result_image = image.copy()
            cv2.drawContours(result_image, contours, -1, color, thickness)
            return result_image
        except Exception as e:
            logging.error(f"Error drawing contours: {e}")
            return image
