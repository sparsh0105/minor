"""
Road Detection Module.

This module provides lane detection and road boundary identification
using computer vision techniques with OpenCV.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings


@dataclass
class LaneInfo:
    """Information about detected lanes."""
    
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    lane_center: Optional[Tuple[int, int]] = None
    lane_departure_warning: bool = False
    lane_confidence: float = 0.0
    lane_angle: float = 0.0


@dataclass
class RoadInfo:
    """Information about detected road conditions."""
    
    lanes: LaneInfo
    road_width: float = 0.0
    road_quality: str = "unknown"
    curvature: float = 0.0
    visibility: float = 0.0


class RoadDetector:
    """
    Road detection system for lane detection and road boundary identification.
    
    This class implements lane detection using computer vision techniques
    including edge detection, Hough line transforms, and lane fitting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the road detector.
        
        Args:
            config: Optional configuration dictionary to override default settings.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Lane detection parameters
        self.roi_ratio = self.config.get(
            "lane_detection_roi_ratio", 
            settings.detection.lane_detection_roi_ratio
        )
        self.min_line_length = self.config.get(
            "lane_detection_min_line_length",
            settings.detection.lane_detection_min_line_length
        )
        self.max_line_gap = self.config.get(
            "lane_detection_max_line_gap",
            settings.detection.lane_detection_max_line_gap
        )
        self.rho = self.config.get(
            "lane_detection_rho",
            settings.detection.lane_detection_rho
        )
        self.theta = self.config.get(
            "lane_detection_theta",
            settings.detection.lane_detection_theta
        )
        self.threshold = self.config.get(
            "lane_detection_threshold",
            settings.detection.lane_detection_threshold
        )
        
        self.logger.info("Road detector initialized successfully")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for lane detection.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Preprocessed image ready for lane detection.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            return edges
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise
    
    def create_roi_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create region of interest mask for lane detection.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            ROI mask as numpy array.
        """
        try:
            height, width = image.shape[:2]
            
            # Define ROI vertices (trapezoid shape)
            roi_vertices = np.array([
                [
                    (0, height),
                    (width // 2 - width * 0.1, height * self.roi_ratio),
                    (width // 2 + width * 0.1, height * self.roi_ratio),
                    (width, height)
                ]
            ], dtype=np.int32)
            
            # Create mask
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, roi_vertices, 255)
            
            # Apply mask
            masked_image = cv2.bitwise_and(image, mask)
            
            return masked_image
        except Exception as e:
            self.logger.error(f"Error creating ROI mask: {e}")
            raise
    
    def detect_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect lines in the image using Hough line transform.
        
        Args:
            image: Preprocessed image with edges.
            
        Returns:
            List of detected lines.
        """
        try:
            lines = cv2.HoughLinesP(
                image,
                rho=self.rho,
                theta=self.theta * np.pi / 180,
                threshold=self.threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            return lines if lines is not None else []
        except Exception as e:
            self.logger.error(f"Error detecting lines: {e}")
            return []
    
    def separate_lanes(self, lines: List[np.ndarray], image_shape: Tuple[int, int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Separate detected lines into left and right lanes.
        
        Args:
            lines: List of detected lines.
            image_shape: Shape of the input image (height, width).
            
        Returns:
            Tuple of (left_lanes, right_lanes).
        """
        try:
            height, width = image_shape[:2]
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                else:
                    continue
                
                # Filter lines based on slope and position
                if slope < -0.5:  # Left lane (negative slope)
                    left_lines.append(line[0])
                elif slope > 0.5:  # Right lane (positive slope)
                    right_lines.append(line[0])
            
            return left_lines, right_lines
        except Exception as e:
            self.logger.error(f"Error separating lanes: {e}")
            return [], []
    
    def fit_lane_line(self, lines: List[np.ndarray], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Fit a single lane line from multiple line segments.
        
        Args:
            lines: List of line segments for one lane.
            image_shape: Shape of the input image (height, width).
            
        Returns:
            Fitted lane line as numpy array or None if no valid lines.
        """
        try:
            if not lines:
                return None
            
            height, width = image_shape[:2]
            
            # Extract all points
            all_x = []
            all_y = []
            
            for line in lines:
                x1, y1, x2, y2 = line
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])
            
            if len(all_x) < 2:
                return None
            
            # Fit polynomial (linear for simplicity)
            coeffs = np.polyfit(all_y, all_x, 1)
            
            # Generate points for the fitted line
            y1 = height
            y2 = int(height * self.roi_ratio)
            x1 = int(coeffs[0] * y1 + coeffs[1])
            x2 = int(coeffs[0] * y2 + coeffs[1])
            
            return np.array([x1, y1, x2, y2])
        except Exception as e:
            self.logger.error(f"Error fitting lane line: {e}")
            return None
    
    def calculate_lane_center(self, left_lane: Optional[np.ndarray], right_lane: Optional[np.ndarray], image_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Calculate the center point between two lanes.
        
        Args:
            left_lane: Left lane line coordinates.
            right_lane: Right lane line coordinates.
            image_shape: Shape of the input image (height, width).
            
        Returns:
            Center point coordinates or None if lanes are not available.
        """
        try:
            if left_lane is None or right_lane is None:
                return None
            
            height, width = image_shape[:2]
            y = int(height * 0.8)  # Bottom 80% of the image
            
            # Calculate x coordinates for both lanes at the specified y
            left_x = int((y - left_lane[1]) * (left_lane[2] - left_lane[0]) / (left_lane[3] - left_lane[1]) + left_lane[0])
            right_x = int((y - right_lane[1]) * (right_lane[2] - right_lane[0]) / (right_lane[3] - right_lane[1]) + right_lane[0])
            
            center_x = (left_x + right_x) // 2
            return (center_x, y)
        except Exception as e:
            self.logger.error(f"Error calculating lane center: {e}")
            return None
    
    def detect_lane_departure(self, lane_center: Optional[Tuple[int, int]], image_shape: Tuple[int, int], threshold: float = 0.1) -> bool:
        """
        Detect if the vehicle is departing from the lane.
        
        Args:
            lane_center: Center point of the lane.
            image_shape: Shape of the input image (height, width).
            threshold: Threshold for departure detection (0.0 to 1.0).
            
        Returns:
            True if lane departure is detected, False otherwise.
        """
        try:
            if lane_center is None:
                return False
            
            height, width = image_shape[:2]
            image_center_x = width // 2
            
            # Calculate deviation from center
            deviation = abs(lane_center[0] - image_center_x) / (width // 2)
            
            return deviation > threshold
        except Exception as e:
            self.logger.error(f"Error detecting lane departure: {e}")
            return False
    
    def calculate_lane_angle(self, lane: Optional[np.ndarray]) -> float:
        """
        Calculate the angle of a lane line.
        
        Args:
            lane: Lane line coordinates.
            
        Returns:
            Angle in degrees.
        """
        try:
            if lane is None:
                return 0.0
            
            x1, y1, x2, y2 = lane
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            return angle
        except Exception as e:
            self.logger.error(f"Error calculating lane angle: {e}")
            return 0.0
    
    def detect_lanes(self, image: np.ndarray) -> LaneInfo:
        """
        Detect lanes in the input image.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            LaneInfo object containing lane detection results.
        """
        try:
            # Preprocess image
            edges = self.preprocess_image(image)
            
            # Create ROI mask
            masked_edges = self.create_roi_mask(edges)
            
            # Detect lines
            lines = self.detect_lines(masked_edges)
            
            # Separate into left and right lanes
            left_lines, right_lines = self.separate_lanes(lines, image.shape)
            
            # Fit lane lines
            left_lane = self.fit_lane_line(left_lines, image.shape)
            right_lane = self.fit_lane_line(right_lines, image.shape)
            
            # Calculate lane center
            lane_center = self.calculate_lane_center(left_lane, right_lane, image.shape)
            
            # Detect lane departure
            lane_departure = self.detect_lane_departure(lane_center, image.shape)
            
            # Calculate lane angle (average of both lanes)
            left_angle = self.calculate_lane_angle(left_lane)
            right_angle = self.calculate_lane_angle(right_lane)
            lane_angle = (left_angle + right_angle) / 2 if left_lane is not None and right_lane is not None else 0.0
            
            # Calculate confidence based on number of detected lines
            total_lines = len(left_lines) + len(right_lines)
            confidence = min(total_lines / 10.0, 1.0)  # Normalize to 0-1
            
            return LaneInfo(
                left_lane=left_lane,
                right_lane=right_lane,
                lane_center=lane_center,
                lane_departure_warning=lane_departure,
                lane_confidence=confidence,
                lane_angle=lane_angle
            )
        except Exception as e:
            self.logger.error(f"Error detecting lanes: {e}")
            return LaneInfo()
    
    def detect_road_conditions(self, image: np.ndarray) -> RoadInfo:
        """
        Detect road conditions including lanes and road quality.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            RoadInfo object containing road condition analysis.
        """
        try:
            # Detect lanes
            lane_info = self.detect_lanes(image)
            
            # Calculate road width
            road_width = 0.0
            if lane_info.left_lane is not None and lane_info.right_lane is not None:
                # Calculate distance between lanes at bottom of image
                height = image.shape[0]
                y = int(height * 0.9)
                
                left_x = int((y - lane_info.left_lane[1]) * (lane_info.left_lane[2] - lane_info.left_lane[0]) / (lane_info.left_lane[3] - lane_info.left_lane[1]) + lane_info.left_lane[0])
                right_x = int((y - lane_info.right_lane[1]) * (lane_info.right_lane[2] - lane_info.right_lane[0]) / (lane_info.right_lane[3] - lane_info.right_lane[1]) + lane_info.right_lane[0])
                
                road_width = abs(right_x - left_x)
            
            # Assess road quality based on lane detection confidence
            if lane_info.lane_confidence > 0.7:
                road_quality = "good"
            elif lane_info.lane_confidence > 0.4:
                road_quality = "fair"
            else:
                road_quality = "poor"
            
            # Calculate curvature (simplified)
            curvature = abs(lane_info.lane_angle) / 90.0  # Normalize to 0-1
            
            # Calculate visibility (based on edge detection quality)
            edges = self.preprocess_image(image)
            visibility = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return RoadInfo(
                lanes=lane_info,
                road_width=road_width,
                road_quality=road_quality,
                curvature=curvature,
                visibility=visibility
            )
        except Exception as e:
            self.logger.error(f"Error detecting road conditions: {e}")
            return RoadInfo(lanes=LaneInfo())
    
    def draw_lanes(self, image: np.ndarray, lane_info: LaneInfo) -> np.ndarray:
        """
        Draw detected lanes on the image.
        
        Args:
            image: Input image as numpy array.
            lane_info: Lane information to draw.
            
        Returns:
            Image with drawn lanes.
        """
        try:
            result_image = image.copy()
            
            # Draw left lane
            if lane_info.left_lane is not None:
                x1, y1, x2, y2 = lane_info.left_lane
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw right lane
            if lane_info.right_lane is not None:
                x1, y1, x2, y2 = lane_info.right_lane
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw lane center
            if lane_info.lane_center is not None:
                center_x, center_y = lane_info.lane_center
                cv2.circle(result_image, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Draw lane departure warning
            if lane_info.lane_departure_warning:
                cv2.putText(result_image, "LANE DEPARTURE WARNING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return result_image
        except Exception as e:
            self.logger.error(f"Error drawing lanes: {e}")
            return image
