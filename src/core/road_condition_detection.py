"""
Road Condition Detection Module.

This module provides road condition detection including pothole detection,
crack detection, and surface quality assessment using computer vision techniques.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import settings


@dataclass
class Pothole:
    """Information about a detected pothole."""
    
    center: Tuple[int, int]
    area: int
    perimeter: float
    depth_estimate: float
    severity: str  # 'low', 'medium', 'high'
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Crack:
    """Information about a detected crack."""
    
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    length: float
    width: float
    severity: str  # 'low', 'medium', 'high'
    type: str  # 'linear', 'network', 'alligator'


@dataclass
class SurfaceDefect:
    """Information about surface defects."""
    
    defect_type: str  # 'pothole', 'crack', 'patch', 'wear'
    center: Tuple[int, int]
    area: int
    severity: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class RoadConditionResult:
    """Complete road condition analysis result."""
    
    potholes: List[Pothole]
    cracks: List[Crack]
    surface_defects: List[SurfaceDefect]
    overall_quality: str  # 'excellent', 'good', 'fair', 'poor', 'critical'
    quality_score: float  # 0.0 to 1.0
    maintenance_priority: str  # 'low', 'medium', 'high', 'urgent'
    processing_time: float


class RoadConditionDetector:
    """
    Road condition detection system for identifying potholes, cracks, and surface defects.
    
    This class uses computer vision techniques including edge detection, contour analysis,
    and morphological operations to detect various road surface defects.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the road condition detector.
        
        Args:
            config: Optional configuration dictionary to override default settings.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Detection parameters
        self.pothole_min_area = self.config.get(
            "pothole_min_area",
            settings.detection.pothole_min_area
        )
        self.crack_min_length = self.config.get(
            "crack_min_length",
            settings.detection.crack_min_length
        )
        self.surface_quality_threshold = self.config.get(
            "surface_quality_threshold",
            settings.detection.surface_quality_threshold
        )
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3,
            'critical': 0.0
        }
        
        self.logger.info("Road condition detector initialized successfully")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for road condition detection.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Preprocessed grayscale image.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply histogram equalization to improve contrast
            equalized = cv2.equalizeHist(blurred)
            
            return equalized
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise
    
    def detect_potholes(self, image: np.ndarray) -> List[Pothole]:
        """
        Detect potholes in the road surface.
        
        Args:
            image: Preprocessed grayscale image.
            
        Returns:
            List of detected potholes.
        """
        try:
            # Apply adaptive thresholding to highlight dark regions (potholes)
            adaptive_thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            potholes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area >= self.pothole_min_area:
                    # Calculate contour properties
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center
                    center = (x + w // 2, y + h // 2)
                    
                    # Estimate depth based on area and perimeter
                    depth_estimate = self._estimate_pothole_depth(area, perimeter)
                    
                    # Determine severity
                    severity = self._determine_pothole_severity(area, depth_estimate)
                    
                    pothole = Pothole(
                        center=center,
                        area=int(area),
                        perimeter=perimeter,
                        depth_estimate=depth_estimate,
                        severity=severity,
                        bbox=(x, y, x + w, y + h)
                    )
                    potholes.append(pothole)
            
            return potholes
        except Exception as e:
            self.logger.error(f"Error detecting potholes: {e}")
            return []
    
    def detect_cracks(self, image: np.ndarray) -> List[Crack]:
        """
        Detect cracks in the road surface.
        
        Args:
            image: Preprocessed grayscale image.
            
        Returns:
            List of detected cracks.
        """
        try:
            # Apply edge detection using Canny
            edges = cv2.Canny(image, 50, 150)
            
            # Apply morphological operations to connect crack segments
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cracks = []
            for contour in contours:
                # Calculate contour length
                perimeter = cv2.arcLength(contour, False)
                
                # Filter by minimum length
                if perimeter >= self.crack_min_length:
                    # Approximate contour to get start and end points
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, False)
                    
                    if len(approx) >= 2:
                        # Get start and end points
                        start_point = tuple(approx[0][0])
                        end_point = tuple(approx[-1][0])
                        
                        # Calculate crack properties
                        length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
                        width = self._estimate_crack_width(contour)
                        
                        # Determine crack type
                        crack_type = self._classify_crack_type(contour, length)
                        
                        # Determine severity
                        severity = self._determine_crack_severity(length, width)
                        
                        crack = Crack(
                            start_point=start_point,
                            end_point=end_point,
                            length=length,
                            width=width,
                            severity=severity,
                            type=crack_type
                        )
                        cracks.append(crack)
            
            return cracks
        except Exception as e:
            self.logger.error(f"Error detecting cracks: {e}")
            return []
    
    def detect_surface_defects(self, image: np.ndarray) -> List[SurfaceDefect]:
        """
        Detect general surface defects and irregularities.
        
        Args:
            image: Preprocessed grayscale image.
            
        Returns:
            List of detected surface defects.
        """
        try:
            # Apply Laplacian filter to detect edges and irregularities
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Apply threshold to get defect regions
            _, thresh = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            defects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area >= 100:  # Minimum area for surface defects
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center
                    center = (x + w // 2, y + h // 2)
                    
                    # Classify defect type
                    defect_type = self._classify_surface_defect(contour, area)
                    
                    # Determine severity
                    severity = self._determine_defect_severity(area, defect_type)
                    
                    # Calculate confidence based on area and shape
                    confidence = min(area / 1000.0, 1.0)
                    
                    defect = SurfaceDefect(
                        defect_type=defect_type,
                        center=center,
                        area=int(area),
                        severity=severity,
                        confidence=confidence,
                        bbox=(x, y, x + w, y + h)
                    )
                    defects.append(defect)
            
            return defects
        except Exception as e:
            self.logger.error(f"Error detecting surface defects: {e}")
            return []
    
    def _estimate_pothole_depth(self, area: float, perimeter: float) -> float:
        """
        Estimate pothole depth based on area and perimeter.
        
        Args:
            area: Pothole area in pixels.
            perimeter: Pothole perimeter in pixels.
            
        Returns:
            Estimated depth in arbitrary units.
        """
        try:
            # Simple heuristic: larger area and more irregular shape suggests deeper pothole
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            depth = (area / 1000.0) * (1 - circularity)  # Normalize and factor in irregularity
            return min(depth, 10.0)  # Cap at 10
        except Exception as e:
            self.logger.error(f"Error estimating pothole depth: {e}")
            return 0.0
    
    def _determine_pothole_severity(self, area: float, depth: float) -> str:
        """
        Determine pothole severity based on area and depth.
        
        Args:
            area: Pothole area in pixels.
            depth: Estimated pothole depth.
            
        Returns:
            Severity level string.
        """
        try:
            # Combine area and depth for severity assessment
            severity_score = (area / 5000.0) + (depth / 5.0)
            
            if severity_score < 0.3:
                return "low"
            elif severity_score < 0.6:
                return "medium"
            else:
                return "high"
        except Exception as e:
            self.logger.error(f"Error determining pothole severity: {e}")
            return "unknown"
    
    def _estimate_crack_width(self, contour: np.ndarray) -> float:
        """
        Estimate crack width from contour.
        
        Args:
            contour: Crack contour.
            
        Returns:
            Estimated width in pixels.
        """
        try:
            # Use distance transform to estimate width
            mask = np.zeros((contour.shape[0], contour.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Calculate distance transform
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # Get maximum distance (approximate width)
            width = np.max(dist_transform) * 2
            return width
        except Exception as e:
            self.logger.error(f"Error estimating crack width: {e}")
            return 0.0
    
    def _classify_crack_type(self, contour: np.ndarray, length: float) -> str:
        """
        Classify crack type based on contour properties.
        
        Args:
            contour: Crack contour.
            length: Crack length.
            
        Returns:
            Crack type string.
        """
        try:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity and aspect ratio
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Classify based on properties
            if circularity > 0.3:
                return "alligator"  # Network-like pattern
            elif aspect_ratio > 3 or aspect_ratio < 0.33:
                return "linear"  # Long and thin
            else:
                return "network"  # Complex pattern
        except Exception as e:
            self.logger.error(f"Error classifying crack type: {e}")
            return "unknown"
    
    def _determine_crack_severity(self, length: float, width: float) -> str:
        """
        Determine crack severity based on length and width.
        
        Args:
            length: Crack length in pixels.
            width: Crack width in pixels.
            
        Returns:
            Severity level string.
        """
        try:
            # Combine length and width for severity assessment
            severity_score = (length / 200.0) + (width / 10.0)
            
            if severity_score < 0.5:
                return "low"
            elif severity_score < 1.0:
                return "medium"
            else:
                return "high"
        except Exception as e:
            self.logger.error(f"Error determining crack severity: {e}")
            return "unknown"
    
    def _classify_surface_defect(self, contour: np.ndarray, area: float) -> str:
        """
        Classify surface defect type based on contour properties.
        
        Args:
            contour: Defect contour.
            area: Defect area.
            
        Returns:
            Defect type string.
        """
        try:
            # Calculate contour properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Classify based on properties
            if circularity > 0.7:
                return "patch"  # Circular/oval patches
            elif aspect_ratio > 2 or aspect_ratio < 0.5:
                return "crack"  # Linear defects
            elif area > 2000:
                return "pothole"  # Large defects
            else:
                return "wear"  # General wear
        except Exception as e:
            self.logger.error(f"Error classifying surface defect: {e}")
            return "unknown"
    
    def _determine_defect_severity(self, area: float, defect_type: str) -> str:
        """
        Determine defect severity based on area and type.
        
        Args:
            area: Defect area in pixels.
            defect_type: Type of defect.
            
        Returns:
            Severity level string.
        """
        try:
            # Different thresholds for different defect types
            thresholds = {
                'pothole': {'low': 1000, 'medium': 3000, 'high': 5000},
                'crack': {'low': 500, 'medium': 1500, 'high': 3000},
                'patch': {'low': 800, 'medium': 2000, 'high': 4000},
                'wear': {'low': 200, 'medium': 600, 'high': 1200}
            }
            
            type_thresholds = thresholds.get(defect_type, thresholds['wear'])
            
            if area < type_thresholds['low']:
                return "low"
            elif area < type_thresholds['medium']:
                return "medium"
            else:
                return "high"
        except Exception as e:
            self.logger.error(f"Error determining defect severity: {e}")
            return "unknown"
    
    def calculate_road_quality_score(self, potholes: List[Pothole], cracks: List[Crack], defects: List[SurfaceDefect]) -> float:
        """
        Calculate overall road quality score.
        
        Args:
            potholes: List of detected potholes.
            cracks: List of detected cracks.
            defects: List of detected surface defects.
            
        Returns:
            Quality score between 0.0 and 1.0.
        """
        try:
            # Base score
            score = 1.0
            
            # Deduct points for potholes
            for pothole in potholes:
                if pothole.severity == "high":
                    score -= 0.1
                elif pothole.severity == "medium":
                    score -= 0.05
                else:
                    score -= 0.02
            
            # Deduct points for cracks
            for crack in cracks:
                if crack.severity == "high":
                    score -= 0.05
                elif crack.severity == "medium":
                    score -= 0.03
                else:
                    score -= 0.01
            
            # Deduct points for surface defects
            for defect in defects:
                if defect.severity == "high":
                    score -= 0.03
                elif defect.severity == "medium":
                    score -= 0.02
                else:
                    score -= 0.01
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
        except Exception as e:
            self.logger.error(f"Error calculating road quality score: {e}")
            return 0.5
    
    def determine_overall_quality(self, quality_score: float) -> str:
        """
        Determine overall road quality based on score.
        
        Args:
            quality_score: Quality score between 0.0 and 1.0.
            
        Returns:
            Quality level string.
        """
        try:
            if quality_score >= self.quality_thresholds['excellent']:
                return "excellent"
            elif quality_score >= self.quality_thresholds['good']:
                return "good"
            elif quality_score >= self.quality_thresholds['fair']:
                return "fair"
            elif quality_score >= self.quality_thresholds['poor']:
                return "poor"
            else:
                return "critical"
        except Exception as e:
            self.logger.error(f"Error determining overall quality: {e}")
            return "unknown"
    
    def determine_maintenance_priority(self, potholes: List[Pothole], cracks: List[Crack], defects: List[SurfaceDefect]) -> str:
        """
        Determine maintenance priority based on detected issues.
        
        Args:
            potholes: List of detected potholes.
            cracks: List of detected cracks.
            defects: List of detected surface defects.
            
        Returns:
            Maintenance priority string.
        """
        try:
            # Count high severity issues
            high_severity_count = 0
            medium_severity_count = 0
            
            for pothole in potholes:
                if pothole.severity == "high":
                    high_severity_count += 1
                elif pothole.severity == "medium":
                    medium_severity_count += 1
            
            for crack in cracks:
                if crack.severity == "high":
                    high_severity_count += 1
                elif crack.severity == "medium":
                    medium_severity_count += 1
            
            for defect in defects:
                if defect.severity == "high":
                    high_severity_count += 1
                elif defect.severity == "medium":
                    medium_severity_count += 1
            
            # Determine priority
            if high_severity_count >= 3:
                return "urgent"
            elif high_severity_count >= 1 or medium_severity_count >= 5:
                return "high"
            elif medium_severity_count >= 2:
                return "medium"
            else:
                return "low"
        except Exception as e:
            self.logger.error(f"Error determining maintenance priority: {e}")
            return "unknown"
    
    def detect_road_conditions(self, image: np.ndarray) -> RoadConditionResult:
        """
        Perform complete road condition detection and analysis.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            RoadConditionResult object with complete analysis.
        """
        import time
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Detect different types of defects
            potholes = self.detect_potholes(processed_image)
            cracks = self.detect_cracks(processed_image)
            surface_defects = self.detect_surface_defects(processed_image)
            
            # Calculate quality metrics
            quality_score = self.calculate_road_quality_score(potholes, cracks, surface_defects)
            overall_quality = self.determine_overall_quality(quality_score)
            maintenance_priority = self.determine_maintenance_priority(potholes, cracks, surface_defects)
            
            processing_time = time.time() - start_time
            
            return RoadConditionResult(
                potholes=potholes,
                cracks=cracks,
                surface_defects=surface_defects,
                overall_quality=overall_quality,
                quality_score=quality_score,
                maintenance_priority=maintenance_priority,
                processing_time=processing_time
            )
        except Exception as e:
            self.logger.error(f"Error detecting road conditions: {e}")
            return RoadConditionResult(
                potholes=[],
                cracks=[],
                surface_defects=[],
                overall_quality="unknown",
                quality_score=0.0,
                maintenance_priority="unknown",
                processing_time=time.time() - start_time
            )
    
    def draw_detections(self, image: np.ndarray, result: RoadConditionResult) -> np.ndarray:
        """
        Draw detected road conditions on the image.
        
        Args:
            image: Input image as numpy array.
            result: Road condition detection result.
            
        Returns:
            Image with drawn detections.
        """
        try:
            result_image = image.copy()
            
            # Draw potholes
            for pothole in result.potholes:
                x1, y1, x2, y2 = pothole.bbox
                color = (0, 0, 255) if pothole.severity == "high" else (0, 165, 255) if pothole.severity == "medium" else (0, 255, 255)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(result_image, f"Pothole ({pothole.severity})", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw cracks
            for crack in result.cracks:
                color = (255, 0, 0) if crack.severity == "high" else (255, 165, 0) if crack.severity == "medium" else (255, 255, 0)
                cv2.line(result_image, crack.start_point, crack.end_point, color, 2)
                cv2.putText(result_image, f"Crack ({crack.severity})", crack.start_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw surface defects
            for defect in result.surface_defects:
                x1, y1, x2, y2 = defect.bbox
                color = (128, 0, 128) if defect.severity == "high" else (128, 128, 0) if defect.severity == "medium" else (128, 128, 128)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(result_image, f"{defect.defect_type} ({defect.severity})", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw quality information
            quality_text = f"Quality: {result.overall_quality} ({result.quality_score:.2f})"
            priority_text = f"Priority: {result.maintenance_priority}"
            cv2.putText(result_image, quality_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, priority_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return result_image
        except Exception as e:
            self.logger.error(f"Error drawing detections: {e}")
            return image
