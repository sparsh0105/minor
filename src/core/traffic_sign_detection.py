"""
Traffic Sign Detection Module.

This module provides traffic sign detection and classification using both
YOLOv5 deep learning model and traditional computer vision techniques.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Using traditional CV methods only.")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

from ..config.settings import settings


@dataclass
class TrafficSign:
    """Information about a detected traffic sign."""
    
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    method: str  # 'yolo' or 'traditional'


@dataclass
class TrafficSignDetectionResult:
    """Result of traffic sign detection."""
    
    signs: List[TrafficSign]
    total_signs: int
    processing_time: float
    method_used: str


class TrafficSignDetector:
    """
    Traffic sign detection system using YOLOv5 and traditional computer vision.
    
    This class provides both deep learning-based detection using YOLOv5
    and fallback traditional computer vision methods using contour detection
    and SVM classification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the traffic sign detector.
        
        Args:
            config: Optional configuration dictionary to override default settings.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # YOLO model settings
        self.yolo_model_path = self.config.get(
            "yolo_model_path",
            settings.models.yolo_model_path
        )
        self.yolo_confidence_threshold = self.config.get(
            "yolo_confidence_threshold",
            settings.models.yolo_confidence_threshold
        )
        self.yolo_iou_threshold = self.config.get(
            "yolo_iou_threshold",
            settings.models.yolo_iou_threshold
        )
        
        # Traditional CV settings
        self.min_area = self.config.get(
            "traffic_sign_min_area",
            settings.detection.traffic_sign_min_area
        )
        self.max_area = self.config.get(
            "traffic_sign_max_area",
            settings.detection.traffic_sign_max_area
        )
        self.aspect_ratio_range = self.config.get(
            "traffic_sign_aspect_ratio_range",
            settings.detection.traffic_sign_aspect_ratio_range
        )
        
        # Initialize models
        self.yolo_model = None
        self.svm_model = None
        self.scaler = None
        self.label_encoder = None
        
        # Traffic sign class names (YOLO format)
        self.class_names = [
            "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60",
            "speed_limit_70", "speed_limit_80", "speed_limit_90", "speed_limit_100",
            "speed_limit_110", "speed_limit_120", "stop", "yield", "no_entry",
            "no_parking", "no_overtaking", "one_way", "roundabout", "traffic_light",
            "pedestrian_crossing", "school_zone", "construction", "slippery_road",
            "sharp_turn_left", "sharp_turn_right", "merge", "lane_ends",
            "road_narrows", "bridge", "tunnel", "railroad_crossing", "animal_crossing",
            "falling_rocks", "steep_hill", "winding_road", "bump", "dip",
            "soft_shoulder", "narrow_bridge", "low_clearance", "weight_limit",
            "height_limit", "width_limit", "length_limit", "axle_load_limit",
            "truck_route", "bus_route", "bicycle_route", "pedestrian_route",
            "emergency_services", "hospital", "gas_station", "restaurant",
            "hotel", "camping", "parking", "rest_area", "scenic_view",
            "recreation_area", "historic_site", "museum", "zoo", "airport",
            "train_station", "bus_station", "ferry", "taxi", "other"
        ]
        
        # Traditional CV class names (SVM format)
        self.traditional_class_names = [
            "non_traffic_sign", "speed_limit", "stop", "yield", "no_entry",
            "no_parking", "no_overtaking", "one_way", "roundabout", "traffic_light",
            "pedestrian_crossing", "school_zone", "other"
        ]
        
        self._initialize_models()
        self.logger.info("Traffic sign detector initialized successfully")
    
    def _initialize_models(self) -> None:
        """Initialize YOLO and SVM models."""
        try:
            # Initialize YOLO model if available
            if YOLO_AVAILABLE and Path(self.yolo_model_path).exists():
                self.yolo_model = YOLO(self.yolo_model_path)
                self.logger.info(f"YOLO model loaded from {self.yolo_model_path}")
            else:
                self.logger.warning("YOLO model not available, using traditional methods only")
            
            # Initialize SVM model for traditional detection
            self._load_svm_model()
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def _load_svm_model(self) -> None:
        """Load pre-trained SVM model for traditional detection."""
        try:
            # Try to load pre-trained SVM model
            svm_model_path = Path(self.yolo_model_path).parent / "traffic_sign_svm.pkl"
            if svm_model_path.exists():
                with open(svm_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.svm_model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.label_encoder = model_data['label_encoder']
                self.logger.info("SVM model loaded successfully")
            else:
                self.logger.warning("SVM model not found, traditional detection will use basic classification")
        except Exception as e:
            self.logger.error(f"Error loading SVM model: {e}")
    
    def detect_with_yolo(self, image: np.ndarray) -> List[TrafficSign]:
        """
        Detect traffic signs using YOLO model.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            List of detected traffic signs.
        """
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.yolo_model(image, conf=self.yolo_confidence_threshold, iou=self.yolo_iou_threshold)
            
            signs = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Extract class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                        
                        # Calculate center and area
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        area = (x2 - x1) * (y2 - y1)
                        
                        sign = TrafficSign(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            center=center,
                            area=area,
                            method="yolo"
                        )
                        signs.append(sign)
            
            return signs
        except Exception as e:
            self.logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def preprocess_for_traditional_cv(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for traditional computer vision detection.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Preprocessed image.
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for traffic signs (red, blue, yellow, white)
            # Red range
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue range
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Yellow range
            lower_yellow = np.array([20, 50, 50])
            upper_yellow = np.array([30, 255, 255])
            
            # White range
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            # Create masks for each color
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # Combine all masks
            combined_mask = cv2.bitwise_or(mask_red, mask_blue)
            combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
            combined_mask = cv2.bitwise_or(combined_mask, mask_white)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            return combined_mask
        except Exception as e:
            self.logger.error(f"Error preprocessing image for traditional CV: {e}")
            return np.zeros_like(image[:, :, 0])
    
    def find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the preprocessed mask.
        
        Args:
            mask: Preprocessed binary mask.
            
        Returns:
            List of contours.
        """
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except Exception as e:
            self.logger.error(f"Error finding contours: {e}")
            return []
    
    def filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter contours based on area and aspect ratio.
        
        Args:
            contours: List of contours to filter.
            
        Returns:
            List of filtered contours.
        """
        try:
            filtered_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if self.min_area <= area <= self.max_area:
                    # Calculate aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filter by aspect ratio
                    if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                        filtered_contours.append(contour)
            
            return filtered_contours
        except Exception as e:
            self.logger.error(f"Error filtering contours: {e}")
            return []
    
    def extract_features(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Extract features from a contour for SVM classification.
        
        Args:
            image: Original image.
            contour: Contour to extract features from.
            
        Returns:
            Feature vector.
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Crop the region
            roi = image[y:y+h, x:x+w]
            
            # Resize to standard size
            roi_resized = cv2.resize(roi, (32, 32))
            
            # Convert to grayscale
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            
            # Flatten to create feature vector
            features = roi_gray.flatten()
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros(32 * 32)
    
    def classify_with_svm(self, features: np.ndarray) -> Tuple[int, str, float]:
        """
        Classify features using SVM model.
        
        Args:
            features: Feature vector.
            
        Returns:
            Tuple of (class_id, class_name, confidence).
        """
        try:
            if self.svm_model is None or self.scaler is None:
                # Return default classification if model not available
                return 0, "unknown", 0.5
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict class
            class_id = self.svm_model.predict(features_scaled)[0]
            confidence = max(self.svm_model.predict_proba(features_scaled)[0])
            
            # Get class name
            class_name = self.traditional_class_names[class_id] if class_id < len(self.traditional_class_names) else "unknown"
            
            return class_id, class_name, confidence
        except Exception as e:
            self.logger.error(f"Error in SVM classification: {e}")
            return 0, "unknown", 0.0
    
    def detect_with_traditional_cv(self, image: np.ndarray) -> List[TrafficSign]:
        """
        Detect traffic signs using traditional computer vision methods.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            List of detected traffic signs.
        """
        try:
            # Preprocess image
            mask = self.preprocess_for_traditional_cv(image)
            
            # Find contours
            contours = self.find_contours(mask)
            
            # Filter contours
            filtered_contours = self.filter_contours(contours)
            
            signs = []
            for contour in filtered_contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract features
                features = self.extract_features(image, contour)
                
                # Classify using SVM
                class_id, class_name, confidence = self.classify_with_svm(features)
                
                # Skip non-traffic signs
                if class_name == "non_traffic_sign":
                    continue
                
                # Calculate center and area
                center = (x + w // 2, y + h // 2)
                area = w * h
                
                sign = TrafficSign(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x, y, x + w, y + h),
                    center=center,
                    area=area,
                    method="traditional"
                )
                signs.append(sign)
            
            return signs
        except Exception as e:
            self.logger.error(f"Error in traditional CV detection: {e}")
            return []
    
    def detect_signs(self, image: np.ndarray, use_yolo: bool = True) -> TrafficSignDetectionResult:
        """
        Detect traffic signs in the input image.
        
        Args:
            image: Input image as numpy array.
            use_yolo: Whether to use YOLO model (True) or traditional CV (False).
            
        Returns:
            TrafficSignDetectionResult object containing detection results.
        """
        import time
        start_time = time.time()
        
        try:
            signs = []
            method_used = "none"
            
            if use_yolo and self.yolo_model is not None:
                # Try YOLO detection first
                signs = self.detect_with_yolo(image)
                method_used = "yolo"
                
                # If no signs detected with YOLO, try traditional CV as fallback
                if not signs:
                    signs = self.detect_with_traditional_cv(image)
                    method_used = "traditional_fallback"
            else:
                # Use traditional CV only
                signs = self.detect_with_traditional_cv(image)
                method_used = "traditional"
            
            processing_time = time.time() - start_time
            
            return TrafficSignDetectionResult(
                signs=signs,
                total_signs=len(signs),
                processing_time=processing_time,
                method_used=method_used
            )
        except Exception as e:
            self.logger.error(f"Error detecting traffic signs: {e}")
            return TrafficSignDetectionResult(
                signs=[],
                total_signs=0,
                processing_time=time.time() - start_time,
                method_used="error"
            )
    
    def draw_detections(self, image: np.ndarray, result: TrafficSignDetectionResult) -> np.ndarray:
        """
        Draw detected traffic signs on the image.
        
        Args:
            image: Input image as numpy array.
            result: Detection result to draw.
            
        Returns:
            Image with drawn detections.
        """
        try:
            result_image = image.copy()
            
            for sign in result.signs:
                x1, y1, x2, y2 = sign.bbox
                
                # Choose color based on method
                color = (0, 255, 0) if sign.method == "yolo" else (255, 0, 0)
                
                # Draw bounding box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{sign.class_name}: {sign.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(result_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw center point
                center_x, center_y = sign.center
                cv2.circle(result_image, (center_x, center_y), 3, color, -1)
            
            # Draw processing info
            info_text = f"Signs: {result.total_signs} | Method: {result.method_used} | Time: {result.processing_time:.3f}s"
            cv2.putText(result_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return result_image
        except Exception as e:
            self.logger.error(f"Error drawing detections: {e}")
            return image
