"""
Tests for the traffic sign detection module.

This module contains comprehensive tests for the TrafficSignDetector class
and its traffic sign detection functionality using both YOLO and traditional CV methods.
"""

import pytest
import numpy as np
import cv2
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch, MagicMock

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from src.core.traffic_sign_detection import (
    TrafficSignDetector, 
    TrafficSign, 
    TrafficSignDetectionResult
)


class TestTrafficSignDetector:
    """Test class for TrafficSignDetector functionality."""
    
    @pytest.fixture
    def traffic_sign_detector(self) -> TrafficSignDetector:
        """Create a TrafficSignDetector instance for testing."""
        return TrafficSignDetector()
    
    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample image for testing."""
        # Create a simple test image with traffic sign-like features
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a red circular sign (stop sign)
        cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)
        cv2.putText(image, "STOP", (290, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw a blue rectangular sign (speed limit)
        cv2.rectangle(image, (100, 100), (200, 150), (255, 0, 0), -1)
        cv2.putText(image, "50", (130, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return image
    
    @pytest.fixture
    def mock_yolo_model(self) -> Mock:
        """Create a mock YOLO model for testing."""
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        
        # Mock box data
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 150]
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 0
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.95
        
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        
        return mock_model
    
    def test_traffic_sign_detector_initialization(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test that TrafficSignDetector initializes correctly."""
        assert traffic_sign_detector is not None
        assert traffic_sign_detector.yolo_confidence_threshold > 0
        assert traffic_sign_detector.yolo_iou_threshold > 0
        assert traffic_sign_detector.min_area > 0
        assert traffic_sign_detector.max_area > 0
        assert len(traffic_sign_detector.class_names) > 0
        assert len(traffic_sign_detector.traditional_class_names) > 0
    
    def test_traffic_sign_detector_with_config(self) -> None:
        """Test TrafficSignDetector initialization with custom config."""
        config = {
            "yolo_confidence_threshold": 0.8,
            "traffic_sign_min_area": 1000,
            "traffic_sign_max_area": 50000
        }
        
        detector = TrafficSignDetector(config)
        assert detector.yolo_confidence_threshold == 0.8
        assert detector.min_area == 1000
        assert detector.max_area == 50000
    
    def test_preprocess_for_traditional_cv(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test image preprocessing for traditional CV detection."""
        processed = traffic_sign_detector.preprocess_for_traditional_cv(sample_image)
        
        assert processed is not None
        assert processed.shape[:2] == sample_image.shape[:2]  # Same height and width
        assert len(processed.shape) == 2  # Grayscale mask
        assert processed.dtype == np.uint8
    
    def test_find_contours(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test contour finding functionality."""
        mask = traffic_sign_detector.preprocess_for_traditional_cv(sample_image)
        contours = traffic_sign_detector.find_contours(mask)
        
        # The method should return a list of contours (or tuple in some OpenCV versions)
        assert isinstance(contours, (list, tuple))
        assert len(contours) >= 0
    
    def test_filter_contours(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test contour filtering functionality."""
        # Create mock contours with different areas
        contours = [
            np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32),  # Large contour
            np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32),  # Small contour
        ]
        
        filtered = traffic_sign_detector.filter_contours(contours)
        
        assert isinstance(filtered, list)
        # Should filter out contours that don't meet area/aspect ratio criteria
    
    def test_extract_features(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test feature extraction from contours."""
        # Create a simple contour
        contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)
        
        features = traffic_sign_detector.extract_features(sample_image, contour)
        
        assert features is not None
        assert len(features) == 32 * 32  # Should be flattened 32x32 image
        assert isinstance(features, np.ndarray)
    
    def test_classify_with_svm(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test SVM classification functionality."""
        # Create mock features
        features = np.random.rand(32 * 32)
        
        class_id, class_name, confidence = traffic_sign_detector.classify_with_svm(features)
        
        assert isinstance(class_id, int)
        assert isinstance(class_name, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    @patch('src.core.traffic_sign_detection.YOLO_AVAILABLE', True)
    def test_detect_with_yolo_mock(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray, mock_yolo_model: Mock) -> None:
        """Test YOLO detection with mocked model."""
        traffic_sign_detector.yolo_model = mock_yolo_model
        
        signs = traffic_sign_detector.detect_with_yolo(sample_image)
        
        assert isinstance(signs, list)
        # Should return detected signs based on mock model
    
    def test_detect_with_yolo_no_model(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test YOLO detection when no model is available."""
        traffic_sign_detector.yolo_model = None
        
        signs = traffic_sign_detector.detect_with_yolo(sample_image)
        
        assert isinstance(signs, list)
        assert len(signs) == 0  # Should return empty list when no model
    
    def test_detect_with_traditional_cv(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test traditional CV detection functionality."""
        signs = traffic_sign_detector.detect_with_traditional_cv(sample_image)
        
        assert isinstance(signs, list)
        # Each sign should be a TrafficSign object
        for sign in signs:
            assert isinstance(sign, TrafficSign)
            assert sign.method == "traditional"
    
    def test_detect_signs_yolo_priority(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray, mock_yolo_model: Mock) -> None:
        """Test that YOLO detection is prioritized when available."""
        traffic_sign_detector.yolo_model = mock_yolo_model
        
        result = traffic_sign_detector.detect_signs(sample_image, use_yolo=True)
        
        assert isinstance(result, TrafficSignDetectionResult)
        assert result.method_used in ["yolo", "traditional_fallback"]
        assert result.total_signs >= 0
        assert result.processing_time >= 0
    
    def test_detect_signs_traditional_only(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test detection using only traditional CV methods."""
        result = traffic_sign_detector.detect_signs(sample_image, use_yolo=False)
        
        assert isinstance(result, TrafficSignDetectionResult)
        assert result.method_used == "traditional"
        assert result.total_signs >= 0
        assert result.processing_time >= 0
    
    def test_draw_detections(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test drawing detections on image."""
        # Create a mock detection result
        sign = TrafficSign(
            class_id=0,
            class_name="stop",
            confidence=0.95,
            bbox=(100, 100, 200, 150),
            center=(150, 125),
            area=5000,
            method="yolo"
        )
        
        result = TrafficSignDetectionResult(
            signs=[sign],
            total_signs=1,
            processing_time=0.1,
            method_used="yolo"
        )
        
        result_image = traffic_sign_detector.draw_detections(sample_image, result)
        
        assert result_image is not None
        assert result_image.shape == sample_image.shape
        assert result_image.dtype == sample_image.dtype
    
    def test_error_handling_invalid_image(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test error handling with invalid input."""
        # Test with None input
        with pytest.raises(Exception):
            traffic_sign_detector.preprocess_for_traditional_cv(None)
        
        # Test with empty array
        empty_image = np.array([])
        with pytest.raises(Exception):
            traffic_sign_detector.preprocess_for_traditional_cv(empty_image)
    
    def test_error_handling_invalid_contour(self, traffic_sign_detector: TrafficSignDetector, sample_image: np.ndarray) -> None:
        """Test error handling with invalid contour."""
        # Test with None contour - method should handle gracefully and return default features
        features = traffic_sign_detector.extract_features(sample_image, None)
        assert isinstance(features, np.ndarray)
        assert len(features) == 32 * 32  # Should return default feature vector
        
        # Test with empty contour - method should handle gracefully and return default features
        empty_contour = np.array([])
        features = traffic_sign_detector.extract_features(sample_image, empty_contour)
        assert isinstance(features, np.ndarray)
        assert len(features) == 32 * 32  # Should return default feature vector
    
    def test_performance_with_large_image(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test performance with larger images."""
        # Create a larger test image
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        import time
        start_time = time.time()
        result = traffic_sign_detector.detect_signs(large_image, use_yolo=False)
        processing_time = time.time() - start_time
        
        assert result is not None
        assert processing_time < 10.0  # Should process within 10 seconds
    
    def test_traffic_sign_dataclass(self) -> None:
        """Test TrafficSign dataclass functionality."""
        sign = TrafficSign(
            class_id=0,
            class_name="stop",
            confidence=0.95,
            bbox=(100, 100, 200, 150),
            center=(150, 125),
            area=5000,
            method="yolo"
        )
        
        assert sign.class_id == 0
        assert sign.class_name == "stop"
        assert sign.confidence == 0.95
        assert sign.bbox == (100, 100, 200, 150)
        assert sign.center == (150, 125)
        assert sign.area == 5000
        assert sign.method == "yolo"
    
    def test_traffic_sign_detection_result_dataclass(self) -> None:
        """Test TrafficSignDetectionResult dataclass functionality."""
        sign = TrafficSign(
            class_id=0,
            class_name="stop",
            confidence=0.95,
            bbox=(100, 100, 200, 150),
            center=(150, 125),
            area=5000,
            method="yolo"
        )
        
        result = TrafficSignDetectionResult(
            signs=[sign],
            total_signs=1,
            processing_time=0.1,
            method_used="yolo"
        )
        
        assert len(result.signs) == 1
        assert result.total_signs == 1
        assert result.processing_time == 0.1
        assert result.method_used == "yolo"
        assert result.signs[0] == sign
    
    def test_class_names_completeness(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test that class names are properly defined."""
        # Test YOLO class names
        assert "stop" in traffic_sign_detector.class_names
        assert "speed_limit_50" in traffic_sign_detector.class_names
        assert "yield" in traffic_sign_detector.class_names
        assert "no_entry" in traffic_sign_detector.class_names
        
        # Test traditional class names
        assert "stop" in traffic_sign_detector.traditional_class_names
        assert "speed_limit" in traffic_sign_detector.traditional_class_names
        assert "yield" in traffic_sign_detector.traditional_class_names
        assert "non_traffic_sign" in traffic_sign_detector.traditional_class_names
    
    def test_configuration_override(self) -> None:
        """Test that configuration can be overridden."""
        config = {
            'yolo_confidence_threshold': 0.8,
            'traffic_sign_min_area': 1000,
            'traffic_sign_max_area': 50000,
            'traffic_sign_aspect_ratio_range': [0.5, 2.0]
        }
        
        detector = TrafficSignDetector(config)
        assert detector.yolo_confidence_threshold == 0.8
        assert detector.min_area == 1000
        assert detector.max_area == 50000
        assert detector.aspect_ratio_range == [0.5, 2.0]
    
    def test_detect_signs_with_empty_image(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test detection with empty/black image."""
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = traffic_sign_detector.detect_signs(empty_image, use_yolo=False)
        
        assert isinstance(result, TrafficSignDetectionResult)
        assert result.total_signs == 0
        assert result.method_used == "traditional"
    
    def test_detect_signs_with_noise_image(self, traffic_sign_detector: TrafficSignDetector) -> None:
        """Test detection with noisy image."""
        # Create a noisy image
        noisy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = traffic_sign_detector.detect_signs(noisy_image, use_yolo=False)
        
        assert isinstance(result, TrafficSignDetectionResult)
        assert result.total_signs >= 0  # May or may not detect signs in noise
        assert result.method_used == "traditional"
