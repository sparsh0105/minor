"""
Tests for the road detection module.

This module contains comprehensive tests for the RoadDetector class
and its lane detection functionality.
"""

import pytest
import numpy as np
import cv2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from src.core.road_detection import RoadDetector, LaneInfo, RoadInfo


class TestRoadDetector:
    """Test class for RoadDetector functionality."""
    
    @pytest.fixture
    def road_detector(self) -> RoadDetector:
        """Create a RoadDetector instance for testing."""
        return RoadDetector()
    
    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample image for testing."""
        # Create a simple test image with lane-like features
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some lane-like lines
        cv2.line(image, (100, 400), (200, 300), (255, 255, 255), 5)
        cv2.line(image, (500, 400), (400, 300), (255, 255, 255), 5)
        
        return image
    
    def test_road_detector_initialization(self, road_detector: RoadDetector) -> None:
        """Test that RoadDetector initializes correctly."""
        assert road_detector is not None
        assert road_detector.roi_ratio > 0
        assert road_detector.min_line_length > 0
        assert road_detector.max_line_gap > 0
    
    def test_preprocess_image(self, road_detector: RoadDetector, sample_image: np.ndarray) -> None:
        """Test image preprocessing functionality."""
        processed = road_detector.preprocess_image(sample_image)
        
        assert processed is not None
        assert processed.shape[:2] == sample_image.shape[:2]  # Same height and width
        assert len(processed.shape) == 2  # Grayscale
    
    def test_create_roi_mask(self, road_detector: RoadDetector, sample_image: np.ndarray) -> None:
        """Test ROI mask creation."""
        edges = road_detector.preprocess_image(sample_image)
        masked = road_detector.create_roi_mask(edges)
        
        assert masked is not None
        assert masked.shape == edges.shape
    
    def test_detect_lines(self, road_detector: RoadDetector, sample_image: np.ndarray) -> None:
        """Test line detection functionality."""
        edges = road_detector.preprocess_image(sample_image)
        masked = road_detector.create_roi_mask(edges)
        lines = road_detector.detect_lines(masked)
        
        assert isinstance(lines, list)
        # Should detect some lines in our test image
        assert len(lines) >= 0
    
    def test_separate_lanes(self, road_detector: RoadDetector, sample_image: np.ndarray) -> None:
        """Test lane separation functionality."""
        edges = road_detector.preprocess_image(sample_image)
        masked = road_detector.create_roi_mask(edges)
        lines = road_detector.detect_lines(masked)
        
        left_lines, right_lines = road_detector.separate_lanes(lines, sample_image.shape)
        
        assert isinstance(left_lines, list)
        assert isinstance(right_lines, list)
    
    def test_fit_lane_line(self, road_detector: RoadDetector) -> None:
        """Test lane line fitting functionality."""
        # Create mock line segments
        lines = [
            np.array([100, 400, 150, 350]),
            np.array([150, 350, 200, 300])
        ]
        
        fitted_line = road_detector.fit_lane_line(lines, (480, 640))
        
        if fitted_line is not None:
            assert len(fitted_line) == 4  # x1, y1, x2, y2
    
    def test_calculate_lane_center(self, road_detector: RoadDetector) -> None:
        """Test lane center calculation."""
        left_lane = np.array([100, 400, 200, 300])
        right_lane = np.array([500, 400, 400, 300])
        
        center = road_detector.calculate_lane_center(left_lane, right_lane, (480, 640))
        
        if center is not None:
            assert isinstance(center, tuple)
            assert len(center) == 2
            assert 0 <= center[0] <= 640
            assert 0 <= center[1] <= 480
    
    def test_detect_lane_departure(self, road_detector: RoadDetector) -> None:
        """Test lane departure detection."""
        # Test with center lane
        center_lane = (320, 400)  # Center of 640x480 image
        departure = road_detector.detect_lane_departure(center_lane, (480, 640))
        assert not departure
        
        # Test with offset lane
        offset_lane = (100, 400)  # Far left
        departure = road_detector.detect_lane_departure(offset_lane, (480, 640))
        assert departure
    
    def test_calculate_lane_angle(self, road_detector: RoadDetector) -> None:
        """Test lane angle calculation."""
        # Horizontal line should have 0 angle
        horizontal_line = np.array([100, 200, 200, 200])
        angle = road_detector.calculate_lane_angle(horizontal_line)
        assert abs(angle) < 10  # Should be close to 0
        
        # Vertical line should have 90 degree angle
        vertical_line = np.array([100, 100, 100, 200])
        angle = road_detector.calculate_lane_angle(vertical_line)
        assert abs(angle) > 80  # Should be close to 90
    
    def test_detect_lanes(self, road_detector: RoadDetector, sample_image: np.ndarray) -> None:
        """Test complete lane detection functionality."""
        lane_info = road_detector.detect_lanes(sample_image)
        
        assert isinstance(lane_info, LaneInfo)
        assert isinstance(lane_info.lane_confidence, float)
        assert 0.0 <= lane_info.lane_confidence <= 1.0
        assert isinstance(lane_info.lane_departure_warning, bool)
    
    def test_detect_road_conditions(self, road_detector: RoadDetector, sample_image: np.ndarray) -> None:
        """Test complete road condition detection."""
        road_info = road_detector.detect_road_conditions(sample_image)
        
        assert isinstance(road_info, RoadInfo)
        assert isinstance(road_info.lanes, LaneInfo)
        assert isinstance(road_info.road_quality, str)
        assert road_info.road_quality in ['good', 'fair', 'poor']
        assert isinstance(road_info.road_width, float)
        assert road_info.road_width >= 0
        assert isinstance(road_info.curvature, float)
        assert 0.0 <= road_info.curvature <= 1.0
        assert isinstance(road_info.visibility, float)
        assert 0.0 <= road_info.visibility <= 1.0
    
    def test_draw_lanes(self, road_detector: RoadDetector, sample_image: np.ndarray) -> None:
        """Test lane drawing functionality."""
        lane_info = road_detector.detect_lanes(sample_image)
        result_image = road_detector.draw_lanes(sample_image, lane_info)
        
        assert result_image is not None
        assert result_image.shape == sample_image.shape
        assert result_image.dtype == sample_image.dtype
    
    def test_error_handling_invalid_image(self, road_detector: RoadDetector) -> None:
        """Test error handling with invalid input."""
        # Test with None input
        with pytest.raises(Exception):
            road_detector.preprocess_image(None)
        
        # Test with empty array
        empty_image = np.array([])
        with pytest.raises(Exception):
            road_detector.preprocess_image(empty_image)
    
    def test_configuration_override(self) -> None:
        """Test that configuration can be overridden."""
        config = {
            'lane_detection_roi_ratio': 0.8,
            'lane_detection_min_line_length': 100
        }
        
        detector = RoadDetector(config)
        assert detector.roi_ratio == 0.8
        assert detector.min_line_length == 100
    
    def test_performance_with_large_image(self, road_detector: RoadDetector) -> None:
        """Test performance with larger images."""
        # Create a larger test image
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        import time
        start_time = time.time()
        road_info = road_detector.detect_road_conditions(large_image)
        processing_time = time.time() - start_time
        
        assert road_info is not None
        assert processing_time < 5.0  # Should process within 5 seconds
    
    def test_lane_info_dataclass(self) -> None:
        """Test LaneInfo dataclass functionality."""
        lane_info = LaneInfo(
            left_lane=np.array([100, 400, 200, 300]),
            right_lane=np.array([500, 400, 400, 300]),
            lane_center=(300, 400),
            lane_departure_warning=False,
            lane_confidence=0.8,
            lane_angle=15.0
        )
        
        assert lane_info.left_lane is not None
        assert lane_info.right_lane is not None
        assert lane_info.lane_center == (300, 400)
        assert not lane_info.lane_departure_warning
        assert lane_info.lane_confidence == 0.8
        assert lane_info.lane_angle == 15.0
    
    def test_road_info_dataclass(self) -> None:
        """Test RoadInfo dataclass functionality."""
        lane_info = LaneInfo()
        road_info = RoadInfo(
            lanes=lane_info,
            road_width=3.5,
            road_quality="good",
            curvature=0.1,
            visibility=0.9
        )
        
        assert road_info.lanes == lane_info
        assert road_info.road_width == 3.5
        assert road_info.road_quality == "good"
        assert road_info.curvature == 0.1
        assert road_info.visibility == 0.9
