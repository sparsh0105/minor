"""
Configuration settings for the Intelligent Traffic System.

This module contains all configuration parameters and settings
for the various components of the system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    
    # YOLO model settings
    yolo_model_path: str = "data/models/Model/weights/best.pt"
    yolo_confidence_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45
    
    # Traffic flow prediction settings
    traffic_flow_model_path: str = "data/models/data/models/traffic_flow_model.pkl"
    traffic_flow_confidence_threshold: float = 0.7
    
    # Road condition detection settings
    road_condition_model_path: str = "data/models/road_condition_model.pkl"
    road_condition_threshold: float = 0.6


@dataclass
class DetectionConfig:
    """Configuration for detection parameters."""
    
    # Lane detection settings
    lane_detection_roi_ratio: float = 0.6
    lane_detection_min_line_length: int = 50
    lane_detection_max_line_gap: int = 10
    lane_detection_rho: int = 1
    lane_detection_theta: float = 1.0
    lane_detection_threshold: int = 50
    
    # Traffic sign detection settings
    traffic_sign_min_area: int = 100
    traffic_sign_max_area: int = 50000
    traffic_sign_aspect_ratio_range: tuple = (0.5, 2.0)
    
    # Road condition detection settings
    pothole_min_area: int = 200
    crack_min_length: int = 50
    surface_quality_threshold: float = 0.3


@dataclass
class ProcessingConfig:
    """Configuration for image and video processing."""
    
    # Image processing settings
    image_width: int = 640
    image_height: int = 480
    image_quality: int = 95
    
    # Video processing settings
    video_fps: int = 30
    video_buffer_size: int = 10
    
    # Real-time processing settings
    real_time_mode: bool = True
    processing_interval: float = 0.1  # seconds


@dataclass
class OutputConfig:
    """Configuration for output and visualization."""
    
    # Output settings
    save_results: bool = True
    output_directory: str = "results"
    output_format: str = "json"  # json, csv, xml
    
    # Visualization settings
    show_visualization: bool = True
    visualization_window_size: tuple = (1280, 720)
    draw_bounding_boxes: bool = True
    draw_lanes: bool = True
    draw_traffic_signs: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/traffic_system.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Settings:
    """Main settings class for the Intelligent Traffic System."""
    
    # Model configuration
    models: ModelConfig = field(default_factory=ModelConfig)
    
    # Detection configuration
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    
    # Processing configuration
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Output configuration
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # System paths
    base_path: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "models")
    results_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    logs_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    def __post_init__(self) -> None:
        """Initialize paths and create directories if they don't exist."""
        self._create_directories()
        self._update_paths()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_path,
            self.models_path,
            self.results_path,
            self.logs_path,
            self.data_path / "datasets",
            self.data_path / "sample_videos",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _update_paths(self) -> None:
        """Update relative paths to absolute paths."""
        # Update model paths
        if not os.path.isabs(self.models.yolo_model_path):
            self.models.yolo_model_path = str(self.models_path / self.models.yolo_model_path)
        
        if not os.path.isabs(self.models.traffic_flow_model_path):
            self.models.traffic_flow_model_path = str(self.models_path / self.models.traffic_flow_model_path)
        
        if not os.path.isabs(self.models.road_condition_model_path):
            self.models.road_condition_model_path = str(self.models_path / self.models.road_condition_model_path)
        
        # Update output paths
        if not os.path.isabs(self.output.output_directory):
            self.output.output_directory = str(self.results_path / self.output.output_directory)
        
        if not os.path.isabs(self.output.log_file):
            self.output.log_file = str(self.logs_path / self.output.log_file)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "models": {
                "yolo_model_path": self.models.yolo_model_path,
                "yolo_confidence_threshold": self.models.yolo_confidence_threshold,
                "yolo_iou_threshold": self.models.yolo_iou_threshold,
                "traffic_flow_model_path": self.models.traffic_flow_model_path,
                "traffic_flow_confidence_threshold": self.models.traffic_flow_confidence_threshold,
                "road_condition_model_path": self.models.road_condition_model_path,
                "road_condition_threshold": self.models.road_condition_threshold,
            },
            "detection": {
                "lane_detection_roi_ratio": self.detection.lane_detection_roi_ratio,
                "lane_detection_min_line_length": self.detection.lane_detection_min_line_length,
                "lane_detection_max_line_gap": self.detection.lane_detection_max_line_gap,
                "lane_detection_rho": self.detection.lane_detection_rho,
                "lane_detection_theta": self.detection.lane_detection_theta,
                "lane_detection_threshold": self.detection.lane_detection_threshold,
                "traffic_sign_min_area": self.detection.traffic_sign_min_area,
                "traffic_sign_max_area": self.detection.traffic_sign_max_area,
                "traffic_sign_aspect_ratio_range": self.detection.traffic_sign_aspect_ratio_range,
                "pothole_min_area": self.detection.pothole_min_area,
                "crack_min_length": self.detection.crack_min_length,
                "surface_quality_threshold": self.detection.surface_quality_threshold,
            },
            "processing": {
                "image_width": self.processing.image_width,
                "image_height": self.processing.image_height,
                "image_quality": self.processing.image_quality,
                "video_fps": self.processing.video_fps,
                "video_buffer_size": self.processing.video_buffer_size,
                "real_time_mode": self.processing.real_time_mode,
                "processing_interval": self.processing.processing_interval,
            },
            "output": {
                "save_results": self.output.save_results,
                "output_directory": self.output.output_directory,
                "output_format": self.output.output_format,
                "show_visualization": self.output.show_visualization,
                "visualization_window_size": self.output.visualization_window_size,
                "draw_bounding_boxes": self.output.draw_bounding_boxes,
                "draw_lanes": self.output.draw_lanes,
                "draw_traffic_signs": self.output.draw_traffic_signs,
                "log_level": self.output.log_level,
                "log_file": self.output.log_file,
                "log_format": self.output.log_format,
            },
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Settings":
        """Create settings from dictionary."""
        models_config = ModelConfig(**config_dict.get("models", {}))
        detection_config = DetectionConfig(**config_dict.get("detection", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))
        
        return cls(
            models=models_config,
            detection=detection_config,
            processing=processing_config,
            output=output_config,
        )


# Global settings instance
settings = Settings()
