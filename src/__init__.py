"""
Intelligent Road and Traffic Monitoring System.

This package provides a comprehensive solution for road and traffic monitoring
using computer vision and machine learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Lakshay Saurabh"
__email__ = "lakshay@example.com"

from .core.road_detection import RoadDetector
from .core.traffic_sign_detection import TrafficSignDetector
from .core.traffic_flow_prediction import TrafficFlowPredictor
from .core.road_condition_detection import RoadConditionDetector

__all__ = [
    "RoadDetector",
    "TrafficSignDetector", 
    "TrafficFlowPredictor",
    "RoadConditionDetector",
]
