"""
Main module for the Intelligent Road and Traffic Monitoring System.

This module provides the unified interface for all traffic monitoring components
including road detection, traffic sign detection, traffic flow prediction,
and road condition detection.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import time

from .core.road_detection import RoadDetector, RoadInfo
from .core.traffic_sign_detection import TrafficSignDetector, TrafficSignDetectionResult
from .core.traffic_flow_prediction import TrafficFlowPredictor, TrafficFlowResult, VehicleCounts
from .core.road_condition_detection import RoadConditionDetector, RoadConditionResult
from .config.settings import settings


class IntelligentTrafficSystem:
    """
    Main class for the Intelligent Road and Traffic Monitoring System.
    
    This class integrates all four core modules:
    - Road Detection (lane detection)
    - Traffic Sign Detection
    - Traffic Flow Prediction
    - Road Condition Detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the intelligent traffic system.
        
        Args:
            config: Optional configuration dictionary to override default settings.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize all detection modules
        self.road_detector = RoadDetector(self.config.get('road_detection', {}))
        self.traffic_sign_detector = TrafficSignDetector(self.config.get('traffic_sign_detection', {}))
        self.traffic_flow_predictor = TrafficFlowPredictor(self.config.get('traffic_flow_prediction', {}))
        self.road_condition_detector = RoadConditionDetector(self.config.get('road_condition_detection', {}))
        
        # System state
        self.is_initialized = True
        self.processing_stats = {
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0
        }
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("Intelligent Traffic System initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            log_level = getattr(logging, settings.output.log_level.upper(), logging.INFO)
            
            # Create logs directory if it doesn't exist
            Path(settings.output.log_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=log_level,
                format=settings.output.log_format,
                handlers=[
                    logging.FileHandler(settings.output.log_file),
                    logging.StreamHandler()
                ]
            )
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def process_image(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Process a single image with all detection modules.
        
        Args:
            image: Input image as numpy array, file path, or Path object.
            
        Returns:
            Dictionary containing all detection results.
        """
        try:
            # Load image if path is provided
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
                if image is None:
                    raise ValueError(f"Could not load image from {image}")
            
            start_time = time.time()
            
            # Process with all modules
            road_info = self.road_detector.detect_road_conditions(image)
            traffic_signs = self.traffic_sign_detector.detect_signs(image)
            road_conditions = self.road_condition_detector.detect_road_conditions(image)
            
            # Estimate vehicle counts for traffic flow prediction
            vehicle_counts = self._estimate_vehicle_counts(image)
            traffic_flow = self.traffic_flow_predictor.predict_traffic_flow(vehicle_counts)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processing_stats['total_frames_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['average_fps'] = (
                self.processing_stats['total_frames_processed'] / 
                self.processing_stats['total_processing_time']
            )
            
            # Compile results
            results = {
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'road_detection': {
                    'lanes_detected': road_info.lanes.left_lane is not None and road_info.lanes.right_lane is not None,
                    'lane_departure_warning': road_info.lanes.lane_departure_warning,
                    'lane_confidence': road_info.lanes.lane_confidence,
                    'road_quality': road_info.road_quality,
                    'road_width': road_info.road_width,
                    'curvature': road_info.curvature,
                    'visibility': road_info.visibility
                },
                'traffic_signs': {
                    'total_signs': traffic_signs.total_signs,
                    'signs': [
                        {
                            'class_name': sign.class_name,
                            'confidence': sign.confidence,
                            'bbox': sign.bbox,
                            'center': sign.center,
                            'method': sign.method
                        }
                        for sign in traffic_signs.signs
                    ],
                    'method_used': traffic_signs.method_used,
                    'processing_time': traffic_signs.processing_time
                },
                'traffic_flow': {
                    'current_counts': {
                        'car_count': traffic_flow.current_counts.car_count,
                        'bike_count': traffic_flow.current_counts.bike_count,
                        'bus_count': traffic_flow.current_counts.bus_count,
                        'truck_count': traffic_flow.current_counts.truck_count,
                        'total_count': traffic_flow.current_counts.total_count
                    },
                    'prediction': {
                        'predicted_situation': traffic_flow.prediction.predicted_situation,
                        'confidence': traffic_flow.prediction.confidence,
                        'congestion_level': traffic_flow.prediction.congestion_level
                    },
                    'historical_trend': traffic_flow.historical_trend,
                    'peak_hour_indicator': traffic_flow.peak_hour_indicator,
                    'processing_time': traffic_flow.processing_time
                },
                'road_conditions': {
                    'potholes_count': len(road_conditions.potholes),
                    'cracks_count': len(road_conditions.cracks),
                    'surface_defects_count': len(road_conditions.surface_defects),
                    'overall_quality': road_conditions.overall_quality,
                    'quality_score': road_conditions.quality_score,
                    'maintenance_priority': road_conditions.maintenance_priority,
                    'processing_time': road_conditions.processing_time
                },
                'system_stats': self.processing_stats.copy()
            }
            
            self.logger.info(f"Image processed successfully in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    def process_video_stream(self, source: Union[str, int, Path], output_path: Optional[str] = None) -> None:
        """
        Process video stream (file or camera) with all detection modules.
        
        Args:
            source: Video source (file path, camera index, or Path object).
            output_path: Optional output path for processed video.
        """
        try:
            # Open video source
            cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {source}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer if output path is provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            self.logger.info(f"Starting video processing: {source}")
            self.logger.info(f"Video properties: {width}x{height} @ {fps}fps")
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                results = self.process_image(frame)
                
                # Draw visualizations
                if settings.output.show_visualization:
                    frame = self._draw_visualizations(frame, results)
                
                # Write frame if output is specified
                if writer:
                    writer.write(frame)
                
                # Display frame
                if settings.output.show_visualization:
                    cv2.imshow('Intelligent Traffic System', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    self.logger.info(f"Processed {frame_count} frames, Current FPS: {current_fps:.2f}")
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            self.logger.info(f"Video processing completed: {frame_count} frames in {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing video stream: {e}")
            raise
    
    def _estimate_vehicle_counts(self, image: np.ndarray) -> VehicleCounts:
        """
        Estimate vehicle counts from image (simplified implementation).
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            VehicleCounts object with estimated counts.
        """
        try:
            # This is a simplified implementation
            # In a real system, you would use object detection models
            # to count vehicles in the image
            
            # For now, we'll use a simple heuristic based on image analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels as a proxy for vehicle density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Estimate vehicle counts based on edge density
            # These are rough estimates for demonstration
            total_vehicles = int(edge_density * 1000)
            car_count = int(total_vehicles * 0.7)
            bike_count = int(total_vehicles * 0.2)
            bus_count = int(total_vehicles * 0.05)
            truck_count = int(total_vehicles * 0.05)
            
            return VehicleCounts(
                car_count=car_count,
                bike_count=bike_count,
                bus_count=bus_count,
                truck_count=truck_count,
                total_count=total_vehicles,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error estimating vehicle counts: {e}")
            return VehicleCounts(
                car_count=0,
                bike_count=0,
                bus_count=0,
                truck_count=0,
                total_count=0,
                timestamp=datetime.now()
            )
    
    def _draw_visualizations(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw all visualizations on the image.
        
        Args:
            image: Input image as numpy array.
            results: Processing results dictionary.
            
        Returns:
            Image with all visualizations drawn.
        """
        try:
            result_image = image.copy()
            
            # Draw road detection results
            if settings.output.draw_lanes and 'road_detection' in results:
                road_info = results['road_detection']
                if road_info['lanes_detected']:
                    # This would require the actual RoadInfo object
                    # For now, we'll just add text
                    cv2.putText(result_image, f"Lanes: Detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if road_info['lane_departure_warning']:
                        cv2.putText(result_image, "LANE DEPARTURE WARNING", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw traffic sign results
            if settings.output.draw_traffic_signs and 'traffic_signs' in results:
                traffic_signs = results['traffic_signs']
                cv2.putText(result_image, f"Traffic Signs: {traffic_signs['total_signs']}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw traffic flow results
            if 'traffic_flow' in results:
                traffic_flow = results['traffic_flow']
                cv2.putText(result_image, f"Traffic: {traffic_flow['prediction']['predicted_situation']}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(result_image, f"Congestion: {traffic_flow['prediction']['congestion_level']}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw road condition results
            if 'road_conditions' in results:
                road_conditions = results['road_conditions']
                cv2.putText(result_image, f"Road Quality: {road_conditions['overall_quality']}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(result_image, f"Priority: {road_conditions['maintenance_priority']}", (10, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw system stats
            if 'system_stats' in results:
                stats = results['system_stats']
                cv2.putText(result_image, f"FPS: {stats['average_fps']:.1f}", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return result_image
        except Exception as e:
            self.logger.error(f"Error drawing visualizations: {e}")
            return image
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save processing results to file.
        
        Args:
            results: Processing results dictionary.
            output_path: Output file path.
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics.
        
        Returns:
            Dictionary containing system status information.
        """
        return {
            'initialized': self.is_initialized,
            'modules': {
                'road_detector': self.road_detector is not None,
                'traffic_sign_detector': self.traffic_sign_detector is not None,
                'traffic_flow_predictor': self.traffic_flow_predictor is not None,
                'road_condition_detector': self.road_condition_detector is not None
            },
            'statistics': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0
        }
        self.logger.info("Statistics reset")


def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent Road and Traffic Monitoring System')
    parser.add_argument('--source', type=str, default='0', help='Video source (file path or camera index)')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--image', type=str, help='Process single image')
    parser.add_argument('--save-results', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize system
    system = IntelligentTrafficSystem(config)
    
    try:
        if args.image:
            # Process single image
            results = system.process_image(args.image)
            print(json.dumps(results, indent=2, default=str))
            
            if args.save_results:
                system.save_results(results, args.save_results)
        else:
            # Process video stream
            system.process_video_stream(args.source, args.output)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"System error: {e}")


if __name__ == "__main__":
    main()
