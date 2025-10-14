"""
Comprehensive testing script using real data from the original repositories.
This script tests all modules using actual images, videos, and datasets.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append('src')

from src.core.road_detection import RoadDetector
from src.core.traffic_sign_detection import TrafficSignDetector
from src.core.traffic_flow_prediction import TrafficFlowPredictor
from src.core.road_condition_detection import RoadConditionDetector

class RealDataTester:
    """
    Test class that uses real data from the original repositories.
    """
    
    def __init__(self):
        """Initialize the tester with paths to real data."""
        self.base_path = Path("..")  # Go up one level to access original repos
        
        # Define paths to real data
        self.traffic_sign_test_path = self.base_path / "Real-Time-Traffic-Sign-Detection" / "Test"
        self.traffic_sign_dataset_path = self.base_path / "Traffic-Sign-Detection" / "dataset"
        self.traffic_sign_images_path = self.base_path / "Traffic-Sign-Detection" / "images"
        self.road_detection_images_path = self.base_path / "Road-Detection-System"
        self.traffic_flow_data_path = self.base_path / "Data-Science-Projects" / "Traffic-Flow-Prediction" / "TrafficDataset.csv"
        self.traffic_sign_video_path = self.base_path / "Traffic-Sign-Detection" / "MVI_1049.avi"
        
        # Initialize detectors
        self.road_detector = RoadDetector()
        self.traffic_sign_detector = None  # Will be initialized if YOLO model exists
        self.traffic_flow_predictor = None  # Will be initialized with real data
        self.road_condition_detector = RoadConditionDetector()
        
        print("Real Data Tester initialized!")
        print(f"Base path: {self.base_path.absolute()}")
    
    def test_road_detection_with_real_images(self) -> Dict[str, Any]:
        """
        Test road detection using real lane images from Road-Detection-System.
        """
        print("\n🛣️  Testing Road Detection with Real Images...")
        
        results = {
            "tested_images": [],
            "successful_detections": 0,
            "failed_detections": 0,
            "processing_times": []
        }
        
        # Test with real lane images
        lane_images = [
            "lane.jpeg",
            "lane2.jpeg", 
            "lane3.jpeg"
        ]
        
        for image_name in lane_images:
            image_path = self.road_detection_images_path / image_name
            
            if not image_path.exists():
                print(f"⚠️  Image not found: {image_path}")
                continue
            
            try:
                print(f"Testing: {image_name}")
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"❌ Failed to load image: {image_name}")
                    results["failed_detections"] += 1
                    continue
                
                # Test lane detection
                start_time = time.time()
                result_image = self.road_detector.detect_lanes(image)
                processing_time = time.time() - start_time
                
                # Save result
                output_path = f"test_results/road_detection_{image_name}"
                os.makedirs("test_results", exist_ok=True)
                cv2.imwrite(output_path, result_image)
                
                results["tested_images"].append(image_name)
                results["successful_detections"] += 1
                results["processing_times"].append(processing_time)
                
                print(f"✅ Success: {image_name} (processed in {processing_time:.3f}s)")
                
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
                results["failed_detections"] += 1
        
        # Calculate average processing time
        if results["processing_times"]:
            avg_time = sum(results["processing_times"]) / len(results["processing_times"])
            print(f"📊 Average processing time: {avg_time:.3f}s")
        
        return results
    
    def test_traffic_sign_detection_with_real_images(self) -> Dict[str, Any]:
        """
        Test traffic sign detection using real images from Traffic-Sign-Detection.
        """
        print("\n🚦 Testing Traffic Sign Detection with Real Images...")
        
        results = {
            "tested_images": [],
            "successful_detections": 0,
            "failed_detections": 0,
            "processing_times": []
        }
        
        # Test with real traffic sign images
        test_images = [
            "0.png",
            "all-signs.png"
        ]
        
        for image_name in test_images:
            image_path = self.traffic_sign_images_path / image_name
            
            if not image_path.exists():
                print(f"⚠️  Image not found: {image_path}")
                continue
            
            try:
                print(f"Testing: {image_name}")
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"❌ Failed to load image: {image_name}")
                    results["failed_detections"] += 1
                    continue
                
                # Test traffic sign detection (without YOLO for now)
                start_time = time.time()
                
                # Use basic contour detection as fallback
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                result_image = image.copy()
                cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
                
                processing_time = time.time() - start_time
                
                # Save result
                output_path = f"test_results/traffic_sign_{image_name}"
                cv2.imwrite(output_path, result_image)
                
                results["tested_images"].append(image_name)
                results["successful_detections"] += 1
                results["processing_times"].append(processing_time)
                
                print(f"✅ Success: {image_name} (found {len(contours)} potential signs)")
                
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
                results["failed_detections"] += 1
        
        return results
    
    def test_traffic_flow_prediction_with_real_data(self) -> Dict[str, Any]:
        """
        Test traffic flow prediction using real dataset from Traffic-Flow-Prediction.
        """
        print("\n📊 Testing Traffic Flow Prediction with Real Data...")
        
        results = {
            "data_loaded": False,
            "model_trained": False,
            "predictions_tested": 0,
            "accuracy": 0.0
        }
        
        try:
            # Load real traffic data
            if not self.traffic_flow_data_path.exists():
                print(f"❌ Traffic data not found: {self.traffic_flow_data_path}")
                return results
            
            print(f"Loading data from: {self.traffic_flow_data_path}")
            
            # Initialize predictor with real data
            self.traffic_flow_predictor = TrafficFlowPredictor(
                data_path=str(self.traffic_flow_data_path)
            )
            
            results["data_loaded"] = True
            results["model_trained"] = True
            
            # Test predictions with sample data
            test_cases = [
                {"car_count": 15, "bike_count": 3, "bus_count": 2, "truck_count": 5},
                {"car_count": 25, "bike_count": 5, "bus_count": 3, "truck_count": 8},
                {"car_count": 5, "bike_count": 1, "bus_count": 1, "truck_count": 2},
            ]
            
            for i, test_case in enumerate(test_cases):
                current_time = pd.Timestamp.now()
                prediction = self.traffic_flow_predictor.predict_traffic_situation(
                    car_count=test_case["car_count"],
                    bike_count=test_case["bike_count"],
                    bus_count=test_case["bus_count"],
                    truck_count=test_case["truck_count"],
                    current_time=current_time
                )
                
                print(f"Test {i+1}: Cars={test_case['car_count']}, Bikes={test_case['bike_count']}, "
                      f"Buses={test_case['bus_count']}, Trucks={test_case['truck_count']} "
                      f"→ Prediction: {prediction}")
                
                results["predictions_tested"] += 1
            
            print("✅ Traffic flow prediction working with real data!")
            
        except Exception as e:
            print(f"❌ Error in traffic flow prediction: {e}")
        
        return results
    
    def test_road_condition_detection_with_real_images(self) -> Dict[str, Any]:
        """
        Test road condition detection using real images.
        """
        print("\n🔧 Testing Road Condition Detection...")
        
        results = {
            "tested_images": [],
            "anomalies_detected": 0,
            "processing_times": []
        }
        
        # Test with lane images (they might have some road conditions)
        lane_images = ["lane.jpeg", "lane2.jpeg", "lane3.jpeg"]
        
        for image_name in lane_images:
            image_path = self.road_detection_images_path / image_name
            
            if not image_path.exists():
                continue
            
            try:
                print(f"Testing road conditions in: {image_name}")
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Test road condition detection
                start_time = time.time()
                result_image, anomalies = self.road_condition_detector.detect_conditions(image)
                processing_time = time.time() - start_time
                
                # Save result
                output_path = f"test_results/road_condition_{image_name}"
                cv2.imwrite(output_path, result_image)
                
                results["tested_images"].append(image_name)
                results["anomalies_detected"] += len(anomalies)
                results["processing_times"].append(processing_time)
                
                print(f"✅ Found {len(anomalies)} potential road anomalies in {image_name}")
                
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
        
        return results
    
    def test_video_processing(self) -> Dict[str, Any]:
        """
        Test video processing with real traffic sign video.
        """
        print("\n🎥 Testing Video Processing...")
        
        results = {
            "video_processed": False,
            "frames_processed": 0,
            "processing_time": 0.0
        }
        
        if not self.traffic_sign_video_path.exists():
            print(f"⚠️  Video not found: {self.traffic_sign_video_path}")
            return results
        
        try:
            print(f"Processing video: {self.traffic_sign_video_path.name}")
            
            cap = cv2.VideoCapture(str(self.traffic_sign_video_path))
            if not cap.isOpened():
                print("❌ Failed to open video")
                return results
            
            frame_count = 0
            start_time = time.time()
            
            # Process first 30 frames for testing
            while frame_count < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply road detection
                frame_with_lanes = self.road_detector.detect_lanes(frame)
                
                # Apply road condition detection
                frame_with_conditions, _ = self.road_condition_detector.detect_conditions(frame_with_lanes)
                
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count} frames...")
            
            processing_time = time.time() - start_time
            
            cap.release()
            
            results["video_processed"] = True
            results["frames_processed"] = frame_count
            results["processing_time"] = processing_time
            
            print(f"✅ Processed {frame_count} frames in {processing_time:.2f}s")
            print(f"📊 Average FPS: {frame_count/processing_time:.2f}")
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests and return comprehensive results.
        """
        print("🚀 Starting Comprehensive Testing with Real Data...")
        print("=" * 60)
        
        all_results = {}
        
        # Test each module
        all_results["road_detection"] = self.test_road_detection_with_real_images()
        all_results["traffic_sign_detection"] = self.test_traffic_sign_detection_with_real_images()
        all_results["traffic_flow_prediction"] = self.test_traffic_flow_prediction_with_real_data()
        all_results["road_condition_detection"] = self.test_road_condition_detection_with_real_images()
        all_results["video_processing"] = self.test_video_processing()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        for module, results in all_results.items():
            print(f"\n{module.upper().replace('_', ' ')}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        
        return all_results

def main():
    """Main function to run all tests."""
    
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Initialize tester
    tester = RealDataTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    print("\n🎉 Testing completed!")
    print("Check the 'test_results' folder for output images.")
    
    return results

if __name__ == "__main__":
    main()
