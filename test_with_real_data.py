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
        """Initialize the tester with paths to local data."""
        self.base_path = Path(".")  # Current directory
        
        # Define paths to local data
        self.traffic_sign_test_path = self.base_path / "data" / "test_images"
        self.traffic_sign_dataset_path = self.base_path / "data" / "test_images" / "traffic_sign_dataset"
        self.traffic_sign_images_path = self.base_path / "data" / "test_images"
        self.road_detection_images_path = self.base_path / "data" / "test_images"
        self.traffic_flow_data_path = self.base_path / "data" / "datasets" / "TrafficDataset.csv"
        self.traffic_sign_video_path = self.base_path / "data" / "test_videos" / "MVI_1049.avi"
        
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
            "lane3.jpeg",
            "lane4.jpeg"
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
                lane_info = self.road_detector.detect_lanes(image)
                processing_time = time.time() - start_time
                
                # Draw lanes on image
                result_image = self.road_detector.draw_lanes(image, lane_info)
                
                # Save result with better naming
                base_name = image_name.split('.')[0]  # Remove extension
                output_path = f"test_results/{base_name}_lane_detection_result.jpeg"
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
            "processing_times": [],
            "yolo_detections": 0,
            "traditional_detections": 0
        }
        
        # Test with real traffic sign images
        test_images = [
            "0.png",
            "all-signs.png",
            "Sample_2.png",
            "Sample_3.png"
        ]
        
        # Try to initialize YOLO detector
        yolo_detector = None
        try:
            # Add the YOLO detection path
            import sys
            sys.path.append('.')
            from test_yolo_detection import YOLOTrafficSignDetector
            
            print("🔄 Initializing YOLO detector...")
            yolo_detector = YOLOTrafficSignDetector()
            if yolo_detector.model is not None:
                print("✅ YOLO detector initialized successfully")
                print(f"   - Model classes: {len(yolo_detector.names)}")
                print(f"   - Device: {yolo_detector.device}")
            else:
                print("⚠️  YOLO detector failed to initialize")
                yolo_detector = None
        except Exception as e:
            print(f"⚠️  YOLO detector not available: {e}")
            yolo_detector = None
        
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
                
                start_time = time.time()
                total_detections = 0
                result_image = image.copy()
                
                # Try YOLO detection first
                if yolo_detector is not None:
                    try:
                        yolo_detections = yolo_detector.detect_signs(image)
                        if yolo_detections:
                            result_image = yolo_detector.draw_detections(image, yolo_detections)
                            total_detections += len(yolo_detections)
                            results["yolo_detections"] += len(yolo_detections)
                            print(f"   🎯 YOLO detected {len(yolo_detections)} signs:")
                            for i, det in enumerate(yolo_detections[:5]):  # Show first 5
                                print(f"      {i+1}. {det['class_name']} ({det['confidence']:.2f})")
                            if len(yolo_detections) > 5:
                                print(f"      ... and {len(yolo_detections) - 5} more")
                    except Exception as e:
                        print(f"   ⚠️  YOLO detection failed: {e}")
                
                # Fallback to traditional CV if no YOLO detections
                if total_detections == 0:
                    # Use basic contour detection as fallback
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw contours
                    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
                    total_detections = len(contours)
                    results["traditional_detections"] += len(contours)
                    print(f"   🔍 Traditional CV found {len(contours)} potential signs")
                
                processing_time = time.time() - start_time
                
                # Save result with better naming
                base_name = image_name.split('.')[0]  # Remove extension
                method = "yolo" if results["yolo_detections"] > 0 else "traditional"
                output_path = f"test_results/traffic_sign_detection_{base_name}_{method}.png"
                cv2.imwrite(output_path, result_image)
                
                results["tested_images"].append(image_name)
                results["successful_detections"] += 1
                results["processing_times"].append(processing_time)
                
                print(f"✅ Success: {image_name} (found {total_detections} signs)")
                
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
            
            # Load and preprocess the data
            import pandas as pd
            from datetime import datetime
            
            data = pd.read_csv(self.traffic_flow_data_path)
            print(f"Loaded {len(data)} records from dataset")
            
            # Preprocess the data to match expected format
            # Convert column names to match expected format
            data_processed = data.copy()
            data_processed = data_processed.rename(columns={
                'CarCount': 'car_count',
                'BikeCount': 'bike_count', 
                'BusCount': 'bus_count',
                'TruckCount': 'truck_count',
                'Total': 'total_count',
                'Traffic Situation': 'traffic_situation'
            })
            
            # Extract temporal features from Time and Date columns
            data_processed['datetime'] = pd.to_datetime(data_processed['Date'] + ' ' + data_processed['Time'])
            data_processed['hour'] = data_processed['datetime'].dt.hour
            data_processed['day_of_week'] = data_processed['datetime'].dt.dayofweek
            data_processed['is_weekend'] = (data_processed['day_of_week'] >= 5).astype(int)
            data_processed['is_peak_hour'] = ((data_processed['hour'] >= 7) & (data_processed['hour'] <= 9) | 
                                            (data_processed['hour'] >= 17) & (data_processed['hour'] <= 19)).astype(int)
            
            # Initialize predictor
            self.traffic_flow_predictor = TrafficFlowPredictor()
            
            # Train the model with real data
            print("Training model with real traffic data...")
            train_results = self.traffic_flow_predictor.train_model(data_processed)
            
            results["data_loaded"] = True
            results["model_trained"] = True
            results["accuracy"] = train_results.get('accuracy', 0.0)
            
            print(f"✅ Model trained with accuracy: {results['accuracy']:.3f}")
            
            # Test predictions with sample data
            test_cases = [
                {"car_count": 15, "bike_count": 3, "bus_count": 2, "truck_count": 5},
                {"car_count": 25, "bike_count": 5, "bus_count": 3, "truck_count": 8},
                {"car_count": 5, "bike_count": 1, "bus_count": 1, "truck_count": 2},
            ]
            
            # Store detailed prediction results
            prediction_results = []
            
            for i, test_case in enumerate(test_cases):
                from src.core.traffic_flow_prediction import VehicleCounts
                from datetime import datetime
                
                # Create VehicleCounts object
                vehicle_counts = VehicleCounts(
                    car_count=test_case["car_count"],
                    bike_count=test_case["bike_count"],
                    bus_count=test_case["bus_count"],
                    truck_count=test_case["truck_count"],
                    total_count=sum(test_case.values()),
                    timestamp=datetime.now()
                )
                
                prediction = self.traffic_flow_predictor.predict_traffic_situation(vehicle_counts)
                
                # Store detailed result
                prediction_result = {
                    "test_case": i + 1,
                    "input": test_case,
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat()
                }
                prediction_results.append(prediction_result)
                
                # Display formatted result
                self._display_prediction_result(i + 1, test_case, prediction)
                
                results["predictions_tested"] += 1
            
            # Store results to file
            self._save_prediction_results(prediction_results, train_results)
            
            print("✅ Traffic flow prediction working with real data!")
            
        except Exception as e:
            print(f"❌ Error in traffic flow prediction: {e}")
        
        return results
    
    def _display_prediction_result(self, test_num: int, test_case: Dict[str, int], prediction) -> None:
        """
        Display a formatted traffic flow prediction result.
        
        Args:
            test_num: Test case number.
            test_case: Input vehicle counts.
            prediction: Prediction result object.
        """
        print(f"\n{'='*60}")
        print(f"🚦 TRAFFIC FLOW PREDICTION - TEST {test_num}")
        print(f"{'='*60}")
        
        # Input data
        print(f"📊 INPUT DATA:")
        print(f"   🚗 Cars:     {test_case['car_count']:3d}")
        print(f"   🏍️  Bikes:    {test_case['bike_count']:3d}")
        print(f"   🚌 Buses:    {test_case['bus_count']:3d}")
        print(f"   🚛 Trucks:   {test_case['truck_count']:3d}")
        print(f"   📈 Total:    {sum(test_case.values()):3d}")
        
        # Prediction results
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   🚦 Traffic Situation: {prediction.predicted_situation.upper()}")
        print(f"   📊 Confidence:        {prediction.confidence:.1%}")
        print(f"   🚧 Congestion Level:  {prediction.congestion_level.upper()}")
        
        # Future projection
        future = prediction.predicted_vehicle_counts
        print(f"\n🔮 FUTURE PROJECTION (15 min ahead):")
        print(f"   🚗 Cars:     {future.car_count:3d}")
        print(f"   🏍️  Bikes:    {future.bike_count:3d}")
        print(f"   🚌 Buses:    {future.bus_count:3d}")
        print(f"   🚛 Trucks:   {future.truck_count:3d}")
        print(f"   📈 Total:    {future.total_count:3d}")
        
        # Features used
        if prediction.features_used:
            print(f"\n🔧 FEATURES USED:")
            print(f"   {', '.join(prediction.features_used)}")
        
        print(f"{'='*60}")
    
    def _save_prediction_results(self, prediction_results: List[Dict], train_results: Dict) -> None:
        """
        Save detailed prediction results to files.
        
        Args:
            prediction_results: List of prediction results.
            train_results: Training results dictionary.
        """
        import json
        import csv
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON results
        json_file = results_dir / f"traffic_flow_predictions_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model_accuracy": train_results.get('accuracy', 0.0),
                "training_results": train_results,
                "predictions": prediction_results
            }, f, indent=2, default=str)
        
        # Save CSV summary
        csv_file = results_dir / f"traffic_flow_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Test_Case', 'Input_Cars', 'Input_Bikes', 'Input_Buses', 'Input_Trucks', 'Input_Total',
                'Predicted_Situation', 'Confidence', 'Congestion_Level',
                'Future_Cars', 'Future_Bikes', 'Future_Buses', 'Future_Trucks', 'Future_Total',
                'Timestamp'
            ])
            
            for result in prediction_results:
                pred = result['prediction']
                future = pred.predicted_vehicle_counts
                writer.writerow([
                    result['test_case'],
                    result['input']['car_count'],
                    result['input']['bike_count'],
                    result['input']['bus_count'],
                    result['input']['truck_count'],
                    sum(result['input'].values()),
                    pred.predicted_situation,
                    f"{pred.confidence:.3f}",
                    pred.congestion_level,
                    future.car_count,
                    future.bike_count,
                    future.bus_count,
                    future.truck_count,
                    future.total_count,
                    result['timestamp']
                ])
        
        print(f"\n💾 RESULTS SAVED:")
        print(f"   📄 Detailed JSON: {json_file}")
        print(f"   📊 CSV Summary:   {csv_file}")
        
        # Create summary report
        self._create_summary_report(prediction_results, train_results, timestamp)
    
    def _create_summary_report(self, prediction_results: List[Dict], train_results: Dict, timestamp: str) -> None:
        """
        Create a comprehensive summary report with visualizations.
        
        Args:
            prediction_results: List of prediction results.
            train_results: Training results dictionary.
            timestamp: Timestamp for file naming.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Traffic Flow Prediction Analysis Report', fontsize=16, fontweight='bold')
            
            # 1. Input vs Predicted Vehicle Counts
            ax1 = axes[0, 0]
            test_cases = [f"Test {r['test_case']}" for r in prediction_results]
            input_totals = [sum(r['input'].values()) for r in prediction_results]
            predicted_totals = [r['prediction'].predicted_vehicle_counts.total_count for r in prediction_results]
            
            x = range(len(test_cases))
            width = 0.35
            
            ax1.bar([i - width/2 for i in x], input_totals, width, label='Input Total', alpha=0.8)
            ax1.bar([i + width/2 for i in x], predicted_totals, width, label='Predicted Total', alpha=0.8)
            ax1.set_xlabel('Test Cases')
            ax1.set_ylabel('Vehicle Count')
            ax1.set_title('Input vs Predicted Vehicle Counts')
            ax1.set_xticks(x)
            ax1.set_xticklabels(test_cases)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Confidence Scores
            ax2 = axes[0, 1]
            confidences = [r['prediction'].confidence for r in prediction_results]
            colors = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' for c in confidences]
            
            bars = ax2.bar(test_cases, confidences, color=colors, alpha=0.7)
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Prediction Confidence Scores')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{conf:.1%}', ha='center', va='bottom')
            
            # 3. Traffic Situation Distribution
            ax3 = axes[1, 0]
            situations = [r['prediction'].predicted_situation for r in prediction_results]
            situation_counts = {}
            for situation in situations:
                situation_counts[situation] = situation_counts.get(situation, 0) + 1
            
            if situation_counts:
                ax3.pie(situation_counts.values(), labels=situation_counts.keys(), autopct='%1.1f%%', startangle=90)
                ax3.set_title('Traffic Situation Distribution')
            
            # 4. Model Performance Metrics
            ax4 = axes[1, 1]
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [
                train_results.get('accuracy', 0),
                train_results.get('classification_report', {}).get('weighted avg', {}).get('precision', 0),
                train_results.get('classification_report', {}).get('weighted avg', {}).get('recall', 0),
                train_results.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0)
            ]
            
            bars = ax4.bar(metrics, values, color=['#2E8B57', '#4169E1', '#FF6347', '#FFD700'], alpha=0.7)
            ax4.set_ylabel('Score')
            ax4.set_title('Model Performance Metrics')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save the plot
            results_dir = Path("test_results")
            plot_file = results_dir / f"traffic_flow_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📈 Analysis Plot:   {plot_file}")
            
        except ImportError:
            print(f"   ⚠️  Matplotlib/Seaborn not available - skipping visualization")
        except Exception as e:
            print(f"   ⚠️  Error creating visualization: {e}")
    
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
                result = self.road_condition_detector.detect_road_conditions(image)
                processing_time = time.time() - start_time
                
                # Draw detections on image
                result_image = self.road_condition_detector.draw_detections(image, result)
                anomalies = result.potholes + result.cracks + result.surface_defects
                
                # Save result with better naming
                base_name = image_name.split('.')[0]  # Remove extension
                output_path = f"test_results/{base_name}_road_condition_analysis.jpeg"
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
                lane_info = self.road_detector.detect_lanes(frame)
                frame_with_lanes = self.road_detector.draw_lanes(frame, lane_info)
                
                # Apply road condition detection
                condition_result = self.road_condition_detector.detect_road_conditions(frame)
                frame_with_conditions = self.road_condition_detector.draw_detections(frame, condition_result)
                
                # Save every 5th frame for video results
                if frame_count % 5 == 0:
                    output_path = f"test_results/video_frame_{frame_count:03d}_combined_analysis.jpeg"
                    cv2.imwrite(output_path, frame_with_conditions)
                
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
            if module == "traffic_flow_prediction":
                # Enhanced display for traffic flow prediction
                print(f"  📊 Model Accuracy:     {results.get('accuracy', 0):.1%}")
                print(f"  🎯 Predictions Tested: {results.get('predictions_tested', 0)}")
                print(f"  📈 Data Loaded:        {'✅' if results.get('data_loaded', False) else '❌'}")
                print(f"  🤖 Model Trained:      {'✅' if results.get('model_trained', False) else '❌'}")
            else:
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
