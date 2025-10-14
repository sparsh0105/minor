# 🚦 Intelligent Traffic System - Testing Guide

## 📋 Overview

This guide provides comprehensive instructions for testing the **Intelligent Road and Traffic Monitoring System**. The system integrates four core modules:

1. **Road Detection** - Lane detection and road boundary identification
2. **Traffic Sign Detection** - Real-time traffic sign recognition and classification
3. **Traffic Flow Prediction** - Vehicle density and congestion trend forecasting
4. **Road Condition Detection** - Pothole, crack, and surface damage identification

## 🛠️ Prerequisites

### 1. Environment Setup
```bash
# Navigate to project directory
cd intelligent_traffic_system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Model Setup
```bash
# Extract YOLO model (if not already done)
cd data/models
unzip Model.zip  # On Windows: use 7-Zip or WinRAR
cd ../..
```

## 🧪 Testing Methods

### Method 1: Quick System Test
Test all modules with sample data:

```bash
python test_with_real_data.py
```

**Expected Output:**
- ✅ Road detection results on lane images
- ✅ Traffic sign detection on test images
- ✅ Traffic flow prediction using CSV data
- ✅ Road condition detection on sample images

### Method 2: Individual Module Testing

#### 2.1 Test Road Detection
```bash
python -c "
from src.core.road_detection import RoadDetector
detector = RoadDetector()
result = detector.detect_lanes('data/test_images/lane.jpeg')
print('Road detection result:', result)
"
```

#### 2.2 Test Traffic Sign Detection
```bash
python -c "
from src.core.traffic_sign_detection import TrafficSignDetector
detector = TrafficSignDetector()
result = detector.detect_signs('data/test_images/0.png')
print('Traffic sign detection result:', result)
"
```

#### 2.3 Test Traffic Flow Prediction
```bash
python -c "
from src.core.traffic_flow_prediction import TrafficFlowPredictor
predictor = TrafficFlowPredictor()
result = predictor.predict_flow('data/datasets/TrafficDataset.csv')
print('Traffic flow prediction result:', result)
"
```

#### 2.4 Test Road Condition Detection
```bash
python -c "
from src.core.road_condition_detection import RoadConditionDetector
detector = RoadConditionDetector()
result = detector.detect_conditions('data/test_images/lane.jpeg')
print('Road condition detection result:', result)
"
```

### Method 3: Unit Testing with pytest
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_road_detection.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Method 4: Integration Testing
```bash
# Test the unified system
python src/main.py --input data/test_images/lane.jpeg --output results/

# Test with video input
python src/main.py --input data/test_videos/MVI_1049.avi --output results/
```

## 📊 Test Data Available

### Images
- **Road Detection**: `lane.jpeg`, `lane2.jpeg`, `lane3.jpeg`
- **Traffic Signs**: `0.png`, `10.png`, `all-signs.png`, `test1.jpeg`
- **Traffic Sign Dataset**: `traffic_sign_dataset/` (multiple classes)

### Videos
- **Traffic Sign Video**: `MVI_1049.avi`

### Datasets
- **Traffic Flow**: `TrafficDataset.csv`

### Models
- **YOLO Model**: `Model.zip` (extracted)
- **SVM Model**: `data_svm.dat`

## 🎯 Expected Test Results

### 1. Road Detection
- **Input**: Lane images
- **Output**: Detected lane lines, road boundaries
- **Success Criteria**: Clear lane detection with proper line fitting

### 2. Traffic Sign Detection
- **Input**: Traffic sign images
- **Output**: Bounding boxes, confidence scores, classifications
- **Success Criteria**: Accurate sign detection and classification

### 3. Traffic Flow Prediction
- **Input**: Traffic dataset CSV
- **Output**: Flow predictions, congestion levels
- **Success Criteria**: Reasonable predictions based on historical data

### 4. Road Condition Detection
- **Input**: Road surface images
- **Output**: Condition assessment, damage detection
- **Success Criteria**: Identification of road surface issues

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Check if models are properly extracted
ls data/models/
# Should show: Model/ (folder) and data_svm.dat
```

#### 2. Import Errors
```bash
# Ensure package is installed
pip install -e .
# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 3. Missing Dependencies
```bash
# Install missing packages
pip install opencv-python torch torchvision scikit-learn
```

#### 4. CUDA/GPU Issues
```bash
# For CPU-only mode, modify model loading in traffic_sign_detection.py
# Change device='cuda' to device='cpu'
```

## 📈 Performance Testing

### 1. Speed Test
```bash
# Test processing speed
python -c "
import time
from src.core.road_detection import RoadDetector
detector = RoadDetector()
start = time.time()
result = detector.detect_lanes('data/test_images/lane.jpeg')
print(f'Processing time: {time.time() - start:.2f} seconds')
"
```

### 2. Memory Usage
```bash
# Monitor memory usage
python -c "
import psutil
import os
from src.core.traffic_sign_detection import TrafficSignDetector
detector = TrafficSignDetector()
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

## 🎥 Video Testing

### Test with Video Input
```bash
# Process video file
python src/main.py --input data/test_videos/MVI_1049.avi --output results/ --save-video
```

### Real-time Testing (if webcam available)
```bash
# Test with webcam (requires camera)
python src/main.py --input 0 --output results/ --real-time
```

## 📝 Test Report Generation

### Generate Test Report
```bash
# Run comprehensive tests and generate report
python test_with_real_data.py --report results/test_report.html
```

### View Results
```bash
# Check output directory
ls results/
# Should contain: processed images, videos, and reports
```

## ✅ Success Checklist

- [ ] All dependencies installed
- [ ] Models extracted and loaded
- [ ] Road detection working on sample images
- [ ] Traffic sign detection working on test images
- [ ] Traffic flow prediction working with CSV data
- [ ] Road condition detection working on sample images
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Video processing working
- [ ] Results saved to output directory

## 🚀 Next Steps

After successful testing:

1. **Deploy the system** for real-world testing
2. **Fine-tune models** based on your specific use case
3. **Add more test data** for better validation
4. **Implement API endpoints** for web integration
5. **Set up monitoring** for production deployment

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all test data is properly copied
3. Ensure all dependencies are installed
4. Check the logs for specific error messages
5. Test individual modules to isolate issues

---

**Happy Testing! 🎉**

*This system represents a comprehensive solution for intelligent traffic monitoring, combining computer vision, machine learning, and data analysis techniques.*
