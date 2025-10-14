# Intelligent Road and Traffic Monitoring System

## Overview

This project integrates multiple computer vision and machine learning approaches to create a comprehensive intelligent transportation system. The system combines four core modules to enhance road safety and traffic management:

1. **Road Detection** - Lane detection and road boundary identification
2. **Traffic Sign Detection** - Real-time traffic sign recognition and classification
3. **Traffic Flow Prediction** - Vehicle density and congestion forecasting
4. **Road Condition Detection** - Pothole, crack, and surface damage detection

## Project Structure

```
intelligent_traffic_system/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── road_detection.py
│   │   ├── traffic_sign_detection.py
│   │   ├── traffic_flow_prediction.py
│   │   └── road_condition_detection.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_models.py
│   │   └── ml_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processing.py
│   │   ├── data_preprocessing.py
│   │   └── visualization.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_road_detection.py
│   ├── test_traffic_sign_detection.py
│   ├── test_traffic_flow_prediction.py
│   └── test_road_condition_detection.py
├── docs/
│   ├── api_documentation.md
│   └── user_guide.md
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── model_config.yaml
├── data/
│   ├── models/
│   ├── datasets/
│   └── sample_videos/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Features

### Road Detection Module
- Real-time lane detection using OpenCV
- Lane departure warning system
- Road boundary identification
- Support for various road conditions

### Traffic Sign Detection Module
- YOLOv5-based real-time detection (91.75% mAP@0.5)
- Support for 61 different traffic sign classes
- Traditional computer vision fallback with SVM
- 45 fps processing speed

### Traffic Flow Prediction Module
- Random Forest Classifier for traffic prediction
- 95% accuracy on traffic situation classification
- Real-time vehicle counting and density analysis
- Congestion level prediction (low, normal, high)

### Road Condition Detection Module
- Pothole and crack detection using computer vision
- Surface quality assessment
- Damage severity classification
- Road maintenance alerts

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd intelligent_traffic_system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from src.main import IntelligentTrafficSystem

# Initialize the system
system = IntelligentTrafficSystem()

# Process video stream
system.process_video_stream("path/to/video.mp4")

# Process single image
results = system.process_image("path/to/image.jpg")
```

### Advanced Usage
```python
from src.core.road_detection import RoadDetector
from src.core.traffic_sign_detection import TrafficSignDetector

# Initialize individual modules
road_detector = RoadDetector()
sign_detector = TrafficSignDetector()

# Process with specific modules
lane_info = road_detector.detect_lanes(image)
signs = sign_detector.detect_signs(image)
```

## Configuration

Edit `config/settings.py` to customize:
- Model paths and parameters
- Detection thresholds
- Output formats
- Logging levels

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_road_detection.py -v
```

## Performance

- **Traffic Sign Detection**: 45 fps on GeForce MX 250
- **Lane Detection**: Real-time processing
- **Traffic Flow Prediction**: 95% accuracy
- **Overall System**: Optimized for real-time applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 implementation from Ultralytics
- OpenCV for computer vision operations
- Scikit-learn for machine learning models
- Original repositories for individual modules
