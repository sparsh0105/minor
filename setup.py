"""
Setup script for the Intelligent Traffic System.

This script handles the installation and setup of the system,
including downloading required models and setting up the environment.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True


def install_dependencies() -> bool:
    """Install required dependencies."""
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def create_directories() -> None:
    """Create necessary directories."""
    directories = [
        "data/models",
        "data/datasets", 
        "data/sample_videos",
        "results",
        "logs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_yolo_model() -> bool:
    """Download YOLOv5 model if not present."""
    model_path = Path("data/models/yolov5s.pt")
    
    if model_path.exists():
        logger.info("YOLO model already exists")
        return True
    
    try:
        logger.info("Downloading YOLOv5 model...")
        url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
        urllib.request.urlretrieve(url, model_path)
        logger.info("YOLO model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download YOLO model: {e}")
        return False


def create_sample_data() -> None:
    """Create sample data files for testing."""
    # Create a sample traffic dataset
    sample_data = {
        "Time": ["12:00:00 AM", "12:15:00 AM", "12:30:00 AM"],
        "Date": ["10-10-2023", "10-10-2023", "10-10-2023"],
        "Day of the week": ["Tuesday", "Tuesday", "Tuesday"],
        "CarCount": [13, 14, 10],
        "BikeCount": [2, 1, 2],
        "BusCount": [2, 1, 2],
        "TruckCount": [24, 36, 32],
        "Total": [41, 52, 46],
        "Traffic Situation": ["normal", "normal", "normal"]
    }
    
    import pandas as pd
    df = pd.DataFrame(sample_data)
    df.to_csv("data/datasets/sample_traffic_data.csv", index=False)
    logger.info("Created sample traffic dataset")


def run_tests() -> bool:
    """Run the test suite."""
    try:
        logger.info("Running tests...")
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
        logger.info("All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {e}")
        return False


def create_config_files() -> None:
    """Create default configuration files."""
    # Create a simple config file if it doesn't exist
    config_path = Path("config/local_config.yaml")
    if not config_path.exists():
        config_content = """# Local Configuration Override
# This file can be used to override default settings

# Example overrides:
# yolo:
#   confidence_threshold: 0.6
# 
# lane_detection:
#   roi_ratio: 0.7
# 
# output:
#   show_visualization: true
#   save_results: true
"""
        config_path.write_text(config_content)
        logger.info("Created local configuration file")


def main() -> None:
    """Main setup function."""
    logger.info("Starting Intelligent Traffic System setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Setup failed during dependency installation")
        sys.exit(1)
    
    # Download models
    if not download_yolo_model():
        logger.warning("YOLO model download failed, but setup can continue")
    
    # Create sample data
    create_sample_data()
    
    # Create config files
    create_config_files()
    
    # Run tests
    if not run_tests():
        logger.warning("Some tests failed, but setup can continue")
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the system using:")
    logger.info("  python examples/basic_usage.py")
    logger.info("  python -m src.main --help")


if __name__ == "__main__":
    main()
