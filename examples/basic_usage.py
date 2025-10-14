"""
Basic usage example for the Intelligent Traffic System.

This script demonstrates how to use the system for processing
images and video streams.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.main import IntelligentTrafficSystem


def process_single_image_example() -> None:
    """Example of processing a single image."""
    print("=== Single Image Processing Example ===")
    
    # Initialize the system
    system = IntelligentTrafficSystem()
    
    # Create a sample image (you can replace this with a real image path)
    sample_image = create_sample_image()
    
    # Process the image
    results = system.process_image(sample_image)
    
    # Print results
    print(f"Processing time: {results['processing_time']:.3f}s")
    print(f"Road quality: {results['road_detection']['road_quality']}")
    print(f"Traffic signs detected: {results['traffic_signs']['total_signs']}")
    print(f"Traffic situation: {results['traffic_flow']['prediction']['predicted_situation']}")
    print(f"Road condition: {results['road_conditions']['overall_quality']}")
    
    # Save results
    system.save_results(results, "results/sample_image_results.json")
    print("Results saved to results/sample_image_results.json")


def process_video_example() -> None:
    """Example of processing a video stream."""
    print("=== Video Processing Example ===")
    
    # Initialize the system
    system = IntelligentTrafficSystem()
    
    # Process video (use 0 for webcam, or provide video file path)
    video_source = 0  # Change to video file path if needed
    
    try:
        system.process_video_stream(video_source, "results/processed_video.mp4")
        print("Video processing completed")
    except KeyboardInterrupt:
        print("Video processing interrupted by user")
    except Exception as e:
        print(f"Error processing video: {e}")


def create_sample_image() -> np.ndarray:
    """Create a sample image for testing."""
    # Create a sample image with some road-like features
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw road surface
    cv2.rectangle(image, (0, 300), (640, 480), (50, 50, 50), -1)
    
    # Draw lane markings
    cv2.line(image, (150, 300), (200, 480), (255, 255, 255), 3)
    cv2.line(image, (450, 300), (400, 480), (255, 255, 255), 3)
    
    # Draw a simple traffic sign (circle)
    cv2.circle(image, (500, 150), 30, (0, 0, 255), -1)
    cv2.putText(image, "STOP", (480, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add some texture to simulate road surface
    noise = np.random.randint(0, 30, (180, 640, 3), dtype=np.uint8)
    image[300:480, :] = cv2.add(image[300:480, :], noise)
    
    return image


def system_status_example() -> None:
    """Example of checking system status."""
    print("=== System Status Example ===")
    
    # Initialize the system
    system = IntelligentTrafficSystem()
    
    # Get system status
    status = system.get_system_status()
    
    print(f"System initialized: {status['initialized']}")
    print("Module status:")
    for module, is_available in status['modules'].items():
        print(f"  {module}: {'Available' if is_available else 'Not Available'}")
    
    print(f"Total frames processed: {status['statistics']['total_frames_processed']}")
    print(f"Average FPS: {status['statistics']['average_fps']:.2f}")


def main() -> None:
    """Main function to run examples."""
    print("Intelligent Traffic System - Basic Usage Examples")
    print("=" * 50)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Run examples
        process_single_image_example()
        print()
        
        system_status_example()
        print()
        
        # Uncomment the following line to run video processing
        # process_video_example()
        
        print("Examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
