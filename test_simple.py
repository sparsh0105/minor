#!/usr/bin/env python3
"""
Simple standalone test for traffic sign detection.
Run with: python test_simple.py
"""

import sys
import os
import numpy as np
import cv2

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.traffic_sign_detection import TrafficSignDetector, TrafficSign, TrafficSignDetectionResult

def create_test_image():
    """Create a simple test image with traffic signs."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a red circular sign (stop sign)
    cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)
    cv2.putText(image, "STOP", (290, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw a blue rectangular sign (speed limit)
    cv2.rectangle(image, (100, 100), (200, 150), (255, 0, 0), -1)
    cv2.putText(image, "50", (130, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return image

def test_traffic_sign_detector():
    """Test the traffic sign detector."""
    print("🚦 Testing Traffic Sign Detector...")
    
    # Create detector
    detector = TrafficSignDetector()
    print("✅ Detector created successfully")
    
    # Create test image
    test_image = create_test_image()
    print("✅ Test image created")
    
    # Test traditional CV detection
    print("🔍 Testing traditional CV detection...")
    result = detector.detect_signs(test_image, use_yolo=False)
    
    print(f"📊 Results:")
    print(f"   - Total signs detected: {result.total_signs}")
    print(f"   - Method used: {result.method_used}")
    print(f"   - Processing time: {result.processing_time:.3f}s")
    
    if result.signs:
        print("🚦 Detected signs:")
        for i, sign in enumerate(result.signs):
            print(f"   {i+1}. {sign.class_name} (confidence: {sign.confidence:.2f})")
            print(f"      Bbox: {sign.bbox}")
            print(f"      Center: {sign.center}")
    else:
        print("   No signs detected")
    
    # Test drawing detections
    print("🎨 Testing detection drawing...")
    result_image = detector.draw_detections(test_image, result)
    print("✅ Detection drawing completed")
    
    # Save result image
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"💾 Result saved to: {output_path}")
    
    return True

def test_data_classes():
    """Test the data classes."""
    print("\n📋 Testing Data Classes...")
    
    # Test TrafficSign
    sign = TrafficSign(
        class_id=0,
        class_name="stop",
        confidence=0.95,
        bbox=(100, 100, 200, 150),
        center=(150, 125),
        area=5000,
        method="traditional"
    )
    print("✅ TrafficSign created successfully")
    print(f"   - Class: {sign.class_name}")
    print(f"   - Confidence: {sign.confidence}")
    print(f"   - Method: {sign.method}")
    
    # Test TrafficSignDetectionResult
    result = TrafficSignDetectionResult(
        signs=[sign],
        total_signs=1,
        processing_time=0.1,
        method_used="traditional"
    )
    print("✅ TrafficSignDetectionResult created successfully")
    print(f"   - Total signs: {result.total_signs}")
    print(f"   - Method used: {result.method_used}")
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("🚦 TRAFFIC SIGN DETECTION SIMPLE TEST")
    print("=" * 60)
    
    try:
        # Test data classes
        test_data_classes()
        
        # Test detector
        test_traffic_sign_detector()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
