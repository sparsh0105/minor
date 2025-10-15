#!/usr/bin/env python3
"""
Test traffic sign detection on real traffic signs image.
Run with: python test_real_signs.py
"""

import sys
import os
import numpy as np
import cv2

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.traffic_sign_detection import TrafficSignDetector, TrafficSign, TrafficSignDetectionResult

def test_real_traffic_signs():
    """Test traffic sign detection on real image."""
    print("🚦 Testing Traffic Sign Detection on Real Image...")
    
    # Load the real traffic signs image
    image_path = "data/test_images/all-signs.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    print(f"✅ Loaded image: {image_path}")
    print(f"   - Image shape: {image.shape}")
    print(f"   - Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Create detector
    detector = TrafficSignDetector()
    print("✅ Detector created successfully")
    
    # Test with YOLO (if available)
    print("\n🔍 Testing YOLO detection...")
    yolo_result = detector.detect_signs(image, use_yolo=True)
    
    print(f"📊 YOLO Results:")
    print(f"   - Total signs detected: {yolo_result.total_signs}")
    print(f"   - Method used: {yolo_result.method_used}")
    print(f"   - Processing time: {yolo_result.processing_time:.3f}s")
    
    if yolo_result.signs:
        print("🚦 YOLO Detected signs:")
        for i, sign in enumerate(yolo_result.signs):
            print(f"   {i+1}. {sign.class_name} (confidence: {sign.confidence:.2f})")
            print(f"      Bbox: {sign.bbox}")
            print(f"      Center: {sign.center}")
    else:
        print("   No signs detected with YOLO")
    
    # Test with traditional CV
    print("\n🔍 Testing Traditional CV detection...")
    cv_result = detector.detect_signs(image, use_yolo=False)
    
    print(f"📊 Traditional CV Results:")
    print(f"   - Total signs detected: {cv_result.total_signs}")
    print(f"   - Method used: {cv_result.method_used}")
    print(f"   - Processing time: {cv_result.processing_time:.3f}s")
    
    if cv_result.signs:
        print("🚦 Traditional CV Detected signs:")
        for i, sign in enumerate(cv_result.signs):
            print(f"   {i+1}. {sign.class_name} (confidence: {sign.confidence:.2f})")
            print(f"      Bbox: {sign.bbox}")
            print(f"      Center: {sign.center}")
    else:
        print("   No signs detected with Traditional CV")
    
    # Draw detections and save results
    print("\n🎨 Drawing detections...")
    
    # YOLO result image
    if yolo_result.total_signs > 0:
        yolo_image = detector.draw_detections(image, yolo_result)
        yolo_output = "yolo_detection_result.jpg"
        cv2.imwrite(yolo_output, yolo_image)
        print(f"💾 YOLO result saved to: {yolo_output}")
    
    # Traditional CV result image
    if cv_result.total_signs > 0:
        cv_image = detector.draw_detections(image, cv_result)
        cv_output = "traditional_detection_result.jpg"
        cv2.imwrite(cv_output, cv_image)
        print(f"💾 Traditional CV result saved to: {cv_output}")
    
    # Combined result (use the method that detected more signs)
    best_result = yolo_result if yolo_result.total_signs >= cv_result.total_signs else cv_result
    combined_image = detector.draw_detections(image, best_result)
    combined_output = "best_detection_result.jpg"
    cv2.imwrite(combined_output, combined_image)
    print(f"💾 Best result saved to: {combined_output}")
    
    return True

def main():
    """Main test function."""
    print("=" * 70)
    print("🚦 TRAFFIC SIGN DETECTION - REAL IMAGE TEST")
    print("=" * 70)
    
    try:
        test_real_traffic_signs()
        
        print("\n" + "=" * 70)
        print("🎉 TEST COMPLETED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
