#!/usr/bin/env python3
"""
Quick test script for the Intelligent Traffic System.
This script performs basic functionality tests on all modules.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports() -> bool:
    """Test if all modules can be imported successfully."""
    print("🔍 Testing imports...")
    try:
        from src.core.road_detection import RoadDetector
        from src.core.traffic_sign_detection import TrafficSignDetector
        from src.core.traffic_flow_prediction import TrafficFlowPredictor
        from src.core.road_condition_detection import RoadConditionDetector
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_files() -> bool:
    """Test if all required data files exist."""
    print("📁 Checking data files...")
    
    required_files = [
        "data/test_images/lane.jpeg",
        "data/test_images/0.png",
        "data/datasets/TrafficDataset.csv",
        "data/test_videos/MVI_1049.avi",
        "data/models/data_svm.dat"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required data files found")
        return True

def test_road_detection() -> bool:
    """Test road detection module."""
    print("🛣️ Testing road detection...")
    try:
        from src.core.road_detection import RoadDetector
        detector = RoadDetector()
        result = detector.detect_lanes("data/test_images/lane.jpeg")
        if result and result.get("success"):
            print("✅ Road detection working")
            return True
        else:
            print("❌ Road detection failed")
            return False
    except Exception as e:
        print(f"❌ Road detection error: {e}")
        return False

def test_traffic_sign_detection() -> bool:
    """Test traffic sign detection module."""
    print("🚦 Testing traffic sign detection...")
    try:
        from src.core.traffic_sign_detection import TrafficSignDetector
        detector = TrafficSignDetector()
        result = detector.detect_signs("data/test_images/0.png")
        if result and result.get("success"):
            print("✅ Traffic sign detection working")
            return True
        else:
            print("❌ Traffic sign detection failed")
            return False
    except Exception as e:
        print(f"❌ Traffic sign detection error: {e}")
        return False

def test_traffic_flow_prediction() -> bool:
    """Test traffic flow prediction module."""
    print("📊 Testing traffic flow prediction...")
    try:
        from src.core.traffic_flow_prediction import TrafficFlowPredictor
        predictor = TrafficFlowPredictor()
        result = predictor.predict_flow("data/datasets/TrafficDataset.csv")
        if result and result.get("success"):
            print("✅ Traffic flow prediction working")
            return True
        else:
            print("❌ Traffic flow prediction failed")
            return False
    except Exception as e:
        print(f"❌ Traffic flow prediction error: {e}")
        return False

def test_road_condition_detection() -> bool:
    """Test road condition detection module."""
    print("🕳️ Testing road condition detection...")
    try:
        from src.core.road_condition_detection import RoadConditionDetector
        detector = RoadConditionDetector()
        result = detector.detect_conditions("data/test_images/lane.jpeg")
        if result and result.get("success"):
            print("✅ Road condition detection working")
            return True
        else:
            print("❌ Road condition detection failed")
            return False
    except Exception as e:
        print(f"❌ Road condition detection error: {e}")
        return False

def main() -> None:
    """Run all quick tests."""
    print("🚦 Intelligent Traffic System - Quick Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Files Test", test_data_files),
        ("Road Detection Test", test_road_detection),
        ("Traffic Sign Detection Test", test_traffic_sign_detection),
        ("Traffic Flow Prediction Test", test_traffic_flow_prediction),
        ("Road Condition Detection Test", test_road_condition_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to use.")
        print("\n📖 Next steps:")
        print("  1. Run: python test_with_real_data.py")
        print("  2. Run: python -m pytest tests/ -v")
        print("  3. Check TESTING_GUIDE.md for detailed instructions")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("📖 See TESTING_GUIDE.md for troubleshooting help.")

if __name__ == "__main__":
    main()
