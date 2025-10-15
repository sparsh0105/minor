#!/usr/bin/env python3
"""
Test YOLO traffic sign detection using the detect.py functionality.
Run with: python test_yolo_detection.py
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import time
from typing import List, Tuple, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import YOLO detection utilities
sys.path.append('Real-Time-Traffic-Sign-Detection/Codes')

# Fix PyTorch loading issue
import torch
torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct'])

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import plot_one_box

class YOLOTrafficSignDetector:
    """YOLO-based traffic sign detector using the original detect.py functionality."""
    
    def __init__(self, weights_path: str = "data/models/Model/weights/best.pt", 
                 img_size: int = 640, conf_thres: float = 0.5, iou_thres: float = 0.45):
        """
        Initialize YOLO traffic sign detector.
        
        Args:
            weights_path: Path to YOLO model weights
            img_size: Input image size
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
        """
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize device
        self.device = select_device('')
        self.half = self.device.type != 'cpu'
        
        # Load model
        self.model = None
        self.names = []
        self.colors = []
        
        self._load_model()
    
    def _load_model_safely(self):
        """Load model with weights_only=False to handle older model formats."""
        try:
            # Try loading with weights_only=False
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            model = checkpoint['model'].float().fuse().eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _load_model(self) -> None:
        """Load YOLO model and get class names."""
        try:
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
            
            print(f"Loading YOLO model from: {self.weights_path}")
            
            # Load model with custom loading to handle PyTorch security restrictions
            self.model = self._load_model_safely()
            self.img_size = check_img_size(self.img_size, s=self.model.stride.max())
            
            if self.half:
                self.model.half()
            
            # Get class names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
            
            print(f"✅ Model loaded successfully!")
            print(f"   - Classes: {len(self.names)}")
            print(f"   - Class names: {list(self.names)}")
            print(f"   - Device: {self.device}")
            print(f"   - Half precision: {self.half}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image maintaining aspect ratio
        h, w = image.shape[:2]
        r = self.img_size / max(h, w)  # ratio
        new_unpad = int(round(min(h, w) * r))
        
        # Resize with padding
        if h > w:
            new_h, new_w = self.img_size, new_unpad
        else:
            new_h, new_w = new_unpad, self.img_size
        
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Add padding to make it square
        top, bottom = (self.img_size - new_h) // 2, (self.img_size - new_h + 1) // 2
        left, right = (self.img_size - new_w) // 2, (self.img_size - new_w + 1) // 2
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert BGR to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
        if self.half:
            img = img.half()
        else:
            img = img.float()
        
        img = img.unsqueeze(0)  # Add batch dimension
        
        return img
    
    def detect_signs(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect traffic signs in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            print("❌ Model not loaded!")
            return []
        
        try:
            # Preprocess image
            img = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                pred = self.model(img, augment=False)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                     classes=None, agnostic=False)
            
            detections = []
            original_shape = image.shape
            
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to original image size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_shape).round()
                    
                    # Process each detection
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        class_id = int(cls)
                        confidence = float(conf)
                        class_name = self.names[class_id] if class_id < len(self.names) else "unknown"
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detections on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Choose color
            color = self.colors[class_id] if class_id < len(self.colors) else [0, 255, 0]
            
            # Draw bounding box
            plot_one_box((x1, y1, x2, y2), result_image, 
                        label=f'{class_name} {conf:.2f}', 
                        color=color, line_thickness=2)
        
        # Add summary info
        info_text = f"Signs: {len(detections)} | Model: YOLO | Time: {time.time():.3f}s"
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image

def test_yolo_detection_on_real_images():
    """Test YOLO detection on real traffic sign images."""
    print("🚦 Testing YOLO Traffic Sign Detection on Real Images...")
    
    # Initialize detector
    detector = YOLOTrafficSignDetector()
    
    if detector.model is None:
        print("❌ Cannot proceed without model!")
        return False
    
    # Test images
    test_images = [
        "data/test_images/all-signs.png",
        "data/test_images/0.png",
        "data/test_images/Sample_2.png",
        "data/test_images/Sample_3.png"
    ]
    
    results = []
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        print(f"\n🔍 Testing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            continue
        
        print(f"   - Image shape: {image.shape}")
        
        # Detect signs
        start_time = time.time()
        detections = detector.detect_signs(image)
        processing_time = time.time() - start_time
        
        print(f"📊 Results:")
        print(f"   - Signs detected: {len(detections)}")
        print(f"   - Processing time: {processing_time:.3f}s")
        
        if detections:
            print("🚦 Detected signs:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
                print(f"      Bbox: {det['bbox']}")
                print(f"      Center: {det['center']}")
        
        # Draw and save results
        if detections:
            result_image = detector.draw_detections(image, detections)
            
            # Save result
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"test_results/yolo_{base_name}_detection.jpg"
            os.makedirs("test_results", exist_ok=True)
            cv2.imwrite(output_path, result_image)
            print(f"💾 Result saved to: {output_path}")
        
        results.append({
            'image_path': image_path,
            'detections': len(detections),
            'processing_time': processing_time,
            'signs': detections
        })
    
    return results

def test_yolo_detection_on_single_image(image_path: str) -> Dict[str, Any]:
    """
    Test YOLO detection on a single image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Detection results dictionary
    """
    print(f"🚦 Testing YOLO Detection on: {image_path}")
    
    # Initialize detector
    detector = YOLOTrafficSignDetector()
    
    if detector.model is None:
        return {'error': 'Model not loaded'}
    
    # Load image
    if not os.path.exists(image_path):
        return {'error': f'Image not found: {image_path}'}
    
    image = cv2.imread(image_path)
    if image is None:
        return {'error': f'Failed to load image: {image_path}'}
    
    # Detect signs
    start_time = time.time()
    detections = detector.detect_signs(image)
    processing_time = time.time() - start_time
    
    # Draw results
    result_image = detector.draw_detections(image, detections)
    
    # Save result
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"test_results/yolo_{base_name}_detection.jpg"
    os.makedirs("test_results", exist_ok=True)
    cv2.imwrite(output_path, result_image)
    
    return {
        'image_path': image_path,
        'detections': detections,
        'total_signs': len(detections),
        'processing_time': processing_time,
        'output_path': output_path,
        'model_info': {
            'weights_path': detector.weights_path,
            'img_size': detector.img_size,
            'conf_thres': detector.conf_thres,
            'iou_thres': detector.iou_thres,
            'device': str(detector.device),
            'class_names': detector.names
        }
    }

def main():
    """Main function."""
    print("=" * 70)
    print("🚦 YOLO TRAFFIC SIGN DETECTION TEST")
    print("=" * 70)
    
    try:
        # Test on multiple images
        results = test_yolo_detection_on_real_images()
        
        # Test on specific image if provided
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            single_result = test_yolo_detection_on_single_image(image_path)
            print(f"\n📊 Single Image Test Result:")
            print(f"   - Total signs: {single_result.get('total_signs', 0)}")
            print(f"   - Processing time: {single_result.get('processing_time', 0):.3f}s")
            print(f"   - Output saved to: {single_result.get('output_path', 'N/A')}")
        
        print("\n" + "=" * 70)
        print("🎉 YOLO DETECTION TEST COMPLETED!")
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
