# Intelligent Traffic System - API Guide

## Features

The Intelligent Traffic System provides 4 core modules for road and traffic analysis:

1. **Road Detection** - Detects and tracks road lanes in real-time, identifies road boundaries, and provides lane departure warnings
2. **Traffic Sign Detection** - Recognizes and classifies traffic signs (stop signs, speed limits, yield signs, etc.) using YOLOv5 deep learning model
3. **Traffic Flow Prediction** - Analyzes vehicle counts and predicts traffic congestion levels (light/moderate/heavy) using machine learning
4. **Road Condition Detection** - Identifies potholes, cracks, and surface defects to assess road quality and maintenance needs

---

## API Endpoints

### Base URL
```
http://localhost:8000
```

### 1. Health Check

**Endpoint:** `GET /health`

**Body:** None

**Response:**
```json
{
  "status": "healthy",
  "system_initialized": true,
  "timestamp": "2024-..."
}
```

---

### 2. Road Detection

**Endpoint:** `POST /api/v1/road/detect`

**Body (form-data):**
- `file` (File) - Road/lane image
- `return_image` (Text) - `true` or `false` - Whether to return processed image

**Test Data:**
- `data/test_images/lane.jpeg`
- `data/test_images/lane2.jpeg`
- `data/test_images/lane3.jpeg`
- `data/test_images/lane4.jpeg`

**Expected Response:**
```json
{
  "success": true,
  "lane_detected": true,
  "lane_confidence": 0.85,
  "lane_angle": 2.5,
  "departure_warning": false,
  "timestamp": "2024-..."
}
```

**What to Test:**
- Use lane images with clear lane markings
- Test with `return_image=true` to see visual results
- Expect `lane_detected: true` for clear road images
- `lane_confidence` ranges from 0.0 to 1.0

---

### 3. Traffic Sign Detection

**Endpoint:** `POST /api/v1/traffic-sign/detect`

**Body (form-data):**
- `file` (File) - Traffic sign image
- `min_confidence` (Text) - `0.5` (default) - Minimum confidence threshold (0.0-1.0)
- `return_image` (Text) - `true` or `false` - Whether to return processed image

**Test Data:**
- `data/test_images/0.png`
- `data/test_images/all-signs.png`
- `data/test_images/Sample_2.png`
- `data/test_images/Sample_3.png`
- `data/test_images/traffic_sign_dataset/(2).png`
- `data/test_images/traffic_sign_dataset/(28).png`

**Expected Response:**
```json
{
  "success": true,
  "signs_detected": 3,
  "total_signs_found": 5,
  "signs": [
    {
      "class_name": "stop sign",
      "confidence": 0.92,
      "bbox": [100, 150, 200, 250],
      "center": [150, 200],
      "method": "yolo"
    }
  ],
  "min_confidence_used": 0.5,
  "processing_time": 0.15,
  "timestamp": "2024-..."
}
```

**What to Test:**
- Use images with visible traffic signs
- Test different `min_confidence` values (0.3, 0.5, 0.7) to see filtering
- `all-signs.png` should detect multiple signs
- YOLO method provides higher accuracy than traditional CV

---

### 4. Traffic Flow Prediction

**Endpoint:** `POST /api/v1/traffic-flow/predict`

**Body (form-data):**
- `car_count` (Text) - Number of cars (required)
- `bike_count` (Text) - Number of bikes (default: 0)
- `bus_count` (Text) - Number of buses (default: 0)
- `truck_count` (Text) - Number of trucks (default: 0)

**Test Data:**
Use realistic vehicle counts:
- **Light Traffic:** `car_count=5, bike_count=2, bus_count=1, truck_count=1`
- **Moderate Traffic:** `car_count=15, bike_count=3, bus_count=2, truck_count=5`
- **Heavy Traffic:** `car_count=25, bike_count=5, bus_count=3, truck_count=8`

**Expected Response:**
```json
{
  "success": true,
  "predicted_situation": "moderate",
  "confidence": 0.87,
  "congestion_level": "normal",
  "predicted_vehicle_counts": {
    "car_count": 18,
    "bike_count": 4,
    "bus_count": 2,
    "truck_count": 6,
    "total_count": 30
  },
  "features_used": ["car_count", "total_count", "hour", "is_peak_hour"],
  "timestamp": "2024-..."
}
```

**What to Test:**
- Test with low counts (light traffic) - should predict "light"
- Test with high counts (heavy traffic) - should predict "heavy"
- `predicted_situation` can be: "light", "moderate", "heavy"
- `congestion_level` can be: "low", "normal", "high"

---

### 5. Road Condition Detection

**Endpoint:** `POST /api/v1/road-condition/detect`

**Body (form-data):**
- `file` (File) - Road surface image
- `return_image` (Text) - `true` or `false` - Whether to return processed image

**Test Data:**
- `data/test_images/lane.jpeg` (may have some road conditions)
- `data/test_images/lane2.jpeg`
- `data/test_images/lane3.jpeg`
- Any road surface images with visible potholes or cracks

**Expected Response:**
```json
{
  "success": true,
  "potholes_detected": 2,
  "cracks_detected": 5,
  "surface_defects_detected": 1,
  "potholes": [
    {
      "center": [320, 240],
      "area": 1500,
      "severity": "medium",
      "bbox": [300, 220, 340, 260]
    }
  ],
  "cracks": [
    {
      "start_point": [100, 200],
      "end_point": [300, 250],
      "length": 250.5,
      "width": 2.3,
      "severity": "low",
      "type": "linear"
    }
  ],
  "surface_defects": [...],
  "overall_quality": "fair",
  "quality_score": 0.65,
  "maintenance_priority": "medium",
  "processing_time": 0.12,
  "timestamp": "2024-..."
}
```

**What to Test:**
- Use road images with visible surface defects
- `overall_quality` can be: "excellent", "good", "fair", "poor", "critical"
- `maintenance_priority` can be: "low", "medium", "high", "urgent"
- `quality_score` ranges from 0.0 (worst) to 1.0 (best)
- Higher defect counts = lower quality score

---

### 6. Complete Image Processing

**Endpoint:** `POST /api/v1/process/image`

**Body (form-data):**
- `file` (File) - Image to process with all modules
- `modules` (Text) - `all` (default) or comma-separated: `road,traffic_sign,road_condition`

**Test Data:**
- Any road image that might have lanes, signs, and road conditions
- `data/test_images/lane4.jpeg`
- `data/test_images/test1.jpeg`

**Expected Response:**
```json
{
  "success": true,
  "results": {
    "road_detection": {...},
    "traffic_sign_detection": {...},
    "road_condition_detection": {...}
  },
  "timestamp": "2024-..."
}
```

**What to Test:**
- Processes image through all 4 modules at once
- Returns comprehensive analysis
- Use for complete road scene analysis

---

## Quick Start in Postman

1. **Start the API:**
   ```bash
   python run_api.py
   ```

2. **Test Health:**
   - Method: `GET`
   - URL: `http://localhost:8000/health`

3. **Test Road Detection:**
   - Method: `POST`
   - URL: `http://localhost:8000/api/v1/road/detect`
   - Body → form-data:
     - `file`: Select `data/test_images/lane.jpeg`
     - `return_image`: `true`

4. **View API Documentation:**
   - Visit: `http://localhost:8000/docs` (Swagger UI)
   - Visit: `http://localhost:8000/redoc` (ReDoc)

---

## Tips

- Always use `form-data` (not raw JSON) for file uploads
- Set `return_image=true` to get visual results with annotations
- Check response `success` field to verify operation
- Processing times are included in responses for performance monitoring
- For traffic flow, use realistic vehicle counts based on actual road scenarios

