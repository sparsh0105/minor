"""
FastAPI application for Intelligent Traffic System.

This module provides REST API endpoints for:
- Road detection
- Traffic sign detection
- Traffic flow prediction
- Road condition detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import io
import json
from pathlib import Path
from datetime import datetime
import logging

import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent.parent.parent))

from src.main import IntelligentTrafficSystem
from src.core.traffic_flow_prediction import VehicleCounts

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Traffic System API",
    description="REST API for road detection, traffic sign detection, traffic flow prediction, and road condition detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the traffic system
traffic_system = IntelligentTrafficSystem()

# Logger
logger = logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types.
        
    Returns:
        Object with numpy types converted to Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """
    Load image from uploaded file.
    
    Args:
        file: Uploaded file object.
        
    Returns:
        Image as numpy array.
        
    Raises:
        HTTPException: If image cannot be loaded.
    """
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint - API information.
    
    Returns:
        API information and status.
    """
    return {
        "name": "Intelligent Traffic System API",
        "version": "1.0.0",
        "status": "running",
        "modules": [
            "road_detection",
            "traffic_sign_detection",
            "traffic_flow_prediction",
            "road_condition_detection"
        ],
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        System health status.
    """
    return {
        "status": "healthy",
        "system_initialized": traffic_system.is_initialized,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/road/detect")
async def detect_roads(
    file: UploadFile = File(...),
    return_image: bool = Form(False)
) -> Dict[str, Any]:
    """
    Detect roads and lanes in an image.
    
    Args:
        file: Image file to process.
        return_image: Whether to return processed image.
        
    Returns:
        Road detection results.
    """
    try:
        image = load_image_from_upload(file)
        
        # Detect lanes
        lane_info = traffic_system.road_detector.detect_lanes(image)
        
        result = {
            "success": True,
            "lane_detected": lane_info.left_lane is not None or lane_info.right_lane is not None,
            "lane_confidence": float(lane_info.lane_confidence),
            "lane_angle": float(lane_info.lane_angle),
            "departure_warning": lane_info.lane_departure_warning,
            "timestamp": datetime.now().isoformat()
        }
        
        if return_image:
            result_image = traffic_system.road_detector.draw_lanes(image, lane_info)
            # Save temporary image
            output_path = Path("temp_road_detection.jpg")
            cv2.imwrite(str(output_path), result_image)
            return FileResponse(
                str(output_path),
                media_type="image/jpeg",
                headers={"X-Result-Data": json.dumps(result)}
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in road detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/traffic-sign/detect")
async def detect_traffic_signs(
    file: UploadFile = File(...),
    return_image: bool = Form(False),
    min_confidence: float = Form(0.5)
) -> Dict[str, Any]:
    """
    Detect traffic signs in an image.
    
    Args:
        file: Image file to process.
        return_image: Whether to return processed image.
        min_confidence: Minimum confidence threshold (filters results after detection).
        
    Returns:
        Traffic sign detection results.
    """
    try:
        image = load_image_from_upload(file)
        
        # Detect traffic signs (method doesn't accept min_confidence, we filter after)
        detection_result = traffic_system.traffic_sign_detector.detect_signs(image)
        
        # Filter signs by confidence threshold
        filtered_signs = [
            sign for sign in detection_result.signs 
            if sign.confidence >= min_confidence
        ]
        
        signs_data = []
        for sign in filtered_signs:
            signs_data.append({
                "class_name": sign.class_name,
                "confidence": float(sign.confidence),
                "bbox": convert_numpy_types(sign.bbox),
                "center": convert_numpy_types(sign.center),
                "method": sign.method
            })
        
        result = {
            "success": True,
            "signs_detected": len(filtered_signs),
            "total_signs_found": len(detection_result.signs),
            "signs": signs_data,
            "min_confidence_used": min_confidence,
            "processing_time": detection_result.processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if return_image:
            # Create a filtered detection result for drawing
            from src.core.traffic_sign_detection import TrafficSignDetectionResult
            filtered_result = TrafficSignDetectionResult(
                signs=filtered_signs,
                total_signs=len(filtered_signs),
                processing_time=detection_result.processing_time,
                method_used=detection_result.method_used
            )
            result_image = traffic_system.traffic_sign_detector.draw_detections(
                image,
                filtered_result
            )
            output_path = Path("temp_traffic_sign.jpg")
            cv2.imwrite(str(output_path), result_image)
            return FileResponse(
                str(output_path),
                media_type="image/jpeg",
                headers={"X-Result-Data": json.dumps(result)}
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in traffic sign detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/traffic-flow/predict")
async def predict_traffic_flow(
    car_count: int = Form(...),
    bike_count: int = Form(0),
    bus_count: int = Form(0),
    truck_count: int = Form(0)
) -> Dict[str, Any]:
    """
    Predict traffic flow based on vehicle counts.
    
    Args:
        car_count: Number of cars.
        bike_count: Number of bikes.
        bus_count: Number of buses.
        truck_count: Number of trucks.
        
    Returns:
        Traffic flow prediction results.
    """
    try:
        # Create vehicle counts object
        vehicle_counts = VehicleCounts(
            car_count=car_count,
            bike_count=bike_count,
            bus_count=bus_count,
            truck_count=truck_count,
            total_count=car_count + bike_count + bus_count + truck_count,
            timestamp=datetime.now()
        )
        
        # Predict traffic situation
        prediction = traffic_system.traffic_flow_predictor.predict_traffic_situation(
            vehicle_counts
        )
        
        result = {
            "success": True,
            "predicted_situation": prediction.predicted_situation,
            "confidence": float(prediction.confidence),
            "congestion_level": prediction.congestion_level,
            "predicted_vehicle_counts": {
                "car_count": prediction.predicted_vehicle_counts.car_count,
                "bike_count": prediction.predicted_vehicle_counts.bike_count,
                "bus_count": prediction.predicted_vehicle_counts.bus_count,
                "truck_count": prediction.predicted_vehicle_counts.truck_count,
                "total_count": prediction.predicted_vehicle_counts.total_count
            },
            "features_used": prediction.features_used,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in traffic flow prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/road-condition/detect")
async def detect_road_conditions(
    file: UploadFile = File(...),
    return_image: bool = Form(False)
) -> Dict[str, Any]:
    """
    Detect road conditions (potholes, cracks) in an image.
    
    Args:
        file: Image file to process.
        return_image: Whether to return processed image.
        
    Returns:
        Road condition detection results.
    """
    try:
        image = load_image_from_upload(file)
        
        # Detect road conditions
        condition_result = traffic_system.road_condition_detector.detect_road_conditions(
            image
        )
        
        potholes_data = []
        for pothole in condition_result.potholes:
            potholes_data.append({
                "center": convert_numpy_types(pothole.center),
                "area": convert_numpy_types(pothole.area),
                "severity": pothole.severity,
                "bbox": convert_numpy_types(pothole.bbox)
            })
        
        cracks_data = []
        for crack in condition_result.cracks:
            cracks_data.append({
                "start_point": convert_numpy_types(crack.start_point),
                "end_point": convert_numpy_types(crack.end_point),
                "length": float(crack.length),
                "width": float(crack.width),
                "severity": crack.severity,
                "type": crack.type
            })
        
        surface_defects_data = []
        for defect in condition_result.surface_defects:
            surface_defects_data.append({
                "defect_type": defect.defect_type,
                "center": convert_numpy_types(defect.center),
                "area": convert_numpy_types(defect.area),
                "severity": defect.severity
            })
        
        result = {
            "success": True,
            "potholes_detected": len(condition_result.potholes),
            "cracks_detected": len(condition_result.cracks),
            "surface_defects_detected": len(condition_result.surface_defects),
            "potholes": potholes_data,
            "cracks": cracks_data,
            "surface_defects": surface_defects_data,
            "overall_quality": condition_result.overall_quality,
            "quality_score": float(condition_result.quality_score),
            "maintenance_priority": condition_result.maintenance_priority,
            "processing_time": condition_result.processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if return_image:
            result_image = traffic_system.road_condition_detector.draw_detections(
                image,
                condition_result
            )
            output_path = Path("temp_road_condition.jpg")
            cv2.imwrite(str(output_path), result_image)
            return FileResponse(
                str(output_path),
                media_type="image/jpeg",
                headers={"X-Result-Data": json.dumps(result)}
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in road condition detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/process/image")
async def process_image_complete(
    file: UploadFile = File(...),
    modules: Optional[str] = Form("all")
) -> Dict[str, Any]:
    """
    Process image with all or selected modules.
    
    Args:
        file: Image file to process.
        modules: Comma-separated list of modules to use (road,traffic_sign,road_condition,all).
        
    Returns:
        Complete processing results.
    """
    try:
        image = load_image_from_upload(file)
        
        # Process with main system
        results = traffic_system.process_image(image)
        
        # Convert numpy types in results for JSON serialization
        results = convert_numpy_types(results)
        
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/system/stats")
async def get_system_stats() -> Dict[str, Any]:
    """
    Get system processing statistics.
    
    Returns:
        System statistics.
    """
    return {
        "success": True,
        "statistics": traffic_system.processing_stats,
        "timestamp": datetime.now().isoformat()
    }


# Router for API versioning
router = app


def main() -> None:
    """Main entry point for running the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

