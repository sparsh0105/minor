# Intelligent Traffic System API

FastAPI-based REST API for the Intelligent Traffic System.

## Installation

```bash
pip install -r requirements_api.txt
```

## Running the API

```bash
# Development server
python -m src.api.main

# Or with uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Health & Info
- `GET /` - API information
- `GET /health` - Health check
- `GET /api/v1/system/stats` - System statistics

### Road Detection
- `POST /api/v1/road/detect` - Detect lanes in image

### Traffic Sign Detection
- `POST /api/v1/traffic-sign/detect` - Detect traffic signs

### Traffic Flow Prediction
- `POST /api/v1/traffic-flow/predict` - Predict traffic flow

### Road Condition Detection
- `POST /api/v1/road-condition/detect` - Detect potholes and cracks

### Complete Processing
- `POST /api/v1/process/image` - Process image with all modules

## Example Usage

### Using curl

```bash
# Road detection
curl -X POST "http://localhost:8000/api/v1/road/detect" \
  -F "file=@lane.jpeg" \
  -F "return_image=false"

# Traffic sign detection
curl -X POST "http://localhost:8000/api/v1/traffic-sign/detect" \
  -F "file=@traffic_sign.jpg" \
  -F "min_confidence=0.5"

# Traffic flow prediction
curl -X POST "http://localhost:8000/api/v1/traffic-flow/predict" \
  -F "car_count=25" \
  -F "bike_count=5" \
  -F "bus_count=3" \
  -F "truck_count=8"
```

### Using Python requests

```python
import requests

# Road detection
with open("lane.jpeg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/road/detect",
        files={"file": f},
        data={"return_image": False}
    )
    print(response.json())
```

## Testing with Postman

1. Import the API collection (create from `/docs` endpoint)
2. Test endpoints with sample images
3. Check responses and error handling

## Next Steps

1. Add MongoDB integration for storing results
2. Add JWT authentication
3. Add request rate limiting
4. Add async processing for videos
5. Add WebSocket support for real-time streams

