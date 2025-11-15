"""
Simple script to run the FastAPI server.

Usage:
    python run_api.py
    python run_api.py --port 8000
    python run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Intelligent Traffic System API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Intelligent Traffic System API...")
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")
    print(f"🔍 ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"💚 Health: http://{args.host}:{args.port}/health")
    print()
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )

