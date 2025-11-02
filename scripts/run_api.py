"""
Script to run the FastAPI backend server.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    print("=" * 60)
    print(f"Starting {settings.APP_NAME} API Server")
    print("=" * 60)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Host: {settings.API_HOST}")
    print(f"Port: {settings.API_PORT}")
    print(f"Reload: {settings.DEBUG}")
    print("=" * 60)
    print(f"\nAPI Documentation: http://localhost:{settings.API_PORT}/docs")
    print(f"ReDoc Documentation: http://localhost:{settings.API_PORT}/redoc")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
