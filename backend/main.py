"""
FastAPI main application.
Entry point for the backend API.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time

from config.settings import settings
from config.database import check_db_connection, init_db
from backend.api.v1 import auth
from backend.utils.logger import get_logger
from backend.utils.exceptions import AnomalyDetectorException

logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready anomaly detection platform with time series analysis",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their processing time."""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    logger.info(
        f"Completed: {request.method} {request.url.path} "
        f"Status: {response.status_code} Time: {process_time:.3f}s"
    )

    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Exception handlers
@app.exception_handler(AnomalyDetectorException)
async def anomaly_detector_exception_handler(request: Request, exc: AnomalyDetectorException):
    """Handle custom application exceptions."""
    logger.error(f"Application error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info("=" * 60)

    # Check database connection
    logger.info("Checking database connection...")
    if check_db_connection():
        logger.info("✓ Database connection successful")

        # Initialize database tables if needed
        try:
            init_db()
            logger.info("✓ Database tables verified/created")
        except Exception as e:
            logger.error(f"✗ Database initialization failed: {e}")
    else:
        logger.error("✗ Database connection failed")
        logger.warning("Application started but database is not available")

    logger.info("=" * 60)
    logger.info(f"API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info(f"ReDoc Documentation: http://{settings.API_HOST}:{settings.API_PORT}/redoc")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown."""
    logger.info("Shutting down application...")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running.

    Returns service status and database connection status.
    """
    db_status = check_db_connection()

    return {
        "status": "healthy" if db_status else "degraded",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "connected" if db_status else "disconnected"
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Anomaly Detection Platform API",
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(auth.router, prefix="/api/v1")


# Run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
