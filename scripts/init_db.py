"""
Database initialization script.
Creates all database tables and optionally seeds with sample data.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.database import init_db, check_db_connection, engine
from config.settings import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Initialize the database."""
    logger.info("=" * 60)
    logger.info("Database Initialization Script")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Database URL: {settings.get_database_url()}")
    logger.info("=" * 60)

    # Check database connection
    logger.info("Checking database connection...")
    if not check_db_connection():
        logger.error("Failed to connect to database. Please check your configuration.")
        logger.error("Make sure PostgreSQL is running and credentials are correct.")
        sys.exit(1)

    # Initialize database
    logger.info("Creating database tables...")
    try:
        # Import all models to ensure they're registered
        from backend.models import User, Dataset, AnalysisResult, UserSession

        # Create all tables
        init_db()

        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        logger.info(f"Successfully created {len(tables)} tables:")
        for table in tables:
            logger.info(f"  ✓ {table}")

        logger.info("=" * 60)
        logger.info("Database initialization completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
