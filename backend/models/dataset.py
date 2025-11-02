"""
Dataset model for storing uploaded and fetched time series data.
"""
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from config.database import Base


class Dataset(Base):
    """Dataset model for storing time series data from various sources."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)  # 'upload', 'stock', 'crypto', 'weather', 'synthetic'
    file_path = Column(String(500), nullable=True)  # For uploaded files
    data_json = Column(JSONB, nullable=True)  # Stored time series data as JSON
    dataset_metadata = Column(JSONB, nullable=True)  # Additional metadata (column names, data types, etc.)
    row_count = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="datasets")
    analysis_results = relationship("AnalysisResult", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', source='{self.source_type}', rows={self.row_count})>"
