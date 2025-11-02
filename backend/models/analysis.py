"""
Analysis result model for storing anomaly detection results.
"""
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from config.database import Base


class AnalysisResult(Base):
    """Analysis result model for storing anomaly detection outcomes."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    algorithm = Column(String(50), nullable=False)  # 'zscore', 'iqr', 'moving_avg'
    parameters = Column(JSONB, nullable=False)  # Algorithm parameters used
    anomalies_json = Column(JSONB, nullable=False)  # Detected anomalies with indices and values
    anomaly_count = Column(Integer, nullable=False)
    execution_time_ms = Column(Integer, nullable=True)  # Time taken for analysis
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="analysis_results")
    dataset = relationship("Dataset", back_populates="analysis_results")

    def __repr__(self):
        return (
            f"<AnalysisResult(id={self.id}, algorithm='{self.algorithm}', "
            f"anomalies={self.anomaly_count}, dataset_id={self.dataset_id})>"
        )
