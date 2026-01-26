"""
Configuration management using pydantic-settings
Loads configuration from environment variables and .env file
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings"""
    
    # Paths
    model_path: str = Field(default="semantic_segmentation_model.h5", description="Path to trained model")
    dataset_root: str = Field(default="Dataset-Segmentation", description="Root directory for dataset")
    csv_path: str = Field(default="New_dataset.csv", description="Path to dataset CSV")
    
    # Image processing
    max_image_size_mb: int = Field(default=10, description="Maximum image size in MB")
    patch_size: int = Field(default=256, description="Image patch size for model")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    
    # Computed properties
    @property
    def max_image_size_bytes(self) -> int:
        """Convert max image size from MB to bytes"""
        return self.max_image_size_mb * 1024 * 1024
    
    @property
    def image_dir(self) -> Path:
        """Path to images directory"""
        return Path(self.dataset_root) / "images"
    
    @property
    def mask_dir(self) -> Path:
        """Path to masks directory"""
        return Path(self.dataset_root) / "masks"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v
    
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from env
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (for dependency injection)"""
    return settings
