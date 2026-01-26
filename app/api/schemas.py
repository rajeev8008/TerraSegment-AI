"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional


class SegmentResponse(BaseModel):
    """Response schema for segmentation endpoint"""
    original_image: str = Field(..., description="Base64 encoded original image")
    segmentation_mask: str = Field(..., description="Base64 encoded segmentation mask")
    percentages: Dict[str, float] = Field(..., description="Terrain class percentages")
    building: float
    land: float
    road: float
    vegetation: float
    water: float
    unlabeled: float


class SearchRequest(BaseModel):
    """Request schema for search endpoint"""
    water: float = Field(default=0, ge=0, le=100, description="Water percentage")
    road: float = Field(default=0, ge=0, le=100, description="Road percentage")
    vegetation: float = Field(default=0, ge=0, le=100, description="Vegetation percentage")
    buildings: float = Field(default=0, ge=0, le=100, description="Building percentage")
    land: float = Field(default=0, ge=0, le=100, description="Land percentage")
    
    @field_validator('water', 'road', 'vegetation', 'buildings', 'land')
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        """Validate percentage is within valid range"""
        if v < 0 or v > 100:
            raise ValueError("Percentage must be between 0 and 100")
        return round(v, 2)


class SearchResponse(BaseModel):
    """Response schema for search endpoint"""
    image: str = Field(..., description="Base64 encoded image")
    water: float
    road: float
    vegetation: float
    buildings: float
    land: float
    distance: float = Field(..., description="Distance from search criteria")
    image_name: str = Field(..., description="Name of the matched image")


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    version: str
    
class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
