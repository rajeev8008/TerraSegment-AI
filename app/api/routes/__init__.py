"""Routes package init"""
from app.api.routes.health import health_bp
from app.api.routes.segmentation import segmentation_bp
from app.api.routes.search import search_bp

__all__ = ["health_bp", "segmentation_bp", "search_bp"]
