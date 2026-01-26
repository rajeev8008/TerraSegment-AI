"""
Health check endpoint
"""
from flask import Blueprint, jsonify
import os

from app.core.config import settings
from models.model_loader import get_model_loader
import app

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns service status and model availability
    """
    model_loader = get_model_loader()
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader.is_loaded(),
        'model_path': settings.model_path,
        'model_exists': os.path.exists(settings.model_path),
        'version': app.__version__,
        'environment': settings.environment
    }), 200


@health_bp.route('/', methods=['GET'])
def root():
    """Root endpoint - API info"""
    return jsonify({
        'name': 'TerraSegment AI API',
        'version': app.__version__,
        'endpoints': {
            'health': '/health',
            'segment': '/api/v1/segment (POST)',
            'search': '/api/v1/search (POST)',
        }
    }), 200
