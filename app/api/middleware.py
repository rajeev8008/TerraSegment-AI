"""API middleware - CORS, error handling, logging"""
from flask import jsonify, request
from functools import wraps
import time

from app.core.logger import get_logger
from app.core.exceptions import TerraSegmentError

logger = get_logger(__name__)


def log_request(f):
    """Decorator to log API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        logger.info(
            f"Request started: {request.method} {request.path}",
        )
        
        try:
            response = f(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)
            
            status_code = response[1] if isinstance(response, tuple) else 200
            logger.info(
                f"Request completed: {request.method} {request.path} "
                f"[{status_code}] {duration_ms}ms"
            )
            
            return response
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Request failed: {request.method} {request.path} "
                f"{duration_ms}ms - {str(e)}"
            )
            raise
    
    return decorated_function


def register_error_handlers(app):
    """Register global error handlers"""
    
    @app.errorhandler(TerraSegmentError)
    def handle_terrasegment_error(e):
        """Handle custom application errors"""
        logger.warning(f"Application error: {e.message}")
        return jsonify({'error': e.message}), e.status_code
    
    @app.errorhandler(404)
    def handle_not_found(e):
        """Handle 404 errors"""
        logger.warning(f"Not found: {request.path}")
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        """Handle 405 errors"""
        logger.warning(f"Method not allowed: {request.method} {request.path}")
        return jsonify({'error': 'Method not allowed'}), 405
    
    @app.errorhandler(500)
    def handle_internal_error(e):
        """Handle 500 errors"""
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


def setup_cors(app):
    """Setup CORS headers"""
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
