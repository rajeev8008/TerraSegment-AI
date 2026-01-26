"""
TerraSegment AI - Main Flask Application
Production-ready version with modular architecture
"""
from flask import Flask, render_template
import os

from app.core.config import settings
from app.core.logger import get_logger, setup_logging
from app.api.middleware import register_error_handlers, setup_cors
from app.api.routes import health_bp, segmentation_bp, search_bp

# Setup logging
setup_logging()
logger = get_logger(__name__)


def create_app():
    """
    Application factory pattern
    Creates and configures the Flask application
    """
    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static'
    )
    
    # Load configuration
    app.config['MAX_CONTENT_LENGTH'] = settings.max_image_size_bytes
    app.config['JSON_SORT_KEYS'] = False
    
    # Setup middleware
    setup_cors(app)
    register_error_handlers(app)
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(segmentation_bp)
    app.register_blueprint(search_bp)
    
    # Web UI route
    @app.route('/ui')
    @app.route('/index.html')
    def index():
        """Serve the web UI"""
        return render_template('index.html')
    
    logger.info(f"TerraSegment AI initialized - Environment: {settings.environment}")
    logger.info(f"Model path: {settings.model_path}")
    logger.info(f"Debug mode: {settings.debug}")
    
    return app


def main():
    """Main entry point"""
    app = create_app()
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug
    )


if __name__ == '__main__':
    main()
