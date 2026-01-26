"""
Search endpoint
"""
from flask import Blueprint, request, jsonify

from app.core.logger import get_logger
from app.core.exceptions import TerraSegmentError
from app.services import ImageService, SearchService

logger = get_logger(__name__)
search_bp = Blueprint('search', __name__)

# Initialize services
image_service = ImageService()
search_service = SearchService(image_service)


@search_bp.route('/api/v1/search', methods=['POST'])
def search_image():
    """
    Search dataset for image matching terrain composition
    
    Request (JSON):
        - water: float (0-100)
        - road: float (0-100)
        - vegetation: float (0-100)
        - buildings: float (0-100)
        - land: float (0-100)
    
    Response:
        - image: Base64 encoded matched image
        - Actual percentages of matched image
        - distance: How close the match is
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        # Extract percentages
        target_percentages = {
            'water': float(data.get('water', 0)),
            'road': float(data.get('road', 0)),
            'vegetation': float(data.get('vegetation', 0)),
            'buildings': float(data.get('buildings', 0)),
            'land': float(data.get('land', 0))
        }
        
        logger.info(f"Searching for image with: {target_percentages}")
        
        # Search and load image
        image_rgb, match_info = search_service.search_and_load_image(target_percentages)
        
        # Convert to base64
        image_base64 = image_service.image_to_base64(image_rgb)
        
        response = {
            'image': image_base64,
            'water': match_info['water'],
            'road': match_info['road'],
            'vegetation': match_info['vegetation'],
            'buildings': match_info['buildings'],
            'land': match_info['land'],
            'distance': match_info['distance'],
            'image_name': match_info['image_path'].split('/')[-1]
        }
        
        logger.info(f"Search completed: matched {match_info['image_path']}")
        return jsonify(response), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    except TerraSegmentError as e:
        logger.warning(f"Search error: {e.message}")
        return jsonify({'error': e.message}), e.status_code
    
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@search_bp.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        stats = search_service.get_dataset_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({'error': str(e)}), 500
