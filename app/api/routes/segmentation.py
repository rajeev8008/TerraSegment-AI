"""
Segmentation endpoint
"""
from flask import Blueprint, request, jsonify

from app.core.logger import get_logger
from app.core.exceptions import TerraSegmentError
from app.services import ImageService, SegmentationService
from models.model_loader import get_model_loader

logger = get_logger(__name__)
segmentation_bp = Blueprint('segmentation', __name__)

# Initialize services (will be done in app factory pattern later)
image_service = ImageService()
model_loader = get_model_loader()
segmentation_service = SegmentationService(model_loader, image_service)


@segmentation_bp.route('/api/v1/segment', methods=['POST'])
def segment_image():
    """
    Segment uploaded aerial image
    
    Request:
        - file: Multipart file upload (JPG or PNG)
    
    Response:
        - original_image: Base64 encoded original
        - segmentation_mask: Base64 encoded colored mask
        - percentages: Terrain composition percentages
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Processing segmentation request for file: {file.filename}")
        
        # Validate file
        file_bytes = file.read()
        image_service.validate_image_file(file.filename, len(file_bytes))
        
        # Load image
        image_rgb = image_service.load_image_from_bytes(file_bytes)
        
        # Segment and analyze
        mask, colored_mask, percentages = segmentation_service.segment_and_analyze(image_rgb)
        
        # Convert to base64
        original_base64 = image_service.image_to_base64(image_rgb)
        mask_base64 = image_service.image_to_base64(colored_mask)
        
        response = {
            'original_image': original_base64,
            'segmentation_mask': mask_base64,
            'percentages': percentages,
            'building': percentages.get('building', 0),
            'land': percentages.get('land', 0),
            'road': percentages.get('road', 0),
            'vegetation': percentages.get('vegetation', 0),
            'water': percentages.get('water', 0),
            'unlabeled': percentages.get('unlabeled', 0)
        }
        
        logger.info(f"Segmentation completed successfully for {file.filename}")
        return jsonify(response), 200
        
    except TerraSegmentError as e:
        logger.warning(f"Segmentation error: {e.message}")
        return jsonify({'error': e.message}), e.status_code
    
    except Exception as e:
        logger.error(f"Unexpected error during segmentation: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
