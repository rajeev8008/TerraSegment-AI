"""
Segmentation service
Handles semantic segmentation logic
"""
import numpy as np
from typing import Dict, Tuple

from app.core.logger import get_logger
from app.core.exceptions import InferenceError
from app.services.image_service import ImageService
from models.model_loader import ModelLoader

logger = get_logger(__name__)


class SegmentationService:
    """Service for semantic segmentation operations"""
    
    # Terrain class mapping (consistent with training)
    CLASS_LABELS = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled']
    
    # Color mapping for visualization (RGB format)
    CLASS_COLORS = {
        0: (60, 16, 152),        # Building - Purple
        1: (132, 41, 246),       # Land - Violet
        2: (110, 193, 228),      # Road - Cyan
        3: (254, 221, 58),       # Vegetation - Yellow
        4: (226, 169, 41),       # Water - Orange
        5: (155, 155, 155)       # Unlabeled - Gray
    }
    
    def __init__(
        self, 
        model_loader: ModelLoader,
        image_service: ImageService
    ):
        """
        Initialize segmentation service
        
        Args:
            model_loader: Model loader instance
            image_service: Image service instance
        """
        self.model_loader = model_loader
        self.image_service = image_service
    
    def segment_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Run semantic segmentation on image
        
        Args:
            image_rgb: Input image as RGB numpy array
            
        Returns:
            Segmentation mask (HxW) with class IDs
            
        Raises:
            InferenceError: If segmentation fails
        """
        try:
            # Load model (uses cache if already loaded)
            model = self.model_loader.load_model()
            
            # Preprocess image
            image_input = self.image_service.preprocess_for_model(image_rgb)
            
            # Run inference
            logger.debug("Running model inference...")
            prediction = model.predict(image_input, verbose=0)
            
            # Convert probabilities to class IDs
            segmentation_mask = np.argmax(prediction[0], axis=-1)
            
            logger.info(f"Segmentation completed: shape={segmentation_mask.shape}")
            return segmentation_mask
            
        except Exception as e:
            logger.error(f"Segmentation failed: {str(e)}")
            raise InferenceError(f"Segmentation failed: {str(e)}")
    
    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert numeric segmentation mask to RGB visualization
        
        Args:
            mask: Segmentation mask with class IDs (HxW)
            
        Returns:
            Colored mask (HxWx3)
        """
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_id, color in self.CLASS_COLORS.items():
            colored_mask[mask == class_id] = color
        
        logger.debug("Mask colorized")
        return colored_mask
    
    def calculate_percentages(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate terrain class percentages from segmentation mask
        
        Args:
            mask: Segmentation mask with class IDs
            
        Returns:
            Dictionary mapping class names to percentages
        """
        total_pixels = mask.size
        percentages = {}
        
        for class_id, label in enumerate(self.CLASS_LABELS):
            count = np.sum(mask == class_id)
            percentage = (count / total_pixels) * 100
            percentages[label.lower()] = round(percentage, 2)
        
        logger.debug(f"Calculated percentages: {percentages}")
        return percentages
    
    def segment_and_analyze(
        self, 
        image_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Complete segmentation pipeline: segment, colorize, and analyze
        
        Args:
            image_rgb: Input image as RGB numpy array
            
        Returns:
            Tuple of (segmentation_mask, colored_mask, percentages)
        """
        # Segment image
        mask = self.segment_image(image_rgb)
        
        # Colorize for visualization
        colored_mask = self.colorize_mask(mask)
        
        # Calculate percentages
        percentages = self.calculate_percentages(mask)
        
        logger.info("Complete segmentation pipeline finished")
        return mask, colored_mask, percentages
