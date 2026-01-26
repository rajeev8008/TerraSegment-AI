"""
Search service
Handles searching pre-computed dataset
"""
import os
import pandas as pd
from typing import Dict, Optional
import numpy as np

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import DatasetNotFoundError, InvalidImageError
from app.services.image_service import ImageService

logger = get_logger(__name__)


class SearchService:
    """Service for searching dataset by terrain composition"""
    
    def __init__(self, image_service: ImageService):
        """
        Initialize search service
        
        Args:
            image_service: Image service instance
        """
        self.image_service = image_service
        self._dataset: Optional[pd.DataFrame] = None
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the pre-computed dataset CSV"""
        csv_path = settings.csv_path
        
        if not os.path.exists(csv_path):
            logger.error(f"Dataset CSV not found: {csv_path}")
            raise DatasetNotFoundError(f"Dataset CSV not found at {csv_path}")
        
        try:
            self._dataset = pd.read_csv(csv_path)
            logger.info(f"Dataset loaded: {len(self._dataset)} images")
            logger.debug(f"Dataset columns: {self._dataset.columns.tolist()}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise DatasetNotFoundError(f"Failed to load dataset: {str(e)}")
    
    def find_closest_image(
        self, 
        target_percentages: Dict[str, float]
    ) -> Dict:
        """
        Find the closest matching image based on terrain percentages
        Uses Manhattan distance (L1 norm)
        
        Args:
            target_percentages: Target composition percentages
                Expected keys: water, road, vegetation, buildings, land
            
        Returns:
            Dictionary with image information and actual percentages
            
        Raises:
            ValueError: If total percentage exceeds 100%
        """
        # Validate input
        total = sum(target_percentages.values())
        if total > 100:
            raise ValueError(f"Total percentage exceeds 100%! Current: {total}%")
        
        logger.debug(f"Searching for image with percentages: {target_percentages}")
        
        # Calculate distance to each image in dataset
        temp_data = self._dataset.copy()
        temp_data['distance'] = (
            abs(temp_data['Water'] - target_percentages.get('water', 0)) + 
            abs(temp_data['Road'] - target_percentages.get('road', 0)) + 
            abs(temp_data['Vegetation'] - target_percentages.get('vegetation', 0)) + 
            abs(temp_data['Building'] - target_percentages.get('buildings', 0)) + 
            abs(temp_data['Land'] - target_percentages.get('land', 0))
        )
        
        # Find closest match
        closest_row = temp_data.loc[temp_data['distance'].idxmin()]
        
        logger.info(f"Closest match found: {closest_row['Image_Path']} (distance: {closest_row['distance']:.2f})")
        
        return {
            'image_path': closest_row['Image_Path'],
            'water': float(closest_row['Water']),
            'road': float(closest_row['Road']),
            'vegetation': float(closest_row['Vegetation']),
            'buildings': float(closest_row['Building']),
            'land': float(closest_row['Land']),
            'distance': float(closest_row['distance'])
        }
    
    def search_and_load_image(
        self, 
        target_percentages: Dict[str, float]
    ) -> tuple[np.ndarray, Dict]:
        """
        Search for closest image and load it
        
        Args:
            target_percentages: Target terrain composition
            
        Returns:
            Tuple of (image_array, metadata_dict)
        """
        # Find closest match
        match_info = self.find_closest_image(target_percentages)
        
        # Convert mask path to image path
        mask_path = match_info['image_path']
        image_path = mask_path.replace('masks', 'images').replace('.png', '.jpg')
        
        # Load image
        try:
            image_rgb = self.image_service.load_image_from_path(image_path)
            logger.info(f"Loaded image from {image_path}")
            
            return image_rgb, match_info
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            raise InvalidImageError(f"Failed to load matched image: {str(e)}")
    
    def get_dataset_stats(self) -> Dict:
        """
        Get overall dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        if self._dataset is None:
            return {}
        
        stats = {
            'total_images': len(self._dataset),
            'avg_water': float(self._dataset['Water'].mean()),
            'avg_road': float(self._dataset['Road'].mean()),
            'avg_vegetation': float(self._dataset['Vegetation'].mean()),
            'avg_building': float(self._dataset['Building'].mean()),
            'avg_land': float(self._dataset['Land'].mean()),
        }
        
        logger.debug(f"Dataset statistics: {stats}")
        return stats
