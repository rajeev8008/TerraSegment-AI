"""
Model loading and management
"""
import os
from typing import Optional
import tensorflow as tf

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import ModelNotFoundError, ModelLoadError

logger = get_logger(__name__)


class ModelLoader:
    """Handles model loading and caching"""
    
    def __init__(self):
        self._model: Optional[tf.keras.Model] = None
        self._model_path: Optional[str] = None
    
    def load_model(self, model_path: Optional[str] = None) -> tf.keras.Model:
        """
        Load the segmentation model
        Uses cached model if already loaded
        
        Args:
            model_path: Path to model file (uses settings.model_path if None)
            
        Returns:
            Loaded Keras model
            
        Raises:
            ModelNotFoundError: If model file doesn't exist
            ModelLoadError: If model loading fails
        """
        # Use provided path or default from settings
        model_path = model_path or settings.model_path
        
        # Return cached model if already loaded and path hasn't changed
        if self._model is not None and self._model_path == model_path:
            logger.debug(f"Using cached model from {model_path}")
            return self._model
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise ModelNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first using: python training/train.py"
            )
        
        # Load model
        try:
            logger.info(f"Loading model from {model_path}...")
            self._model = tf.keras.models.load_model(
                model_path,
                compile=False  # Skip compilation for faster loading
            )
            self._model_path = model_path
            
            logger.info(f"✓ Model loaded successfully from {model_path}")
            logger.debug(f"Model input shape: {self._model.input_shape}")
            logger.debug(f"Model output shape: {self._model.output_shape}")
            
            return self._model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded"""
        return self._model is not None
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        if self._model is not None:
            logger.info("Unloading model from memory")
            del self._model
            self._model = None
            self._model_path = None


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance (singleton)"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
