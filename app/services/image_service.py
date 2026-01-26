"""
Image processing service
Handles image validation, loading, and preprocessing
"""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from typing import Tuple, Optional

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import (
    InvalidImageError,
    ImageTooLargeError,
    UnsupportedImageFormatError,
)

logger = get_logger(__name__)


class ImageService:
    """Service for image processing operations"""
    
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}
    
    def validate_image_file(
        self, 
        filename: str, 
        file_size: int
    ) -> None:
        """
        Validate uploaded image file
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of file in bytes
            
        Raises:
            ImageTooLargeError: If file exceeds size limit
            UnsupportedImageFormatError: If format not supported
        """
        # Check file size
        if file_size > settings.max_image_size_bytes:
            logger.warning(
                f"Image too large: {file_size} bytes (max: {settings.max_image_size_bytes})"
            )
            raise ImageTooLargeError(settings.max_image_size_mb)
        
        # Check file format
        file_ext = filename.lower().split('.')[-1]
        if f".{file_ext}" not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported image format: {file_ext}")
            raise UnsupportedImageFormatError(file_ext)
        
        logger.debug(f"Image validation passed: {filename} ({file_size} bytes)")
    
    def load_image_from_bytes(
        self, 
        image_bytes: bytes
    ) -> np.ndarray:
        """
        Load image from bytes and convert to RGB numpy array
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image as RGB numpy array
            
        Raises:
            InvalidImageError: If image cannot be decoded
        """
        try:
            # Decode image
            img_array = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                raise InvalidImageError("Failed to decode image")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            logger.debug(f"Image loaded successfully: shape={image_rgb.shape}")
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise InvalidImageError(f"Failed to load image: {str(e)}")
    
    def load_image_from_path(
        self, 
        image_path: str
    ) -> np.ndarray:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as RGB numpy array
            
        Raises:
            InvalidImageError: If image cannot be loaded
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise InvalidImageError(f"Could not load image at path: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug(f"Image loaded from {image_path}: shape={image_rgb.shape}")
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {str(e)}")
            raise InvalidImageError(f"Failed to load image: {str(e)}")
    
    def resize_image(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image as numpy array
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        resized = cv2.resize(image, target_size)
        logger.debug(f"Image resized from {image.shape} to {resized.shape}")
        return resized
    
    def normalize_image(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image (0-255)
            
        Returns:
            Normalized image (0-1)
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized
    
    def image_to_base64(
        self, 
        image_rgb: np.ndarray
    ) -> str:
        """
        Convert numpy array to base64 string
        
        Args:
            image_rgb: Image as RGB numpy array
            
        Returns:
            Base64 encoded string
        """
        try:
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            buffered.seek(0)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.debug("Image converted to base64")
            return img_base64
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise InvalidImageError(f"Failed to convert image: {str(e)}")
    
    def preprocess_for_model(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Preprocess image for model inference
        Resizes to patch size and normalizes
        
        Args:
            image: Input RGB image
            
        Returns:
            Preprocessed image ready for model
        """
        # Resize to model input size
        target_size = (settings.patch_size, settings.patch_size)
        resized = self.resize_image(image, target_size)
        
        # Normalize
        normalized = self.normalize_image(resized)
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        logger.debug(f"Image preprocessed for model: shape={batched.shape}")
        return batched
