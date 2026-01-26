"""
Custom exceptions for TerraSegment application
"""


class TerraSegmentError(Exception):
    """Base exception for all TerraSegment errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class InvalidImageError(TerraSegmentError):
    """Raised when image validation fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class ModelNotFoundError(TerraSegmentError):
    """Raised when model file is not found"""
    def __init__(self, message: str = "Model file not found. Please train the model first."):
        super().__init__(message, status_code=500)


class ModelLoadError(TerraSegmentError):
    """Raised when model loading fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class InferenceError(TerraSegmentError):
    """Raised when model inference fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class DatasetNotFoundError(TerraSegmentError):
    """Raised when dataset files are not found"""
    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ImageTooLargeError(InvalidImageError):
    """Raised when uploaded image exceeds size limit"""
    def __init__(self, max_size_mb: int):
        super().__init__(f"Image too large. Maximum size is {max_size_mb}MB")


class UnsupportedImageFormatError(InvalidImageError):
    """Raised when image format is not supported"""
    def __init__(self, format: str):
        super().__init__(f"Unsupported image format: {format}. Only JPG and PNG are supported.")
