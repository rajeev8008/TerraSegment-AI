"""
Data Augmentation Module
Implements various augmentation strategies for aerial image segmentation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_training_augmentation(heavy=True):
    """
    Get training augmentation pipeline
    
    Args:
        heavy (bool): If True, use heavy augmentation. If False, use light.
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    if heavy:
        return A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=0,
                p=0.7
            ),
            A.Transpose(p=0.5),
            
            # Color/Brightness transforms (aerial images vary in lighting)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),  # Contrast enhancement
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Weather/atmospheric effects
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                alpha_coef=0.08,
                p=0.2
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # Grid distortion (simulates different camera angles)
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1.0),
            ], p=0.3),
        ])
    else:
        # Light augmentation for validation
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
        ])


def get_validation_augmentation():
    """
    Get validation augmentation (minimal, just normalization)
    
    Returns:
        albumentations.Compose: Validation pipeline
    """
    return A.Compose([
        # No augmentation for validation, just ensure consistent format
    ])


def get_preprocessing(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Get preprocessing pipeline (normalization)
    
    Args:
        mean (list): Mean values for normalization (ImageNet by default)
        std (list): Std values for normalization (ImageNet by default)
    
    Returns:
        albumentations.Compose: Preprocessing pipeline
    """
    return A.Compose([
        A.Normalize(mean=mean, std=std),
    ])


def visualize_augmentation(image, mask, augmentation, num_samples=5):
    """
    Visualize augmentation effects
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Original mask
        augmentation: Augmentation pipeline
        num_samples (int): Number of augmented samples to generate
    
    Returns:
        list: List of (augmented_image, augmented_mask) tuples
    """
    samples = []
    for _ in range(num_samples):
        augmented = augmentation(image=image, mask=mask)
        samples.append((augmented['image'], augmented['mask']))
    return samples


# Example usage
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    
    # Load sample image and mask
    image = cv2.imread("sample_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread("sample_mask.png", cv2.IMREAD_GRAYSCALE)
    
    # Get augmentation
    aug = get_training_augmentation(heavy=True)
    
    # Visualize
    samples = visualize_augmentation(image, mask, aug, num_samples=6)
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab10')
    axes[1].set_title("Original Mask")
    axes[1].axis('off')
    
    # Augmented samples
    for i, (aug_img, aug_mask) in enumerate(samples):
        axes[2 + i*2].imshow(aug_img)
        axes[2 + i*2].set_title(f"Augmented {i+1}")
        axes[2 + i*2].axis('off')
        
        axes[3 + i*2].imshow(aug_mask, cmap='tab10')
        axes[3 + i*2].set_title(f"Aug Mask {i+1}")
        axes[3 + i*2].axis('off')
    
    plt.tight_layout()
    plt.savefig("augmentation_visualization.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to augmentation_visualization.png")
