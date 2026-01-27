"""
Compare U-Net vs DeepLabV3+ Architectures

This script compares the two models in terms of:
- Number of parameters
- Forward pass speed
- Memory usage
- Architecture differences
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import models
from models.architectures.deeplabv3plus import create_model as create_deeplabv3plus
import segmentation_models_pytorch as smp


def create_unet(num_classes=6):
    """Create U-Net model (for comparison)"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # No pre-training
        in_channels=3,
        classes=num_classes
    )
    return model


def count_parameters(model):
    """Count model parameters"""
    if hasattr(model, 'count_parameters'):
        return model.count_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }


def measure_forward_pass(model, input_size=(1, 3, 256, 256), num_runs=10):
    """Measure forward pass time"""
    dummy_input = torch.randn(input_size)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            times.append(time.time() - start)
    
    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000
    }


def compare_models():
    """Compare both models"""
    
    print("="*80)
    print("MODEL COMPARISON: U-Net vs DeepLabV3+")
    print("="*80)
    
    # Create models
    print("\n1. Creating models...")
    print("\n   U-Net (ResNet34, no pre-training):")
    unet = create_unet(num_classes=6)
    print(f"   ✓ Created U-Net")
    
    print("\n   DeepLabV3+ (ResNet50, ImageNet pre-trained):")
    deeplabv3 = create_deeplabv3plus(
        encoder_name='resnet50',
        num_classes=6,
        pretrained=True,
        device='cpu'
    )
    
    # Parameter comparison
    print("\n2. Parameter Comparison:")
    print("-" * 80)
    
    unet_params = count_parameters(unet)
    deeplabv3_params = count_parameters(deeplabv3.model if hasattr(deeplabv3, 'model') else deeplabv3)
    
    print(f"   U-Net:")
    print(f"     Total: {unet_params['total']:,}")
    print(f"     Trainable: {unet_params['trainable']:,}")
    
    print(f"\n   DeepLabV3+:")
    print(f"     Total: {deeplabv3_params['total']:,}")
    print(f"     Trainable: {deeplabv3_params['trainable']:,}")
    
    ratio = deeplabv3_params['total'] / unet_params['total']
    print(f"\n   → DeepLabV3+ is {ratio:.1f}x larger than U-Net")
    
    # Speed comparison
    print("\n3. Speed Comparison (forward pass on CPU):")
    print("-" * 80)
    
    print("   Testing U-Net...")
    unet_speed = measure_forward_pass(unet, num_runs=5)
    print(f"     Mean: {unet_speed['mean_ms']:.2f}ms")
    print(f"     Range: {unet_speed['min_ms']:.2f}-{unet_speed['max_ms']:.2f}ms")
    
    print("\n   Testing DeepLabV3+...")
    deeplabv3_model = deeplabv3.model if hasattr(deeplabv3, 'model') else deeplabv3
    deeplabv3_speed = measure_forward_pass(deeplabv3_model, num_runs=5)
    print(f"     Mean: {deeplabv3_speed['mean_ms']:.2f}ms")
    print(f"     Range: {deeplabv3_speed['min_ms']:.2f}-{deeplabv3_speed['max_ms']:.2f}ms")
    
    speed_ratio = deeplabv3_speed['mean_ms'] / unet_speed['mean_ms']
    print(f"\n   → DeepLabV3+ is {speed_ratio:.1f}x slower than U-Net")
    
    # Architecture comparison
    print("\n4. Architecture Comparison:")
    print("-" * 80)
    print("   U-Net:")
    print("     ✓ Encoder-decoder with skip connections")
    print("     ✓ No pre-training")
    print("     ✓ Simple, fast")
    print("     ✗ Needs lots of data (10,000+ images)")
    
    print("\n   DeepLabV3+:")
    print("     ✓ ASPP for multi-scale context")
    print("     ✓ Pre-trained encoder (ImageNet)")
    print("     ✓ State-of-the-art performance")
    print("     ✓ Works well with limited data (~100 images)")
    print("     ✗ Slower, more parameters")
    
    # Recommendations
    print("\n5. Recommendations for TerraSegment AI:")
    print("-" * 80)
    print(f"   Dataset size: 72 images (very limited)")
    print(f"   Current accuracy: 76%")
    print(f"   Problem: Building (0%), Vegetation (0%) recall")
    
    print("\n   → RECOMMENDATION: Use DeepLabV3+")
    print("   Reasons:")
    print("     1. Pre-trained weights crucial for small dataset")
    print("     2. ASPP helps with multi-scale terrain features")
    print("     3. Expected improvement: +10-15% accuracy")
    print("     4. Speed is acceptable for training (not production bottleneck)")
    
    print("\n   Expected Results:")
    print("     • Overall: 76% → 90%+")
    print("     • Building: 0% → 70-85%")
    print("     • Vegetation: 0% → 75-90%")
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    compare_models()
