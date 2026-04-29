import timm
import torch.nn as nn

def build_baseline_model(arch: str, num_classes: int, pretrained: bool = True):
    """
    Standard off-the-shelf models for comparison table.
    Uses same timm library, same input resolution.
    """
    if arch == 'resnet101':
        model = timm.create_model('resnet101', pretrained=pretrained, num_classes=num_classes)
    elif arch == 'vit_base_patch16':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    elif arch == 'efficientnet_b5':
        model = timm.create_model('efficientnet_b5', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown baseline: {arch}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Baseline Model: {arch}")
    print(f"  Parameters: {n_params:,} total")
    print(f"  Output classes: {num_classes}")
    
    return model
