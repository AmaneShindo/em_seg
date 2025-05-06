import segmentation_models_pytorch as smp

def get_unet():
    """
    1-in-1-out U-Net baseline with a ResNet34 encoder.
    """
    return smp.Unet(
        encoder_name="resnet34",     # ← 合法字符串
        encoder_weights=None,        # 或 "imagenet" 若想用预训练
        in_channels=1,
        classes=1,
    )
