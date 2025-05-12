# EM_SEG/src/unet_se.py
import segmentation_models_pytorch as smp

def get_unet_se():
    """
    U-Net baseline with Squeeze-and-Excitation (SCSE) attention in decoder.
    Only one line differs from baseline.
    """
    return smp.Unet(
        encoder_name="se_resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse",   # <-- 添加 SE
    )
