import torch.nn as nn
import segmentation_models_pytorch as smp


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


def get_unet_se_bneck():
    """
    U-Net (ResNet34) with **single SE block** on deepest encoder feature.
    Decoder无任何改动 → 保证尺寸匹配。
    """
    class UnetBneckSE(smp.Unet):
        def __init__(self):
            super().__init__(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=1,
                decoder_attention_type=None,   # 关闭原 scSE
            )
            self.se = SEBlock(self.encoder.out_channels[-1])  # deepest feature

        def forward(self, x):
            feats = self.encoder(x)          # list len=5, shallow→deep
            feats[-1] = self.se(feats[-1])   # 仅在 bottleneck 加 SE
            dec = self.decoder(*feats)       # smp 内部按正确顺序处理
            return self.segmentation_head(dec)

    return UnetBneckSE()
