# unet_trans.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from segmentation_models_pytorch.encoders import get_encoder


def _resize_pos(old, new):
    if old.shape == new.shape:
        return old
    cls_tok, patch_tok = old[:, :1], old[:, 1:]
    dim = patch_tok.shape[-1]
    h0 = int(patch_tok.shape[1] ** 0.5)
    h1 = int((new.shape[1] - 1) ** 0.5)
    patch = patch_tok.reshape(1, h0, h0, dim).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, (h1, h1), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat([cls_tok, patch], 1)


class ViT(nn.Module):
    def __init__(self, img=256,
                 weight="weights/vit_small_patch16_224.safetensors"):
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            img_size=img, in_chans=3,
            num_classes=0, pretrained=False
        )
        st = load_file(weight)
        st = {k: v for k, v in st.items() if not k.startswith("head.")}
        st["pos_embed"] = _resize_pos(st["pos_embed"], self.vit.pos_embed)
        self.vit.load_state_dict(st, strict=False)
        self.dim = self.vit.embed_dim  # 384

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # 灰度→伪RGB
        B = x.size(0)
        t = self.vit.patch_embed(x)
        t = torch.cat([self.vit.cls_token.expand(B, -1, -1), t], 1)
        t = self.vit.pos_drop(t + self.vit.pos_embed)
        for blk in self.vit.blocks:
            t = blk(t)
        t = t[:, 1:].transpose(1, 2)
        h = int(t.shape[-1] ** 0.5)
        return t.reshape(B, self.dim, h, h)  # (B,384,H/patch,W/patch)


class DecoderBlock(nn.Module):
    """上采样 + 跳跃连接 + 两次 3×3 conv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # 1) 上采样 x
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # 2) 对齐 skip 到 x 的空间大小
        skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        # 3) 拼接并两次 conv
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


def get_transunet(img=256):
    # 1) ResNet-34 编码器
    res = get_encoder("resnet34", in_channels=1, depth=5, weights=None)
    vit = ViT(img)

    # 2) 定义解码器各 block
    # 编码器输出 channels = [64@64, 64@32,128@16,256@8,512@4]
    # 深度特征融合后 deep channels=512+384=896@16
    dec1 = DecoderBlock(in_ch=896, skip_ch=256, out_ch=512)  # 16→32
    dec2 = DecoderBlock(in_ch=512, skip_ch=128, out_ch=256)  # 32→64
    dec3 = DecoderBlock(in_ch=256, skip_ch=64,  out_ch=128)  # 64→128
    dec4 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)   # 128→256

    # 最后一层 segmentation head
    seg_head = nn.Conv2d(64, 1, 1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.res = res
            self.vit = vit
            self.dec1 = dec1
            self.dec2 = dec2
            self.dec3 = dec3
            self.dec4 = dec4
            self.head = seg_head

        def forward(self, x):
            # 编码器：得到 5 级特征
            feats = self.res(x)
            # deepest = feats[-1] = 512@4×4
            deepest = feats[-1]
            # ViT 特征 = 384@16×16
            v = self.vit(x)
            # 上采样 deepest → 16×16
            d_up = F.interpolate(deepest, size=v.shape[2:], mode="bilinear", align_corners=False)
            # 融合
            d = torch.cat([v, d_up], dim=1)  # 896@16×16

            # 跳跃特征（倒序取 feats 除去最深层的那四层）
            # feats = [64@64,64@32,128@16,256@8,512@4]
            skip3, skip2, skip1, skip0 = feats[-2], feats[-3], feats[-4], feats[-5]
            # decode
            x = self.dec1(d, skip3)  # →512@32
            x = self.dec2(x, skip2)  # →256@64
            x = self.dec3(x, skip1)  # →128@128
            x = self.dec4(x, skip0)  # →64@256
            return self.head(x)     # →1@256

    return Model()
