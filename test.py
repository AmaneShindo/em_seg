# import torch, torchvision
# print("PyTorch:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import timm, torch
timm.create_model('vit_small_patch16_224', pretrained=True)