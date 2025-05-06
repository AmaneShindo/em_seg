import cv2, torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomCrop(512, 512, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.CenterCrop(512, 512, always_apply=True),
            ToTensorV2(),
        ])

class LucchiDataset(Dataset):
    def __init__(self, root: Path, txt_list: Path, transforms=None):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.ids = [line.strip() for line in open(txt_list)]
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img  = cv2.imread(str(self.img_dir / img_id), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_dir / img_id), cv2.IMREAD_GRAYSCALE)
        img = img.astype("float32") / 255.0
        mask = (mask > 0).astype("float32")

        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
            mask = mask.unsqueeze(0)
        else:
            img = torch.tensor(img).unsqueeze(0)
            mask = torch.tensor(mask).unsqueeze(0)

        return img, mask
