# EM_SEG/src/dataset.py
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------- data augmentation ----------------
def get_transforms(train: bool = True):
    if train:
        return A.Compose(
            [
                A.RandomCrop(512, 512, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.CenterCrop(512, 512, always_apply=True),
                ToTensorV2(),
            ]
        )


# ---------------- dataset ----------------
class LucchiDataset(Dataset):
    """
    Args
    ----
    root : Path to `png/xxx` directory containing two sub-folders:
           `images/` and `masks/`
    txt_list : Path to txt file listing image filenames.
               If None, all pngs under images/ will be loaded (for test set).
    transforms : albumentations.Compose
    """

    def __init__(self, root: Path, txt_list: Path | None, transforms=None):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"

        if txt_list is None:
            # 自动遍历全部切片（测试集场景）
            self.ids = sorted(p.name for p in self.img_dir.glob("*.png"))
        else:
            self.ids = [line.strip() for line in open(txt_list)]

        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = cv2.imread(str(self.img_dir / img_id), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_dir / img_id), cv2.IMREAD_GRAYSCALE)

        img = img.astype("float32") / 255.0
        mask = (mask > 0).astype("float32")  # binarise

        if self.transforms:
            sample = self.transforms(image=img, mask=mask)
            img, mask = sample["image"], sample["mask"]  # img:(1,H,W)  mask:(H,W)
        else:
            img = torch.tensor(img).unsqueeze(0)
            mask = torch.tensor(mask)

        # 确保 mask 最终形状为 (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask
