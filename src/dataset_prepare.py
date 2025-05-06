# -*- coding: utf-8 -*-
"""
Convert Lucchi TIF stacks to PNG slices.
生成：
  data/lucchi/png/{trainval,test}/{images,masks}/*.png
"""
from pathlib import Path
import tifffile as tiff
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "lucchi" / "raw"
OUT  = ROOT / "data" / "lucchi" / "png"

def save_stack(img_tif: Path, gt_tif: Path, out_sub: str):
    imgs = tiff.imread(img_tif)
    gts  = tiff.imread(gt_tif)
    assert imgs.shape == gts.shape
    img_dir = OUT / out_sub / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir  = OUT / out_sub / "masks";  gt_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(imgs.shape[0]), desc=out_sub):
        cv2.imwrite(str(img_dir / f"{idx:04d}.png"), imgs[idx])
        cv2.imwrite(str(gt_dir  / f"{idx:04d}.png"),  gts[idx])

if __name__ == "__main__":
    save_stack(RAW/"training.tif", RAW/"training_groundtruth.tif", "trainval")
    save_stack(RAW/"testing.tif",  RAW/"testing_groundtruth.tif",  "test")
