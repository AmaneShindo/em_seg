import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset import LucchiDataset, get_transforms
from unet_smp_baseline import get_unet

ROOT = Path(__file__).resolve().parent.parent      # ← 修正
PNG  = ROOT / "data" / "lucchi" / "png"
CKPT_DIR = ROOT / "outputs" / "ckpts"; CKPT_DIR.mkdir(parents=True, exist_ok=True)

def dice_coef(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum()
    return (2 * inter + eps) / (preds.sum() + targets.sum() + eps)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_unet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = LucchiDataset(PNG/"trainval", PNG/"train.txt", get_transforms(True))
    val_ds   = LucchiDataset(PNG/"trainval", PNG/"val.txt",   get_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    best_dice = 0.0
    for epoch in range(1, 21):
        model.train(); total_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward(); opt.step()
            total_loss += loss.item()

        model.eval(); dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                dice += dice_coef(logits, masks).item()
        dice /= len(val_loader)
        print(f"[E{epoch:02d}] loss={total_loss/len(train_loader):.4f}  valDice={dice:.4f}")

        torch.save(model.state_dict(), CKPT_DIR / f"epoch{epoch:02d}_dice{dice:.3f}.pth")
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")

if __name__ == "__main__":
    main()
