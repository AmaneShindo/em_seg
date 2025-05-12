# EM_SEG/src/train_unet_se.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, pandas as pd, datetime
from pathlib import Path
from tqdm import tqdm

from dataset import LucchiDataset, get_transforms
# from unet_se import get_unet_se as get_model         # SE-U-Net
from unet_se_bneck import get_unet_se_bneck as get_model
from metrics import dice, iou, precision, recall, f1

RUN_PREFIX = "runSEbn_"

# ---------- mixed Dice + BCE loss ----------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        dice_loss = 1 - (2 * inter + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 0.5 * bce + 0.5 * dice_loss


ROOT = Path(__file__).resolve().parent.parent
PNG = ROOT / "data" / "lucchi" / "png"


def main() -> None:
    # ---------- run dir ----------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = ROOT / "outputs" / "runs" / f"{RUN_PREFIX}{ts}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- hyper ----------
    batch_size = 2
    accum = 4               # gradient accumulation steps => effective batch = 8
    epochs = 30
    patience = 8            # early-stop

    # ---------- dataset ----------
    train_ds = LucchiDataset(PNG / "trainval", PNG / "train.txt", get_transforms(True))
    val_ds = LucchiDataset(PNG / "trainval", PNG / "val.txt", get_transforms(False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # ---------- model ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model().to(device)

    # ---------- optim & sched ----------
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = DiceBCELoss()

    # ---------- logs ----------
    cols = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_f1",
    ]
    log_df = pd.DataFrame(columns=cols)

    best_dice, wait = 0.0, 0

    # ---------- train loop ----------
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for step, (imgs, masks) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch:02d}"), 1
        ):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks) / accum
            loss.backward()

            if step % accum == 0:
                opt.step()
                opt.zero_grad()
            tr_loss += loss.item() * accum
        tr_loss /= len(train_loader)
        scheduler.step()

        # ---------- validation ----------
        model.eval()
        vl_loss = vl_d = vl_i = vl_p = vl_r = vl_f = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                vl_loss += criterion(logits, masks).item()
                vl_d += dice(logits, masks).item()
                vl_i += iou(logits, masks).item()
                vl_p += precision(logits, masks).item()
                vl_r += recall(logits, masks).item()
                vl_f += f1(logits, masks).item()
        n = len(val_loader)
        vl_loss /= n
        vl_d /= n
        vl_i /= n
        vl_p /= n
        vl_r /= n
        vl_f /= n
        print(f"[E{epoch:02d}] trLoss={tr_loss:.4f}  valDice={vl_d:.4f}")

        # ---------- log & checkpoint ----------
        log_df.loc[len(log_df)] = [
            epoch,
            tr_loss,
            vl_loss,
            vl_d,
            vl_i,
            vl_p,
            vl_r,
            vl_f,
        ]
        torch.save(model.state_dict(), RUN_DIR / f"epoch{epoch:02d}.pth")

        if vl_d > best_dice:
            best_dice = vl_d
            wait = 0
            torch.save(model.state_dict(), RUN_DIR / "best.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ---------- save metrics ----------
    csv = RUN_DIR / "metrics.csv"
    log_df.to_csv(csv, index=False)

    # ---------- plots ----------
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.savefig(RUN_DIR / "loss_curve.png", dpi=300); plt.clf()

    plt.plot(log_df["epoch"], log_df["val_dice"], label="dice")
    plt.plot(log_df["epoch"], log_df["val_iou"], label="iou")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.legend()
    plt.savefig(RUN_DIR / "dice_iou_curve.png", dpi=300)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
