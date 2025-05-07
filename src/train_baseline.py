import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, pandas as pd, datetime
from pathlib import Path
from tqdm import tqdm
from dataset import LucchiDataset, get_transforms
from unet_smp_baseline import get_unet
from metrics import dice, iou, precision, recall, f1   # ← 若改名则用 seg_metrics

ROOT = Path(__file__).resolve().parent.parent
PNG  = ROOT / "data" / "lucchi" / "png"

def main():
    # ---------- run dir ----------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = ROOT / "outputs" / "runs" / f"run_{ts}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR = RUN_DIR

    # ---------- dataset ----------
    train_ds = LucchiDataset(PNG/"trainval", PNG/"train.txt", get_transforms(True))
    val_ds   = LucchiDataset(PNG/"trainval", PNG/"val.txt",   get_transforms(False))

    # Windows 可先用 num_workers=0；若想提速再调大并保持 main() 包裹
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    # ---------- model ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_unet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # ---------- logging ----------
    cols = ["epoch","train_loss","val_loss","val_dice","val_iou",
            "val_precision","val_recall","val_f1"]
    log_df = pd.DataFrame(columns=cols)
    best_dice = 0.0

    for epoch in range(1, 21):
        # train
        model.train(); tr_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"E{epoch:02d}"):
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(); out = model(imgs)
            loss = criterion(out, masks); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # val
        model.eval(); vl_loss=vl_d=vl_i=vl_p=vl_r=vl_f=0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                vl_loss += criterion(out, masks).item()
                vl_d += dice(out,masks).item(); vl_i += iou(out,masks).item()
                vl_p += precision(out,masks).item(); vl_r += recall(out,masks).item()
                vl_f += f1(out,masks).item()
        n=len(val_loader); vl_loss/=n; vl_d/=n; vl_i/=n; vl_p/=n; vl_r/=n; vl_f/=n
        print(f"[E{epoch:02d}] trLoss={tr_loss:.4f}  valDice={vl_d:.4f}")

        log_df.loc[len(log_df)] = [epoch,tr_loss,vl_loss,vl_d,vl_i,vl_p,vl_r,vl_f]

        ckpt = CKPT_DIR / f"epoch{epoch:02d}.pth"
        torch.save(model.state_dict(), ckpt)
        if vl_d > best_dice:
            best_dice = vl_d
            torch.save(model.state_dict(), CKPT_DIR/"best.pth")

    # save csv
    csv_path = RUN_DIR / "metrics.csv"; log_df.to_csv(csv_path,index=False)

    # plots
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig(RUN_DIR/"loss_curve.png", dpi=300); plt.clf()

    plt.plot(log_df["epoch"], log_df["val_dice"], label="dice")
    plt.plot(log_df["epoch"], log_df["val_iou"], label="iou")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("score")
    plt.savefig(RUN_DIR/"dice_iou_curve.png", dpi=300)

if __name__ == "__main__":
    # Windows 多进程安全启动
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
