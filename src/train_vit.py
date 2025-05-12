# EM_SEG/src/train_vit.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt, pandas as pd, datetime
from tqdm import tqdm

from dataset import LucchiDataset
from unet_trans import get_transunet
from metrics import dice, iou, precision, recall, f1

ROOT = Path(__file__).resolve().parent.parent
PNG  = ROOT / "data" / "lucchi" / "png"

# ---------------- transforms ----------------
CROP = 256          # 若后续 512 微调，改成 512 并重新训练 / finetune

def tfms(train=True):
    if train:
        return A.Compose([
            A.RandomCrop(CROP, CROP, always_apply=True),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.CenterCrop(CROP, CROP, always_apply=True),
            ToTensorV2(),
        ])

# ---------------- focal + dice ----------------
class FocalDice(nn.Module):
    def __init__(self, alpha=.8, gamma=2.):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, tgt):
        prob = torch.sigmoid(logits)
        # focal
        focal = -( self.alpha  * tgt * (1-prob)**self.gamma * prob.clamp(1e-6).log()
                 + (1-self.alpha)*(1-tgt)*prob**self.gamma * (1-prob).clamp(1e-6).log() ).mean()
        # dice
        dice_loss = 1 - (2*(prob*tgt).sum((1,2,3))+1)/((prob+tgt).sum((1,2,3))+1)
        return focal + dice_loss.mean()

# ---------------- main ----------------
def main():

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = ROOT / "outputs" / "runs" / f"runViT_{ts}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    batch, accum   = 2, 2            # 实际显存等效 batch=4
    epochs, warmup = 50, 10
    base_lr        = 3e-4

    train_ds = LucchiDataset(PNG/"trainval", PNG/"train.txt", tfms(True))
    val_ds   = LucchiDataset(PNG/"trainval", PNG/"val.txt", tfms(False))
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = get_transunet(img=CROP).to(device)

    # freeze ViT for warm-up
    for p in model.vit.parameters():
        p.requires_grad = False

    opt   = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                        lr=base_lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    criterion = FocalDice()

    cols = ["epoch","train_loss","val_loss","val_dice","val_iou",
            "val_precision","val_recall","val_f1"]
    log_df = pd.DataFrame(columns=cols); best, stagnate = 0, 0

    for ep in range(1, epochs+1):

        # -------- unfreeze ViT ----------
        if ep == warmup+1:
            for p in model.vit.parameters(): p.requires_grad = True
            opt.add_param_group({"params": model.vit.parameters()})

        # -------- train ----------
        model.train(); opt.zero_grad(); tr_loss = 0
        pbar = tqdm(tl, desc=f"E{ep:02d}")
        for step, (img, mask) in enumerate(pbar, 1):
            img, mask = img.to(device), mask.to(device)
            loss = criterion(model(img), mask) / accum
            loss.backward()
            if step % accum == 0:
                opt.step(); opt.zero_grad()
            tr_loss += loss.item()*accum
        tr_loss /= len(tl)

        # lr schedule
        sched.step()

        # -------- validation ----------
        model.eval(); vl_loss=vl_d=vl_i=vl_p=vl_r=vl_f=0
        with torch.no_grad():
            for img,mask in vl:
                img,mask = img.to(device),mask.to(device)
                out = model(img)
                vl_loss+=criterion(out,mask).item()
                vl_d+=dice(out,mask).item(); vl_i+=iou(out,mask).item()
                vl_p+=precision(out,mask).item(); vl_r+=recall(out,mask).item(); vl_f+=f1(out,mask).item()
        n=len(vl); vl_loss/=n; vl_d/=n; vl_i/=n; vl_p/=n; vl_r/=n; vl_f/=n
        print(f"[E{ep:02d}] trLoss={tr_loss:.4f}  valDice={vl_d:.4f}")

        # -------- logging / ckpt ----------
        log_df.loc[len(log_df)] = [ep,tr_loss,vl_loss,vl_d,vl_i,vl_p,vl_r,vl_f]
        torch.save(model.state_dict(), RUN_DIR/f"epoch{ep:02d}.pth")
        if vl_d > best:
            best, stagnate = vl_d, 0
            torch.save(model.state_dict(), RUN_DIR/"best.pth")
        else:
            stagnate += 1
            if stagnate >= 15:      # early-stop
                print(f"Early stop at epoch {ep}")
                break

    # -------- save metrics & curves --------
    log_df.to_csv(RUN_DIR/"metrics.csv", index=False)

    plt.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["val_loss"],   label="val_loss")
    plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR/"loss_curve.png", dpi=300); plt.clf()

    plt.plot(log_df["epoch"], log_df["val_dice"], label="dice")
    plt.plot(log_df["epoch"], log_df["val_iou"],  label="iou")
    plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR/"dice_iou_curve.png", dpi=300)

if __name__ == "__main__":
    import torch.multiprocessing as mp; mp.freeze_support(); main()
