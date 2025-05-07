import torch, pandas as pd
from pathlib import Path
from dataset import LucchiDataset, get_transforms
from unet_smp_baseline import get_unet
from metrics import dice, iou, precision, recall, f1

ROOT = Path(__file__).resolve().parent.parent
PNG  = ROOT / "data" / "lucchi" / "png"

# 找最新 run_* 目录
RUNS = sorted((ROOT/"outputs"/"runs").glob("run_*"), key=lambda p:p.stat().st_mtime)
RUN_DIR = RUNS[-1]; CKPT = RUN_DIR/"best.pth"
print("evaluate ckpt →", CKPT)

ds = LucchiDataset(PNG/"test", None, get_transforms(False))
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
device="cuda" if torch.cuda.is_available() else "cpu"
model=get_unet(); model.load_state_dict(torch.load(CKPT,map_location=device))
model.to(device).eval()

metrics=dict(dice=0,iou=0,precision=0,recall=0,f1=0)
with torch.no_grad():
    for imgs,masks in loader:
        imgs,masks=imgs.to(device),masks.to(device); out=model(imgs)
        metrics["dice"]+=dice(out,masks).item();     metrics["iou"]+=iou(out,masks).item()
        metrics["precision"]+=precision(out,masks).item()
        metrics["recall"]+=recall(out,masks).item(); metrics["f1"]+=f1(out,masks).item()
for k in metrics: metrics[k]/=len(loader)
print("Test metrics:", metrics)

csv_path = RUN_DIR / "metrics.csv"
df = pd.read_csv(csv_path)
df.loc[len(df)] = ["test","-","-",
                   metrics["dice"], metrics["iou"],
                   metrics["precision"], metrics["recall"], metrics["f1"]]
df.to_csv(csv_path,index=False); print("appended →", csv_path)
