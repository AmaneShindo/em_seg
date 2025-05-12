# EM_SEG/src/evaluate_test.py
import argparse, torch, pandas as pd
from pathlib import Path
from dataset import LucchiDataset, get_transforms
from metrics import dice, iou, precision, recall, f1

ROOT = Path(__file__).resolve().parent.parent
PNG  = ROOT / "data" / "lucchi" / "png"
RUNS_ROOT = ROOT / "outputs" / "runs"


# ---------- helper: choose run dir ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str,
                    help="folder name inside outputs/runs/. "
                         "If omitted, use the most recently modified one.")
    return ap.parse_args()


def pick_run_dir(run_arg: str | None) -> Path:
    if run_arg is None:
        run_dir = max(RUNS_ROOT.glob("*"), key=lambda p: p.stat().st_mtime)
    else:
        run_dir = RUNS_ROOT / run_arg
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
    return run_dir


# ---------- helper: choose proper model loader ----------
def get_model_loader(run_name: str):
    if run_name.startswith("runSEbn_"):        # ← 新增
        from unet_se_bneck import get_unet_se_bneck as _loader
    elif run_name.startswith("runSE_"):
        from unet_se import get_unet_se as _loader
    elif run_name.startswith("runViTSE_"):
        from unet_trans import get_transunet_se as _loader
    elif run_name.startswith("runViT_"):
        from unet_trans import get_transunet as _loader
    else:
        from unet_smp_baseline import get_unet as _loader
    return _loader



# ---------- main ----------
def main():
    args = parse_args()
    run_dir = pick_run_dir(args.run)
    ckpt = run_dir / "best.pth"
    print("Evaluate ckpt →", ckpt)

    # select correct model architecture
    get_model = get_model_loader(run_dir.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # # dataset & loader
    # ds = LucchiDataset(PNG / "test", None, get_transforms(False))
    # loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    # ------------ dataset & loader ------------
    if run_dir.name.startswith("runViT_"):   # ViT 系列 → 256 crop
        from albumentations import Compose, CenterCrop
        from albumentations.pytorch import ToTensorV2
        tfm = Compose([CenterCrop(256, 256, always_apply=True), ToTensorV2()])
    else:                                    # 其它模型 → 512 crop
        from dataset import get_transforms
        tfm = get_transforms(False)

    ds = LucchiDataset(PNG / "test", None, tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)



    # compute metrics
    m = dict(dice=0, iou=0, precision=0, recall=0, f1=0)
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            m["dice"] += dice(out, mask).item()
            m["iou"] += iou(out, mask).item()
            m["precision"] += precision(out, mask).item()
            m["recall"] += recall(out, mask).item()
            m["f1"] += f1(out, mask).item()
    for k in m:
        m[k] /= len(loader)

    print("Test metrics:", m)

    # append metrics to csv
    csv = run_dir / "metrics.csv"
    df = pd.read_csv(csv)
    df.loc[len(df)] = ["test", "-", "-",
                       m["dice"], m["iou"],
                       m["precision"], m["recall"], m["f1"]]
    df.to_csv(csv, index=False)
    print("appended →", csv)


if __name__ == "__main__":
    main()
