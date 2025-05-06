from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # ← 修正
PNG  = ROOT / "data" / "lucchi" / "png" / "trainval" / "images"

ids = sorted([p.name for p in PNG.glob("*.png")])   # 0–164
train_ids = ids[:140]
val_ids   = ids[140:]

txt_root = PNG.parents[1]          # png/
(txt_root / "train.txt").write_text("\n".join(train_ids))
(txt_root / "val.txt").write_text("\n".join(val_ids))
print("train:", len(train_ids), "val:", len(val_ids))
