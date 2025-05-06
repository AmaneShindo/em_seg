import torch, matplotlib.pyplot as plt
from pathlib import Path
from dataset import LucchiDataset, get_transforms
from unet_smp_baseline import get_unet

ROOT = Path(__file__).resolve().parent.parent      # ← 修正
PNG  = ROOT / "data" / "lucchi" / "png"
CKPT = ROOT / "outputs" / "ckpts" / "best.pth"
OUT  = ROOT / "outputs" / "vis"; OUT.mkdir(parents=True, exist_ok=True)

ds = LucchiDataset(PNG/"trainval", PNG/"val.txt", get_transforms(False))
img, mask = ds[0]

model = get_unet()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
with torch.no_grad():
    pred = torch.sigmoid(model(img.unsqueeze(0)))[0,0].numpy()

fig = plt.figure(figsize=(9,3))
for i, (title, data) in enumerate(zip(
        ["Image", "GT", "Pred"], [img[0], mask[0], pred>0.5])):
    ax = fig.add_subplot(1,3,i+1); ax.set_title(title); ax.axis("off")
    ax.imshow(data, cmap="gray")
plt.tight_layout()
save_path = OUT / "sample.png"
plt.savefig(save_path, dpi=300); plt.close()
print("Saved visualisation →", save_path)
