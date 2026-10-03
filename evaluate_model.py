from __future__ import annotations
import argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR
from src.geowetland_real.model import SmallUNet
from src.geowetland_real.metrics import segmentation_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    device = torch.device(a.device)
    d = np.load(PROCESSED_DIR / "dataset.npz")
    X, y, split = d["X"], d["y"], d["split"]
    idx = np.where(split == 2)[0]
    if not len(idx):
        raise RuntimeError("No test patches in spatial split.")

    ck = torch.load(MODELS_DIR / "best_unet.pt", map_location=device)
    model = SmallUNet(int(ck["in_channels"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(idx), 8):
            batch = torch.from_numpy(X[idx[i:i + 8]]).to(device)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy()[:, 0])

    prob = np.concatenate(probs)
    met = segmentation_metrics(y[idx], prob)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
    print(json.dumps(met, indent=2))

    out = RESULTS_DIR / "predictions"
    out.mkdir(exist_ok=True)
    for j, k in enumerate(idx[:5]):
        rgb = np.stack([X[k, 2], X[k, 1], X[k, 0]], axis=-1)
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(rgb)
        axs[0].set_title("Sentinel-2 RGB")
        axs[1].imshow(y[k], cmap="gray")
        axs[1].set_title("Reference")
        axs[2].imshow(prob[j] >= 0.5, cmap="gray")
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"sample_{j}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
