from __future__ import annotations
import argparse, json, random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, SEED
from src.geowetland_real.model import SmallUNet
from src.geowetland_real.metrics import segmentation_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable. Run python check_gpu.py.")

    device = torch.device(a.device)
    d = np.load(PROCESSED_DIR / "dataset.npz")
    X, y, split = d["X"], d["y"], d["split"]

    def loader(code, shuffle):
        idx = np.where(split == code)[0]
        ds = TensorDataset(
            torch.from_numpy(X[idx]),
            torch.from_numpy(y[idx, None].astype(np.float32)),
        )
        return DataLoader(
            ds,
            batch_size=a.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    tr, va = loader(0, True), loader(1, False)
    model = SmallUNet(X.shape[1]).to(device)

    yt = y[split == 0]
    pos = float(yt.sum())
    neg = float(yt.size - pos)
    pw = max(1.0, min(20.0, neg / max(pos, 1.0)))

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    best = -1.0
    history = []

    for epoch in range(1, a.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        probs, truths = [], []
        with torch.no_grad():
            for xb, yb in va:
                p = torch.sigmoid(model(xb.to(device))).cpu().numpy()
                probs.append(p)
                truths.append(yb.numpy())

        if probs:
            met = segmentation_metrics(np.concatenate(truths), np.concatenate(probs))
            score = met["iou"]
        else:
            met = {"iou": 0.0}
            score = 0.0

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **met,
        }
        history.append(rec)
        print(rec)

        if score > best:
            best = score
            torch.save(
                {"model": model.state_dict(), "in_channels": X.shape[1]},
                MODELS_DIR / "best_unet.pt",
            )

    (RESULTS_DIR / "train_metrics.json").write_text(
        json.dumps({"best_val_iou": best, "history": history}, indent=2),
        encoding="utf-8",
    )
    print("Saved model:", MODELS_DIR / "best_unet.pt")


if __name__ == "__main__":
    main()
