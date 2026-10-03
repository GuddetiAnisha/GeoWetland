import numpy as np


def segmentation_metrics(y_true, y_prob, threshold=0.5):
    yt = np.asarray(y_true).astype(bool).ravel()
    yp = np.asarray(y_prob).ravel() >= threshold

    tp = int(np.logical_and(yt, yp).sum())
    tn = int(np.logical_and(~yt, ~yp).sum())
    fp = int(np.logical_and(~yt, yp).sum())
    fn = int(np.logical_and(yt, ~yp).sum())

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
