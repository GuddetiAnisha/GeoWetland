import numpy as np
from src.geowetland_real.metrics import segmentation_metrics
from src.geowetland_real.raster_utils import robust_scale


def test_metrics_perfect():
    y = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    m = segmentation_metrics(y, y.astype(float))
    assert m["f1"] > 0.999
    assert m["iou"] > 0.999


def test_robust_scale_range():
    a = np.arange(100, dtype=float).reshape(10, 10)
    b = robust_scale(a)
    assert b.min() >= 0
    assert b.max() <= 1
