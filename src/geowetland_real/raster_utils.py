from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


def crop_remote_asset(href: str, bbox_wgs84, out_path: str | Path) -> Path:
    """Stream a window from a remote COG-like raster and save a local crop."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_HTTP_MULTIRANGE="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84, densify_pts=21)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(window=win)
            transform = src.window_transform(win)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                driver="GTiff",
                tiled=True,
                compress="deflate",
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)
    return out_path


def robust_scale(arr: np.ndarray, low=2.0, high=98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [low, high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)
