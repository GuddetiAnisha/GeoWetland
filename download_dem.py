from __future__ import annotations
import argparse
import rasterio
from rasterio.merge import merge
import pystac_client
import planetary_computer
from config import DEFAULT_BBOX, PC_STAC, DEM_DIR
from src.geowetland_real.raster_utils import crop_remote_asset


def parse_args():
    p = argparse.ArgumentParser(description="Download real Copernicus DEM GLO-30 tiles covering the study area.")
    p.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX)
    return p.parse_args()

def _choose_asset(item):
    preferred = ["data", "dem", "image"]
    for k in preferred:
        if k in item.assets:
            return item.assets[k].href
    for _, a in item.assets.items():
        href = a.href.lower()
        if href.endswith((".tif", ".tiff")):
            return a.href
    raise RuntimeError(f"No raster asset found for DEM item {item.id}; assets={list(item.assets)}")

def main():
    args = parse_args()
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=list(args.bbox), max_items=50)
    items = list(search.items())
    if not items:
        raise SystemExit("No Copernicus DEM tiles found for this bbox.")
    print(f"Found {len(items)} DEM tile(s).")
    tmp_paths = []
    for i, item in enumerate(items):
        href = _choose_asset(item)
        out = DEM_DIR / f"tile_{i:02d}.tif"
        print("Cropping DEM tile", item.id)
        crop_remote_asset(href, args.bbox, out)
        tmp_paths.append(out)

    srcs = [rasterio.open(p) for p in tmp_paths]
    mosaic, transform = merge(srcs)
    profile = srcs[0].profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, count=mosaic.shape[0], compress="deflate")
    out = DEM_DIR / "dem.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    print("Saved:", out)

if __name__ == "__main__":
    main()
