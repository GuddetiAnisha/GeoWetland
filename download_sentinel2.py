from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import pystac_client
import planetary_computer
from config import DEFAULT_BBOX, DEFAULT_DAYS_BACK, DEFAULT_MAX_CLOUD, PC_STAC, S2_DIR
from src.geowetland_real.raster_utils import crop_remote_asset

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

def parse_args():
    p = argparse.ArgumentParser(description="Download/crop the latest low-cloud real Sentinel-2 L2A scene.")
    p.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, metavar=("MINLON","MINLAT","MAXLON","MAXLAT"))
    p.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    p.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    return p.parse_args()

def main():
    args = parse_args()
    S2_DIR.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days_back)
    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=list(args.bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": args.max_cloud}},
        max_items=100,
    )
    items = list(search.items())
    if not items:
        raise SystemExit("No Sentinel-2 L2A scenes matched the bbox/date/cloud filters. Increase --days-back or --max-cloud.")
    items.sort(key=lambda x: (x.datetime or datetime.min.replace(tzinfo=timezone.utc), -float(x.properties.get("eo:cloud_cover", 100))), reverse=True)
    recent = items[:10]
    item = min(recent, key=lambda x: float(x.properties.get("eo:cloud_cover", 100)))
    print("Selected scene:", item.id)
    print("Datetime:", item.datetime)
    print("Cloud cover:", item.properties.get("eo:cloud_cover"))
    print("Available assets:", sorted(item.assets.keys()))

    for band in BANDS:
        key = band if band in item.assets else band.lower()
        if key not in item.assets:
            raise RuntimeError(f"Band {band} not found. Available assets: {sorted(item.assets.keys())}")
        href = item.assets[key].href
        out = S2_DIR / f"{band}.tif"
        print(f"Cropping {band} -> {out}")
        crop_remote_asset(href, args.bbox, out)
    (S2_DIR / "scene_id.txt").write_text(item.id, encoding="utf-8")
    print("Sentinel-2 download complete.")

if __name__ == "__main__":
    main()
