from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
from config import S2_DIR, DEM_DIR, WETLAND_DIR, PROCESSED_DIR, DEFAULT_BBOX, PATCH_SIZE, SEED
from src.geowetland_real.raster_utils import robust_scale

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX)
    p.add_argument("--wetland-vector", default=None)
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    return p.parse_args()


def find_vector(explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    root = WETLAND_DIR / "extracted"
    candidates = [p for p in root.rglob("*") if p.suffix.lower() in {".gpkg", ".shp", ".geojson"}]
    if not candidates:
        raise FileNotFoundError("No wetland vector file found. Run download_wetlands.py or pass --wetland-vector PATH.")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def read_polygon_layer(path: Path):
    if path.suffix.lower() == ".gpkg":
        import pyogrio
        layers = pyogrio.list_layers(path)
        choices = []
        for name, _ in layers:
            try:
                g = gpd.read_file(path, layer=name)
                poly = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
                if len(poly):
                    choices.append((len(poly), name, poly))
            except Exception:
                pass
        if not choices:
            raise RuntimeError(f"No polygon layer found in {path}")
        choices.sort(reverse=True, key=lambda x: x[0])
        print("Selected GPKG layer:", choices[0][1])
        return choices[0][2]
    g = gpd.read_file(path)
    return g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def align_raster(path, ref_profile, ref_shape):
    with rasterio.open(path) as src:
        dest = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )
    return dest


def main():
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ref_path = S2_DIR / "B02.tif"
    if not ref_path.exists():
        raise FileNotFoundError("Missing Sentinel-2 bands. Run download_sentinel2.py first.")

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        h, w = ref.height, ref.width
        transform = ref.transform
        crs = ref.crs
        stack = []
        for b in BANDS:
            p = S2_DIR / f"{b}.tif"
            if not p.exists():
                raise FileNotFoundError(p)
            stack.append(align_raster(p, profile, (h, w)))

    b02, b03, b04, b08, b11, b12 = stack
    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    ndmi = (b08 - b11) / (b08 + b11 + eps)

    dem_path = DEM_DIR / "dem.tif"
    if not dem_path.exists():
        raise FileNotFoundError("Missing DEM. Run download_dem.py first.")
    dem = align_raster(dem_path, profile, (h, w))

    channels = stack + [ndvi, ndwi, ndmi, dem]
    channels = [robust_scale(x) for x in channels]
    image = np.stack(channels, axis=0).astype(np.float32)

    vec_path = find_vector(args.wetland_vector)
    print("Using wetland vector:", vec_path)
    wetlands = read_polygon_layer(vec_path)
    if wetlands.empty:
        raise RuntimeError("Selected reference vector has no polygon geometries.")
    wetlands = wetlands[wetlands.geometry.notna() & ~wetlands.geometry.is_empty].copy()
    if wetlands.crs is None:
        raise RuntimeError("Wetland reference layer has no CRS metadata.")

    roi = gpd.GeoDataFrame(geometry=[box(*args.bbox)], crs="EPSG:4326").to_crs(wetlands.crs)
    wetlands = gpd.clip(wetlands, roi)
    if wetlands.empty:
        raise RuntimeError("No wetland polygons overlap the selected bbox. Choose another --bbox.")
    wetlands = wetlands.to_crs(crs)

    label = rasterize(
        [(g, 1) for g in wetlands.geometry if g is not None and not g.is_empty],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    ps = args.patch_size
    patches, masks, coords = [], [], []
    positives, negatives = [], []

    for y in range(0, h - ps + 1, ps):
        for x in range(0, w - ps + 1, ps):
            X = image[:, y:y + ps, x:x + ps]
            Y = label[y:y + ps, x:x + ps]
            if not np.isfinite(X).all():
                continue
            rec = (X, Y, x, y, float(Y.mean()))
            (positives if Y.any() else negatives).append(rec)

    rng = np.random.default_rng(SEED)
    rng.shuffle(negatives)
    keep_neg = negatives[:max(len(positives) * 2, min(len(negatives), 50))]
    selected = positives + keep_neg
    if len(selected) < 6:
        raise RuntimeError(f"Only {len(selected)} usable patches were created. Increase the bbox or choose a wetter area.")

    rng.shuffle(selected)
    for X, Y, x, y, _ in selected:
        patches.append(X)
        masks.append(Y)
        coords.append((x, y))

    X = np.stack(patches).astype(np.float32)
    y = np.stack(masks).astype(np.uint8)
    coords = np.asarray(coords, dtype=np.int32)

    q1, q2 = np.quantile(coords[:, 0], [0.6, 0.8])
    split = np.where(coords[:, 0] <= q1, 0, np.where(coords[:, 0] <= q2, 1, 2)).astype(np.uint8)

    np.savez_compressed(PROCESSED_DIR / "dataset.npz", X=X, y=y, coords=coords, split=split)
    meta = {
        "channels": [*BANDS, "NDVI", "NDWI", "NDMI", "DEM"],
        "patch_size": ps,
        "n_patches": int(len(X)),
        "positive_patches": int(sum(m.any() for m in y)),
        "train": int((split == 0).sum()),
        "val": int((split == 1).sum()),
        "test": int((split == 2).sum()),
        "wetland_vector": str(vec_path),
    }
    (PROCESSED_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
