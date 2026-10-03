# GeoWetland RealData GPU

Medium-level geospatial ML project for wetland mapping with **real public geospatial data**.

## Real datasets used

1. **Sentinel-2 Level-2A** imagery from Microsoft Planetary Computer's public STAC catalog (Copernicus Sentinel data).
2. **Copernicus DEM GLO-30** elevation tiles from the same STAC catalog.
3. **Swedish wetland reference data** from Naturvårdsverket's public geodata download service.

The downloader searches for the most recent low-cloud Sentinel-2 scene in the configured area/date window. Satellite imagery is therefore current-to-archive availability, not a live sensor stream.

## What the pipeline does

- searches recent Sentinel-2 scenes by bounding box and cloud cover
- streams/crops B02, B03, B04, B08, B11 and B12 bands
- downloads and mosaics Copernicus DEM tiles
- downloads and extracts Swedish wetland reference data
- uses GeoPandas for vector loading, clipping, CRS conversion and geometry handling
- uses Rasterio for raster alignment, reprojection and polygon rasterization
- derives NDVI, NDWI and NDMI
- creates 10-channel image patches
- uses geographic train/validation/test splitting to reduce spatial leakage
- trains a compact U-Net with PyTorch/CUDA
- reports accuracy, precision, recall, F1, IoU and Dice on the held-out spatial test region

## Default study area

The default bounding box is a manageable southern-Sweden area:

`[13.0, 55.5, 13.5, 56.0]`

You can change it in `config.py` or pass `--bbox min_lon min_lat max_lon max_lat` to the download scripts.

## VS Code / PowerShell

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install CUDA-enabled PyTorch separately if `torch.cuda.is_available()` is false. For NVIDIA systems, use the current command from https://pytorch.org/get-started/locally/ that matches your installed driver/CUDA support.

Check GPU:

```powershell
nvidia-smi
python check_gpu.py
```

Run the real-data pipeline:

```powershell
python download_sentinel2.py
python download_dem.py
python download_wetlands.py
python prepare_real_data.py
python train_model.py --device cuda --epochs 25 --batch-size 8
python evaluate_model.py --device cuda
```

Run tests:

```powershell
python -m pytest -q
```

## Outputs

- `data/real/sentinel2/*.tif` cropped real Sentinel-2 bands
- `data/real/dem/dem.tif` real Copernicus DEM crop
- `data/real/wetlands/` extracted Swedish wetland source data
- `data/real/processed/dataset.npz` aligned patch dataset
- `models/best_unet.pt` trained model
- `results/train_metrics.json`
- `results/test_metrics.json`
- `results/predictions/` sample masks

## Important scientific wording

After you successfully run this pipeline you can truthfully say the project uses real Sentinel-2, elevation and Swedish wetland-reference data. Do not claim a specific performance value until your own `results/test_metrics.json` has been generated.

## Data-source notes

- Sentinel-2 and Copernicus DEM access is through the public Microsoft Planetary Computer STAC API.
- Naturvårdsverket wetland data are downloaded from its public geodata server.
- The Swedish reference layer may contain multiple files. `prepare_real_data.py` automatically searches extracted GeoPackage/Shapefile/GeoJSON files and selects the largest readable polygon layer, or you can specify one with `--wetland-vector`.
