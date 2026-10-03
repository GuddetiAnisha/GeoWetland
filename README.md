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
- evaluates multiclass semantic segmentation using accuracy, macro precision/recall/F1, mean IoU, Dice, per-class metrics and a confusion matrix

## Validated multiclass test results

### Overall metrics

| Metric | Result |
|---|---:|
| Pixel Accuracy | **97.89%** |
| Macro Precision | **49.19%** |
| Macro Recall | **43.42%** |
| Macro F1 | **45.27%** |
| Mean IoU | **38.77%** |
| Macro Dice | **45.27%** |

The very high overall pixel accuracy is driven by the dominant class and should **not** be interpreted as balanced multiclass performance. Macro metrics and class-level results provide a more realistic picture.

### Per-class test metrics

| Class | Support | Precision | Recall | F1 | IoU | Dice |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 9,439,604 | 98.30% | 99.60% | **98.95%** | 97.92% | 98.95% |
| 2 | 5,034 | 15.15% | 21.75% | 17.86% | 9.81% | 17.86% |
| 3 | 329,000 | 83.30% | 52.34% | **64.29%** | 47.37% | 64.29% |
| 4 | 7,610 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Confusion matrix

Rows are true classes and columns are predicted classes, ordered 1 through 4.

```text
[
  [9401821,   4318,  33064,  401],
  [   3415,   1095,    524,    0],
  [ 152296,   1628, 172211, 2865],
  [   6480,    187,    943,    0]
]
```

### Result interpretation

- **Class 1:** excellent segmentation performance and the dominant source of overall pixel accuracy.
- **Class 2:** weak minority-class learning, with low precision and recall.
- **Class 3:** strong precision but moderate recall, so many true class-3 pixels are missed.
- **Class 4:** not learned in this run.
- The central limitation is **severe class imbalance**: class 1 contains about **9.44 million pixels**, while classes 2 and 4 contain only about **5 thousand** and **7.6 thousand** pixels.
- Because of the imbalance, the project reports **macro F1, mean IoU, Dice and per-class results** rather than relying on accuracy alone.

### Training behaviour

In the supplied epoch history for epochs 9--25, validation performance fluctuated despite generally decreasing training loss. Among those supplied epochs, epoch 22 reached the strongest macro F1 (**40.39%**) and mean IoU (**34.94%**), while epoch 25 had the lowest training loss (**0.0511**). This gap is consistent with the minority-class imbalance seen in the held-out test results.

The full machine-readable test metrics are stored in:

`results/test_metrics_multiclass.json`

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
- `results/test_metrics_multiclass.json`
- `results/predictions/` sample masks

## Important scientific wording

This project has been executed and evaluated on real geospatial inputs. The current model achieves **97.89% overall pixel accuracy**, but this number is strongly affected by class imbalance. For a balanced assessment, report the **45.27% macro F1**, **38.77% mean IoU**, per-class F1/IoU, and the fact that class 4 was not learned in this run.

## Data-source notes

- Sentinel-2 and Copernicus DEM access is through the public Microsoft Planetary Computer STAC API.
- Naturvårdsverket wetland data are downloaded from its public geodata server.
- The Swedish reference layer may contain multiple files. `prepare_real_data.py` automatically searches extracted GeoPackage/Shapefile/GeoJSON files and selects the largest readable polygon layer, or you can specify one with `--wetland-vector`.
