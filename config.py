from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "real"
S2_DIR = DATA / "sentinel2"
DEM_DIR = DATA / "dem"
WETLAND_DIR = DATA / "wetlands"
PROCESSED_DIR = DATA / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Small southern-Sweden study region for reproducible, manageable downloads.
DEFAULT_BBOX = (13.0, 55.5, 13.5, 56.0)  # min lon, min lat, max lon, max lat
DEFAULT_DAYS_BACK = 120
DEFAULT_MAX_CLOUD = 25.0
PATCH_SIZE = 128
SEED = 42

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Naturvårdsverket national wetland classification package.
WETLAND_ZIP_URL = (
    "https://geodata.naturvardsverket.se/nedladdning/vatmark/"
    "vatmark_nationell_t01t02_fklass.20210319.zip"
)
