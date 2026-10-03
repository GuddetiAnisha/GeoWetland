from __future__ import annotations
import zipfile
import requests
from tqdm import tqdm
from config import WETLAND_DIR, WETLAND_ZIP_URL


def main():
    WETLAND_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = WETLAND_DIR / "naturvardsverket_wetlands.zip"
    if not zip_path.exists():
        print("Downloading Swedish wetland reference data from Naturvårdsverket...")
        with requests.get(WETLAND_ZIP_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    else:
        print("Using existing:", zip_path)
    extract_dir = WETLAND_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    vectors = [p for p in extract_dir.rglob("*") if p.suffix.lower() in {".gpkg", ".shp", ".geojson"}]
    print(f"Extracted {len(vectors)} candidate vector file(s):")
    for p in vectors[:30]:
        print(" -", p)
    if not vectors:
        print("No directly readable polygon vector file was found. Inspect the extracted package format before preparing data.")

if __name__ == "__main__":
    main()
