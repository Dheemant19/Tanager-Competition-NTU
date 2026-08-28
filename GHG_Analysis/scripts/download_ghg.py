"""
scripts/download_ghg.py

Downloads assets for the Brazil Offshore GHG-plumes scenes (Phase 1 + Phase 2).

Two Brazil strips from the same pass, 5 seconds apart:
  - Northern: 20250423_134021_00_4001  (strip_C plume, 795.7 kg/hr)
  - Southern: 20250423_134026_31_4001  (strip_A + strip_B plumes)

Phase 1 assets (small, <100 MB total) - downloaded for BOTH strips:
  ortho_visual        RGB GeoTIFF           ~5–20 MB
  ortho_ql_ch4        CH4 concentration     ~5–20 MB
  ql_ch4_json         CH4 plume polygons    <1 MB
  ortho_beta_udm      Cloud/quality mask    ~5 MB
  thumbnail           Small JPEG preview    <1 MB

Phase 2 assets (large, BOTH strips) - two HDF5 spectral cubes per strip:
  ortho_radiance_hdf5    TOA radiance cube           ~1–2 GB per strip
  ortho_sr_hdf5          Surface reflectance cube    ~1–2 GB per strip

We download BOTH HDF5 products for BOTH strips so we can:
  - Verify strip_C (northern) + strip_B (southern) plume spectra at pixel level
  - Compare radiance vs surface-reflectance for the CH4 matched filter (Phase 3)

Usage:
    python scripts\\download_ghg.py            # Phase 1 only (fast)
    python scripts\\download_ghg.py --hdf5     # Phase 1 + HDF5 for both strips (~4–8 GB)

Files are saved to:
    data/ghg/<item_id>/<asset_type>.<ext>

Resume support: if a file exists and matches the expected Content-Length,
the download is skipped. Partial files are resumed with HTTP Range requests.
"""

import sys
import time
import argparse
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "tanager-ghg-analysis/1.0 (akshatcrypts2022@gmail.com)"
}

# The two Brazil offshore GHG strips
SCENES = {
    "northern": "20250423_134021_00_4001",  # strip with confirmed plumes
    "southern": "20250423_134026_31_4001",  # our primary scene
}

# Phase 1 assets - downloaded for both strips
PHASE1_ASSETS = [
    "ortho_visual",
    "ortho_ql_ch4",
    "ql_ch4_json",
    "ortho_beta_udm",
    "thumbnail",
]

# Phase 2 assets - both HDF5 cubes, both strips
# strip_C (northern) + strip_B/A (southern) each need their own spectral cube
HDF5_ASSETS = ["ortho_radiance_hdf5", "ortho_sr_hdf5"]

# Inventory-derived URLs - hard-coded here so the script works standalone
# (built from the parquet in _inspect_scene.py)
ASSET_URLS = {
    "20250423_134021_00_4001": {
        "ortho_visual":            "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_visual/20250423_134021_00_4001_ortho_visual.tif",
        "ortho_ql_ch4":            "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_ql_ch4/20250423_134021_00_4001_ortho_ql_ch4.tif",
        "ql_ch4_json":             "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ql_ch4_json/20250423_134021_00_4001_ql_ch4_json.geojson",
        "ortho_beta_udm":          "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_beta_udm/20250423_134021_00_4001_ortho_beta_udm.tif",
        "thumbnail":               "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/thumbnail/20250423_134021_00_4001_thumb.png",
        "ortho_radiance_hdf5":     "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_radiance_hdf5/20250423_134021_00_4001_ortho_radiance_hdf5.h5",
        "ortho_sr_hdf5":           "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_sr_hdf5/20250423_134021_00_4001_ortho_sr_hdf5.h5",
        "basic_radiance_hdf5":     "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/basic_radiance_hdf5/20250423_134021_00_4001_basic_radiance_hdf5.h5",
    },
    "20250423_134026_31_4001": {
        "ortho_visual":            "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_visual/20250423_134026_31_4001_ortho_visual.tif",
        "ortho_ql_ch4":            "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_ql_ch4/20250423_134026_31_4001_ortho_ql_ch4.tif",
        "ql_ch4_json":             "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ql_ch4_json/20250423_134026_31_4001_ql_ch4_json.geojson",
        "ortho_beta_udm":          "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_beta_udm/20250423_134026_31_4001_ortho_beta_udm.tif",
        "thumbnail":               "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/thumbnail/20250423_134026_31_4001_thumb.png",
        "ortho_radiance_hdf5":     "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_radiance_hdf5/20250423_134026_31_4001_ortho_radiance_hdf5.h5",
        "ortho_sr_hdf5":           "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/ortho_sr_hdf5/20250423_134026_31_4001_ortho_sr_hdf5.h5",
        "basic_radiance_hdf5":     "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery/basic_radiance_hdf5/20250423_134026_31_4001_basic_radiance_hdf5.h5",
    },
}

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, label: str) -> None:
    """
    Download url to dest with a progress bar.
    - Skips if dest already exists and size matches Content-Length.
    - Resumes partial downloads using HTTP Range header.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # HEAD request to get expected size
    try:
        head = requests.head(url, headers=HEADERS, timeout=30, allow_redirects=True)
        expected_size = int(head.headers.get("Content-Length", 0))
    except Exception:
        expected_size = 0

    existing_size = dest.stat().st_size if dest.exists() else 0

    # Already complete
    if dest.exists() and expected_size > 0 and existing_size == expected_size:
        print(f"  [skip] {label} - already complete ({existing_size / 1e6:.1f} MB)")
        return

    # Partial - resume
    resume_headers = dict(HEADERS)
    mode = "wb"
    if dest.exists() and existing_size > 0 and existing_size < expected_size:
        print(f"  [resume] {label} - resuming from {existing_size / 1e6:.1f} MB")
        resume_headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
    else:
        print(f"  [download] {label}")

    try:
        r = requests.get(url, headers=resume_headers, stream=True, timeout=120)
        r.raise_for_status()

        total = expected_size - existing_size if mode == "ab" else expected_size

        with open(dest, mode) as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"    {dest.name}",
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        final_size = dest.stat().st_size
        print(f"  [done]  {label} -> {dest.name} ({final_size / 1e6:.1f} MB)")

    except Exception as e:
        print(f"  [ERROR] {label}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download GHG scene assets")
    parser.add_argument(
        "--hdf5",
        action="store_true",
        help="Also download both ORTHO HDF5 spectral cubes (~4-8 GB) for both strips",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Download the BASIC (sensor-geometry) radiance cube for both strips (~1.2 GB). "
             "Needed for the per-detector columnwise matched filter (Phase 3 destriping).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "ghg"

    print("=" * 60)
    print("Tanager GHG Scene Download")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()

    # Phase 1 - small assets, both strips
    print("--- Phase 1 assets (both strips) ---")
    for strip_name, item_id in SCENES.items():
        print(f"\n  Strip: {strip_name}  ({item_id})")
        urls = ASSET_URLS[item_id]
        for asset_key in PHASE1_ASSETS:
            if asset_key not in urls:
                print(f"  [skip] {asset_key} - URL not in table")
                continue
            url = urls[asset_key]
            ext = url.split(".")[-1]
            dest = out_dir / item_id / f"{asset_key}.{ext}"
            download_file(url, dest, f"{strip_name}/{asset_key}")
            time.sleep(0.15)  # polite pause between requests

    # Phase 2 - both HDF5 cubes, BOTH strips
    if args.hdf5:
        print()
        print("--- Phase 2 assets (HDF5 cubes for both strips, ~4-8 GB total) ---")
        for strip_name, item_id in SCENES.items():
            print(f"\n  Strip: {strip_name}  ({item_id})")
            for asset_key in HDF5_ASSETS:
                if asset_key not in ASSET_URLS[item_id]:
                    print(f"  [skip] {asset_key} - URL not in table")
                    continue
                url  = ASSET_URLS[item_id][asset_key]
                dest = out_dir / item_id / f"{asset_key}.h5"
                download_file(url, dest, f"{strip_name}/{asset_key}")
                time.sleep(0.5)
    # Phase 3 - basic (sensor-geometry) radiance, both strips, for per-detector destriping
    if args.basic:
        print()
        print("--- Phase 3 asset: BASIC sensor-geometry radiance (both strips, ~1.2 GB) ---")
        for strip_name, item_id in SCENES.items():
            print(f"\n  Strip: {strip_name}  ({item_id})")
            url  = ASSET_URLS[item_id]["basic_radiance_hdf5"]
            dest = out_dir / item_id / "basic_radiance_hdf5.h5"
            download_file(url, dest, f"{strip_name}/basic_radiance_hdf5")
            time.sleep(0.5)

    if not args.hdf5 and not args.basic:
        print()
        print("HDF5 files skipped. Re-run with --hdf5 to download spectral cubes for both strips.")
        print("  ortho_radiance_hdf5  - TOA radiance  (~1-2 GB per strip)")
        print("  ortho_sr_hdf5        - Surface reflectance  (~1-2 GB per strip)")
        print("Example:  python scripts\\download_ghg.py --hdf5")

    print()
    print("Done. Files saved to:", out_dir)
    print("Next: open notebooks/04_tanager_scene_overview.ipynb")


if __name__ == "__main__":
    main()
