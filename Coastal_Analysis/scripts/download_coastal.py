"""
scripts/download_coastal.py

Download Borneo coastal Tanager scene assets.

The default download includes visual, quality-mask, geolocation, and thumbnail
assets. Pass `--hdf5` to also download the orthorectified radiance and
surface-reflectance cubes.

Usage:
    python scripts/download_coastal.py
    python scripts/download_coastal.py --hdf5

Files are saved to `data/coastal/<item_id>/`. Existing complete files are
skipped and partial downloads resume with HTTP Range requests.
"""

import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

# Configuration

HEADERS = {
    "User-Agent": "tanager-coastal-analysis/1.0 (akshatcrypts2022@gmail.com)"
}

# Borneo coastal scene
SCENE_ID = "20250302_030003_92_4001"

# Default assets
PHASE1_ASSETS = [
    "ortho_visual",
    "ortho_beta_udm",
    "geolocation_array",
    "thumbnail",
]

# Optional spectral cubes
HDF5_ASSETS = ["ortho_radiance_hdf5", "ortho_sr_hdf5"]

# URLs retained here for standalone use.
_BASE = "https://storage.googleapis.com/open-cogs/planet-stac/tanager1-release2-core-imagery"

ASSET_URLS = {
    SCENE_ID: {
        "ortho_visual":        f"{_BASE}/ortho_visual/{SCENE_ID}_ortho_visual.tif",
        "ortho_beta_udm":      f"{_BASE}/ortho_beta_udm/{SCENE_ID}_ortho_beta_udm.tif",
        "geolocation_array":   f"{_BASE}/geolocation_array/{SCENE_ID}_geolocation_array.tif",
        "thumbnail":           f"{_BASE}/thumbnail/{SCENE_ID}_thumb.png",
        "ortho_radiance_hdf5": f"{_BASE}/ortho_radiance_hdf5/{SCENE_ID}_ortho_radiance_hdf5.h5",
        "ortho_sr_hdf5":       f"{_BASE}/ortho_sr_hdf5/{SCENE_ID}_ortho_sr_hdf5.h5",
    },
}

# Downloading

def download_file(url: str, dest: Path, label: str) -> None:
    """Download a file, skipping complete files and resuming partial ones."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Get the expected size when available.
    try:
        head = requests.head(url, headers=HEADERS, timeout=30, allow_redirects=True)
        expected_size = int(head.headers.get("Content-Length", 0))
    except Exception:
        expected_size = 0

    existing_size = dest.stat().st_size if dest.exists() else 0

    # Skip complete files.
    if dest.exists() and expected_size > 0 and existing_size == expected_size:
        print(f"  [skip] {label}: already complete ({existing_size / 1e6:.1f} MB)")
        return

    # Resume partial downloads.
    resume_headers = dict(HEADERS)
    mode = "wb"
    if dest.exists() and existing_size > 0 and existing_size < expected_size:
        print(f"  [resume] {label}: resuming from {existing_size / 1e6:.1f} MB")
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


# Command line

def main():
    parser = argparse.ArgumentParser(description="Download Borneo coastal scene assets")
    parser.add_argument(
        "--hdf5",
        action="store_true",
        help="Also download both ortho HDF5 spectral cubes (~2-4 GB)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "coastal"

    print("=" * 60)
    print("Tanager Coastal Scene Download (Borneo / Kutai Timur)")
    print("=" * 60)
    print(f"Scene           : {SCENE_ID}")
    print(f"Output directory: {out_dir}")
    print()

    urls = ASSET_URLS[SCENE_ID]

    # Download default assets.
    print("--- Phase 1 assets ---")
    for asset_key in PHASE1_ASSETS:
        if asset_key not in urls:
            print(f"  [skip] {asset_key}: URL not in table")
            continue
        url = urls[asset_key]
        ext = url.split(".")[-1]
        dest = out_dir / SCENE_ID / f"{asset_key}.{ext}"
        download_file(url, dest, asset_key)
        time.sleep(0.15)

    # Download spectral cubes when requested.
    if args.hdf5:
        print()
        print("--- Phase 2 assets (ortho HDF5 cubes, ~2-4 GB total) ---")
        for asset_key in HDF5_ASSETS:
            if asset_key not in urls:
                print(f"  [skip] {asset_key}: URL not in table")
                continue
            url = urls[asset_key]
            dest = out_dir / SCENE_ID / f"{asset_key}.h5"
            download_file(url, dest, asset_key)
            time.sleep(0.5)
    else:
        print()
        print("HDF5 files skipped. Re-run with --hdf5 to download the spectral cubes.")
        print("  ortho_radiance_hdf5: TOA radiance (~1-2 GB)")
        print("  ortho_sr_hdf5: Surface reflectance (~1-2 GB)")
        print("Example:  python scripts\\download_coastal.py --hdf5")

    print()
    print("Done. Files saved to:", out_dir / SCENE_ID)
    print("Next: open notebooks/01_explore_scene.ipynb")


if __name__ == "__main__":
    main()
