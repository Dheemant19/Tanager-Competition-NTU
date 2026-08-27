"""Build persistent methane preview responses for serverless deployment.

The public basic-radiance objects are hundreds of megabytes. Downloading them
and running CWMF/MAG1C inside a Vercel request is not reliable, so this script
does that work offline and writes the exact JSON payload consumed by /api/ghg.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import h5py
import requests

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from api.ghg import (
    ANALYSIS_CACHE_DIR,
    REFERENCE_CACHE_DIR,
    load_scene_manifest,
    methane_from_root,
    reference_response,
)


def download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)


def build_scene(
    scene_id: str,
    layers: list[str],
    max_size: int,
    overwrite: bool,
) -> None:
    if "reference" in layers:
        reference_destination = REFERENCE_CACHE_DIR / f"{scene_id}.json"
        if overwrite or not reference_destination.exists():
            scene_meta = load_scene_manifest()["scenes"][scene_id]
            print(f"compute {scene_id} reference", flush=True)
            response = reference_response(scene_id, scene_meta, max_size)
            product = response["product"]
            reference_destination.parent.mkdir(parents=True, exist_ok=True)
            reference_destination.write_text(
                json.dumps(
                    {
                        "image": product["image"],
                        "range": product["range"],
                        "bounds": product["bounds"],
                        "valid_pixel_count": product["metrics"]["valid_pixel_count"],
                        "plumes": response.get("plumes", []),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            print(f"wrote {reference_destination.name}", flush=True)
        layers = [layer for layer in layers if layer != "reference"]
        if not layers:
            return

    pending = [
        layer
        for layer in layers
        if overwrite
        or not (ANALYSIS_CACHE_DIR / f"{scene_id}__{layer}__{max_size}.json").exists()
    ]
    if not pending:
        print(f"skip {scene_id}: caches exist", flush=True)
        return
    scene_meta = load_scene_manifest()["scenes"][scene_id]
    url = (scene_meta.get("basic_radiance") or {}).get("url")
    if not url:
        raise RuntimeError(f"{scene_id} has no public basic-radiance URL")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tanager-ghg-") as temporary:
        source = Path(temporary) / f"{scene_id}.h5"
        print(f"download {scene_id}", flush=True)
        download(url, source)
        with h5py.File(source, "r") as root:
            for layer in pending:
                print(f"compute {scene_id} {layer}", flush=True)
                payload = methane_from_root(
                    scene_id,
                    scene_meta,
                    root,
                    layer,
                    max_size,
                    "precomputed_hdf5",
                )
                ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                destination = ANALYSIS_CACHE_DIR / f"{scene_id}__{layer}__{max_size}.json"
                destination.write_text(
                    json.dumps(payload, allow_nan=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                print(
                    f"wrote {destination.name} ({destination.stat().st_size:,} bytes)",
                    flush=True,
                )
    print(f"finished {scene_id} in {time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", help="scene ID; repeat as needed")
    parser.add_argument("--layer", action="append", choices=("cwmf", "artifact", "reference"))
    parser.add_argument("--max-size", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scene_ids = args.scene or list(load_scene_manifest()["scenes"])
    layers = args.layer or ["cwmf", "artifact"]
    for scene_id in scene_ids:
        build_scene(scene_id, layers, args.max_size, args.overwrite)


if __name__ == "__main__":
    main()
