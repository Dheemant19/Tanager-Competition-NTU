"""On-demand three-band composites from Tanager orthorectified products."""

from __future__ import annotations

import json
import math
import struct
import threading
import zlib
from collections import OrderedDict
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import numpy as np

from api.spectrum import (
    DATA_FIELDS,
    PRODUCTS,
    SpectrumError,
    attr_array,
    clean_scalar,
    get_node,
    load_manifest,
    open_hdf5,
    open_kerchunk_root,
)


DEFAULT_WAVELENGTHS = (665.0, 560.0, 490.0)
MIN_PREVIEW_SIZE = 160
MAX_PREVIEW_SIZE = 720
MAX_COMPOSITE_CACHE_ITEMS = 24
MAX_BAND_CACHE_ITEMS = 36

INDEX_RECIPES = {
    "ndvi": {
        "label": "NDVI",
        "targets": (865.0, 665.0),
        "formula": "(NIR - red) / (NIR + red)",
        "palette": (
            (70, 43, 25),
            (181, 126, 54),
            (232, 218, 145),
            (130, 181, 80),
            (24, 105, 52),
        ),
    },
    "ndwi": {
        "label": "NDWI",
        "targets": (560.0, 865.0),
        "formula": "(green - NIR) / (green + NIR)",
        "palette": (
            (111, 78, 55),
            (218, 190, 132),
            (229, 236, 226),
            (92, 181, 190),
            (25, 77, 135),
        ),
    },
    "mndwi": {
        "label": "MNDWI",
        "targets": (560.0, 1610.0),
        "formula": "(green - SWIR1) / (green + SWIR1)",
        "palette": (
            (123, 77, 42),
            (222, 179, 104),
            (235, 238, 224),
            (77, 177, 200),
            (25, 68, 131),
        ),
    },
    "nbr": {
        "label": "NBR",
        "targets": (865.0, 2200.0),
        "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
        "palette": (
            (142, 30, 34),
            (222, 118, 53),
            (239, 220, 154),
            (130, 183, 91),
            (28, 105, 60),
        ),
    },
}

_COMPOSITE_CACHE: OrderedDict[tuple, tuple[bytes, dict]] = OrderedDict()
_COMPOSITE_CACHE_LOCK = threading.Lock()
_BAND_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_BAND_CACHE_LOCK = threading.Lock()


def query_value(params: dict, name: str, default: str | None = None) -> str:
    raw = params.get(name, [default])[0]
    if raw is None:
        raise SpectrumError(400, f"missing required query parameter: {name}")
    return str(raw)


def query_float(params: dict, name: str, default: float) -> float:
    raw = query_value(params, name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise SpectrumError(400, f"invalid {name}: {raw}") from exc
    if not math.isfinite(value):
        raise SpectrumError(400, f"invalid {name}: {raw}")
    return value


def query_int(params: dict, name: str, default: int) -> int:
    raw = query_value(params, name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise SpectrumError(400, f"invalid {name}: {raw}") from exc


def nearest_band_index(wavelengths: np.ndarray, target: float) -> tuple[int, float]:
    if wavelengths.size == 0 or not np.isfinite(wavelengths).any():
        raise SpectrumError(422, "product wavelengths are unavailable")
    index = int(np.nanargmin(np.abs(wavelengths - target)))
    return index, float(wavelengths[index])


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def encode_rgba_png(rgba: np.ndarray) -> bytes:
    """Encode a uint8 RGBA array without adding an image-library dependency."""

    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("RGBA image must have shape (height, width, 4) and dtype uint8")
    height, width, _ = rgba.shape
    rows = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows, level=6))
        + png_chunk(b"IEND", b"")
    )


def stretch_channel(values: np.ndarray, valid: np.ndarray, low_pct: float, high_pct: float) -> tuple[np.ndarray, dict]:
    finite_values = values[valid & np.isfinite(values)]
    if finite_values.size < 2:
        raise SpectrumError(422, "selected band has too few valid preview pixels")
    low, high = np.nanpercentile(finite_values, [low_pct, high_pct])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise SpectrumError(422, "selected band has no usable contrast")
    scaled = np.clip((values - low) / (high - low), 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    return np.rint(scaled * 255).astype(np.uint8), {
        "low_value": float(low),
        "high_value": float(high),
    }


def read_preview_channels(dataset, band_matches: list[tuple[int, float]], stride: int) -> tuple[list[np.ndarray], bool]:
    """Read all unique bands together when the array backend supports it."""

    indices = [index for index, _ in band_matches]
    unique_indices = sorted(set(indices))
    try:
        cube = np.asarray(
            dataset[unique_indices, ::stride, ::stride],
            dtype=np.float32,
        )
        if cube.ndim == 2:
            cube = cube[np.newaxis, ...]
        by_index = {index: cube[position] for position, index in enumerate(unique_indices)}
        return [by_index[index] for index in indices], True
    except (IndexError, NotImplementedError, TypeError, ValueError):
        return [
            np.asarray(dataset[index, ::stride, ::stride], dtype=np.float32)
            for index in indices
        ], False


def cached_preview_channels(
    dataset,
    band_matches: list[tuple[int, float]],
    stride: int,
    product: dict,
    dataset_path: str,
) -> tuple[list[np.ndarray], bool, int]:
    """Reuse downsampled bands when users explore several recipes for one scene."""

    cache_id = product.get("ref_path") or product.get("local_path") or product.get("url")
    if not cache_id or str(cache_id).startswith("memory:"):
        channels, batched = read_preview_channels(dataset, band_matches, stride)
        return channels, batched, 0

    requested_indices = [index for index, _ in band_matches]
    cached_by_index = {}
    with _BAND_CACHE_LOCK:
        for index in set(requested_indices):
            key = (cache_id, dataset_path, index, stride)
            cached = _BAND_CACHE.get(key)
            if cached is not None:
                _BAND_CACHE.move_to_end(key)
                cached_by_index[index] = cached

    missing_matches = [
        match for match in band_matches
        if match[0] not in cached_by_index
    ]
    batched = False
    if missing_matches:
        loaded, batched = read_preview_channels(dataset, missing_matches, stride)
        with _BAND_CACHE_LOCK:
            for match, channel in zip(missing_matches, loaded):
                index = match[0]
                key = (cache_id, dataset_path, index, stride)
                _BAND_CACHE[key] = channel
                _BAND_CACHE.move_to_end(key)
                cached_by_index[index] = channel
            while len(_BAND_CACHE) > MAX_BAND_CACHE_ITEMS:
                _BAND_CACHE.popitem(last=False)

    cache_hits = len(set(requested_indices)) - len({match[0] for match in missing_matches})
    return [cached_by_index[index] for index in requested_indices], batched, cache_hits


def colourise_index(scaled: np.ndarray, palette: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    """Interpolate a compact scientific colour ramp from a 0-255 index image."""

    colours = np.asarray(palette, dtype=np.float32)
    position = scaled.astype(np.float32) / 255 * (len(colours) - 1)
    lower = np.floor(position).astype(np.int16)
    upper = np.minimum(lower + 1, len(colours) - 1)
    fraction = (position - lower)[..., np.newaxis]
    rgb = colours[lower] * (1 - fraction) + colours[upper] * fraction
    return np.rint(rgb).astype(np.uint8)


def composite_from_root(
    product_name: str,
    product: dict,
    root,
    targets: tuple[float, float, float],
    low_pct: float,
    high_pct: float,
    max_size: int,
    source_kind: str,
    recipe: str = "rgb",
) -> tuple[bytes, dict]:
    spec = PRODUCTS[product_name]
    dataset = get_node(root, spec["dataset"])
    if dataset is None:
        raise SpectrumError(422, f"{spec['dataset']} is missing")
    wavelengths = np.asarray(attr_array(dataset.attrs, "wavelengths") or [], dtype=float)
    recipe_definition = INDEX_RECIPES.get(recipe)
    requested_targets = recipe_definition["targets"] if recipe_definition else targets
    band_matches = [nearest_band_index(wavelengths, target) for target in requested_targets]
    good_wavelengths = attr_array(dataset.attrs, "good_wavelengths")
    if product_name == "ortho_sr" and good_wavelengths is not None:
        channel_names = ("A", "B") if recipe_definition else tuple("RGB")
        for channel, (index, wavelength) in zip(channel_names, band_matches):
            if index < len(good_wavelengths) and not bool(good_wavelengths[index]):
                raise SpectrumError(
                    422,
                    f"{channel} target maps to {wavelength:.1f} nm, which this surface-reflectance product flags as a bad band",
                )

    height, width = int(dataset.shape[1]), int(dataset.shape[2])
    stride = max(1, int(math.ceil(max(height, width) / max_size)))
    channels, batched_read, band_cache_hits = cached_preview_channels(
        dataset,
        band_matches,
        stride,
        product,
        spec["dataset"],
    )
    valid = np.ones(channels[0].shape, dtype=bool)
    fill_value = clean_scalar(dataset.attrs.get("_FillValue"))
    for channel in channels:
        valid &= np.isfinite(channel)
        if fill_value is not None:
            valid &= ~np.isclose(channel, fill_value)

    nodata_node = get_node(root, f"{DATA_FIELDS}/nodata_pixels")
    if nodata_node is not None:
        nodata = np.asarray(nodata_node[::stride, ::stride]) != 0
        valid &= ~nodata

    if recipe_definition:
        numerator, denominator_band = channels
        denominator = numerator + denominator_band
        valid &= np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
        index_values = np.zeros(numerator.shape, dtype=np.float32)
        np.divide(
            numerator - denominator_band,
            denominator,
            out=index_values,
            where=valid,
        )
        scaled, stretch = stretch_channel(index_values, valid, low_pct, high_pct)
        rendered_rgb = colourise_index(scaled, recipe_definition["palette"])
        stretches = [stretch]
    else:
        rendered = []
        stretches = []
        for channel in channels:
            scaled, stretch = stretch_channel(channel, valid, low_pct, high_pct)
            rendered.append(scaled)
            stretches.append(stretch)
        rendered_rgb = np.dstack(rendered)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack([rendered_rgb, alpha])
    png = encode_rgba_png(rgba)
    metadata = {
        "product": product_name,
        "source": source_kind,
        "render_mode": "index" if recipe_definition else "rgb",
        "recipe": recipe,
        "recipe_label": recipe_definition["label"] if recipe_definition else "RGB composite",
        "formula": recipe_definition["formula"] if recipe_definition else None,
        "requested_wavelengths_nm": list(requested_targets),
        "matched_wavelengths_nm": [match[1] for match in band_matches],
        "band_indices": [match[0] for match in band_matches],
        "percentile_stretch": [low_pct, high_pct],
        "channel_stretches": stretches,
        "source_shape": [height, width],
        "preview_shape": [int(rgba.shape[0]), int(rgba.shape[1])],
        "stride": stride,
        "batched_band_read": batched_read,
        "band_cache_hits": band_cache_hits,
        "cache_hit": False,
    }
    return png, metadata


def create_composite(
    product_name: str,
    product: dict,
    targets: tuple[float, float, float],
    low_pct: float,
    high_pct: float,
    max_size: int,
    recipe: str = "rgb",
) -> tuple[bytes, dict]:
    with ExitStack() as stack:
        root = open_kerchunk_root(product, stack)
        if root is not None:
            return composite_from_root(
                product_name, product, root, targets, low_pct, high_pct, max_size, "kerchunk", recipe
            )
        root = open_hdf5(product, stack)
        return composite_from_root(
            product_name, product, root, targets, low_pct, high_pct, max_size, "hdf5", recipe
        )


def composite_response(params: dict) -> tuple[bytes, dict]:
    scene_id = query_value(params, "scene_id")
    product_name = query_value(params, "product", "ortho_sr").replace("_hdf5", "")
    if product_name not in PRODUCTS:
        raise SpectrumError(400, f"unknown product: {product_name}")
    recipe = query_value(params, "recipe", "rgb").lower()
    if recipe != "rgb" and recipe not in INDEX_RECIPES:
        raise SpectrumError(400, f"unknown composite recipe: {recipe}")
    if recipe in INDEX_RECIPES and product_name != "ortho_sr":
        raise SpectrumError(400, "calculated index recipes require surface reflectance")
    targets = tuple(
        query_float(params, channel, default)
        for channel, default in zip(("r", "g", "b"), DEFAULT_WAVELENGTHS)
    )
    if recipe == "rgb" and any(target < 376 or target > 2500 for target in targets):
        raise SpectrumError(400, "R, G and B wavelengths must be between 376 and 2500 nm")
    low_pct = query_float(params, "low", 2.0)
    high_pct = query_float(params, "high", 98.0)
    if not (0 <= low_pct < high_pct <= 100):
        raise SpectrumError(400, "stretch percentiles must satisfy 0 <= low < high <= 100")
    max_size = query_int(params, "max_size", 320)
    max_size = max(MIN_PREVIEW_SIZE, min(MAX_PREVIEW_SIZE, max_size))

    cache_key = (scene_id, product_name, recipe, targets, low_pct, high_pct, max_size)
    with _COMPOSITE_CACHE_LOCK:
        cached = _COMPOSITE_CACHE.get(cache_key)
        if cached is not None:
            _COMPOSITE_CACHE.move_to_end(cache_key)
            png, metadata = cached
            return png, {**metadata, "cache_hit": True}

    manifest = load_manifest()
    scene = manifest.get("scenes", {}).get(scene_id)
    if scene is None:
        raise SpectrumError(404, f"scene not found: {scene_id}")
    manifest_key = PRODUCTS[product_name]["manifest_key"]
    product = scene.get("products", {}).get(manifest_key)
    if not product:
        raise SpectrumError(404, f"{manifest_key} is not listed for this scene")
    result = create_composite(product_name, product, targets, low_pct, high_pct, max_size, recipe)
    with _COMPOSITE_CACHE_LOCK:
        _COMPOSITE_CACHE[cache_key] = result
        _COMPOSITE_CACHE.move_to_end(cache_key)
        while len(_COMPOSITE_CACHE) > MAX_COMPOSITE_CACHE_ITEMS:
            _COMPOSITE_CACHE.popitem(last=False)
    return result


class handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            png, metadata = composite_response(parse_qs(urlparse(self.path).query))
            metadata_text = json.dumps(metadata, separators=(",", ":"))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "X-Tanager-Composite")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("X-Tanager-Composite", metadata_text)
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
        except SpectrumError as exc:
            self.send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - API guardrail
            self.send_json(500, {"error": f"unexpected composite API error: {exc}"})
