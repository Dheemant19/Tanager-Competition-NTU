"""Notebook-derived products for scenes in the coastal-water-bodies collection.

The calculations in this module intentionally reproduce the established
workflows in notebooks/Coastal/03_water_quality.ipynb and
03b_quantitative_turbidity.ipynb. They are not new retrieval methods.
"""

from __future__ import annotations

import base64
import json
import math
import threading
from collections import OrderedDict
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import numpy as np
from pyproj import Transformer

from api.composite import (
    MAX_PREVIEW_SIZE,
    MIN_PREVIEW_SIZE,
    cached_preview_channels,
    encode_rgba_png,
    nearest_band_index,
    query_int,
    query_value,
)
from api.spectrum import (
    DATA_FIELDS,
    PRODUCTS,
    SpectrumError,
    attr_array,
    clean_scalar,
    get_node,
    grid_info,
    load_manifest,
    open_hdf5,
    open_kerchunk_root,
)
from api.colormaps import hex_colors, rgba


COASTAL_COLLECTION = "coastal-water-bodies"
RELATIVE_TARGETS = (443.0, 560.0, 665.0, 708.0, 861.0)
MAX_CACHE_ITEMS = 12

_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def scene_is_coastal(scene: dict) -> bool:
    """Return whether the catalogue identifies a scene as coastal water."""

    collections = scene.get("collections") or [scene.get("collection")]
    return COASTAL_COLLECTION in collections


def scene_is_eligible(scene: dict) -> bool:
    """Expose the workflow for every catalogue-identified coastal scene."""

    return scene_is_coastal(scene)


def _clean_channel(channel: np.ndarray, fill_value) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32).copy()
    if fill_value is not None:
        values[np.isclose(values, fill_value)] = np.nan
    return values


def _read_qa(root, name: str, stride: int, shape: tuple[int, int]) -> np.ndarray:
    node = get_node(root, f"{DATA_FIELDS}/{name}")
    if node is None:
        return np.zeros(shape, dtype=bool)
    return np.asarray(node[::stride, ::stride]) != 0


def _finite_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask & np.isfinite(values)]
    if selected.size < 25:
        raise SpectrumError(422, "too few clean water pixels for coastal analysis")
    return selected


def _summary(values: np.ndarray, mask: np.ndarray, low: float, high: float) -> dict:
    selected = _finite_values(values, mask)
    low_value, high_value = np.nanpercentile(selected, [low, high])
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        raise SpectrumError(422, "coastal product has no usable spatial contrast")
    return {
        "range": [float(low_value), float(high_value)],
        "median": float(np.nanmedian(selected)),
        "valid_pixel_count": int(selected.size),
    }


def _render_product(
    values: np.ndarray,
    mask: np.ndarray,
    cmap_name: str,
    low_value: float,
    high_value: float,
) -> str:
    scaled = np.clip((values - low_value) / (high_value - low_value), 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    pixels = rgba(cmap_name, scaled)
    pixels[..., 3] = np.where(mask & np.isfinite(values), 255, 0).astype(np.uint8)
    encoded = base64.b64encode(encode_rgba_png(pixels)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _legend(cmap_name: str, low_value: float, high_value: float, stretch: str) -> dict:
    """Return a compact, explicit colour scale for the rendered PNG."""

    positions = np.linspace(0.0, 1.0, 5)
    return {
        "colors": hex_colors(cmap_name, positions),
        "ticks": [float(low_value), float((low_value + high_value) / 2.0), float(high_value)],
        "stretch": stretch,
    }


def _grid_georeferencing(root, height: int, width: int) -> dict:
    """Describe the exact HDF-EOS ortho grid used to render the coastal image."""

    info = grid_info(root)
    if info["ydim"] != height or info["xdim"] != width:
        raise SpectrumError(422, "surface-reflectance dimensions do not match the HDF-EOS grid")
    inverse = Transformer.from_crs(f"EPSG:{info['epsg']}", "EPSG:4326", always_xy=True)
    projected_corners = (
        (info["ulx"], info["uly"]),
        (info["lrx"], info["uly"]),
        (info["lrx"], info["lry"]),
        (info["ulx"], info["lry"]),
    )
    corners = [inverse.transform(x, y) for x, y in projected_corners]
    longitudes = [corner[0] for corner in corners]
    latitudes = [corner[1] for corner in corners]
    return {
        "registration": "HDF-EOS ortho grid",
        "epsg": int(info["epsg"]),
        "bounds": [
            float(min(longitudes)),
            float(min(latitudes)),
            float(max(longitudes)),
            float(max(latitudes)),
        ],
        "corners": [[float(lon), float(lat)] for lon, lat in corners],
        "grid_shape": [height, width],
        "pixel_size_m": [float(abs(info["xres"])), float(abs(info["yres"]))],
    }


def _preview_note(stride: int) -> str:
    if stride == 1:
        return "Maps and summaries use the full-resolution grid."
    return f"Preview samples every {stride} source pixels; maps and summaries use this grid."


def _rank_correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    selected = mask & np.isfinite(left) & np.isfinite(right)
    x, y = left[selected], right[selected]
    if x.size < 25 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and sorted_values[stop] == sorted_values[start]:
                stop += 1
            result[order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
        return result

    correlation = np.corrcoef(ranks(x), ranks(y))[0, 1]
    return float(correlation) if np.isfinite(correlation) else None


def _window_indices(wavelengths: np.ndarray, good: np.ndarray, low: float, high: float) -> np.ndarray:
    indices = np.where((wavelengths >= low) & (wavelengths <= high) & good)[0]
    if indices.size == 0:
        raise SpectrumError(422, f"no good surface-reflectance bands in {low:.0f}-{high:.0f} nm")
    return indices


def _read_indices(
    dataset,
    wavelengths: np.ndarray,
    indices: np.ndarray,
    stride: int,
    product: dict,
    fill_value,
) -> np.ndarray:
    matches = [(int(index), float(wavelengths[index])) for index in indices]
    channels, _, _ = cached_preview_channels(
        dataset, matches, stride, product, PRODUCTS["ortho_sr"]["dataset"]
    )
    cleaned = [_clean_channel(channel, fill_value) for channel in channels]
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(cleaned), axis=0)


def _dogliotti_product(
    dataset,
    wavelengths: np.ndarray,
    good: np.ndarray,
    stride: int,
    product: dict,
    fill_value,
    nodata: np.ndarray,
    cloud: np.ndarray,
    cirrus: np.ndarray,
) -> dict:
    """Exact Phase-3b band windows, de-glint, switch, and saturation mask."""

    red_indices = _window_indices(wavelengths, good, 620.0, 670.0)
    nir_indices = _window_indices(wavelengths, good, 841.0, 876.0)
    swir_indices = _window_indices(wavelengths, good, 1600.0, 1620.0)
    green_indices = _window_indices(wavelengths, good, 558.0, 562.0)
    rho_red = _read_indices(dataset, wavelengths, red_indices, stride, product, fill_value)
    rho_nir = _read_indices(dataset, wavelengths, nir_indices, stride, product, fill_value)
    rho_swir = _read_indices(dataset, wavelengths, swir_indices, stride, product, fill_value)
    rho_green = _read_indices(dataset, wavelengths, green_indices, stride, product, fill_value)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (rho_green - rho_nir) / (rho_green + rho_nir)
    water = (
        ~nodata
        & ~cloud
        & ~cirrus
        & np.isfinite(rho_red)
        & (ndwi > 0)
    )
    rho_red_corrected = np.clip(rho_red - rho_swir, 0, None)
    rho_nir_corrected = np.clip(rho_nir - rho_swir, 0, None)

    a_red, c_red = 228.1, 0.1641
    a_nir, c_nir = 3078.9, 0.2112
    blend_low, blend_high = 0.05, 0.07

    def single_band(reflectance: np.ndarray, scale: float, saturation: float) -> np.ndarray:
        clipped = np.clip(reflectance, 0, 0.9 * saturation)
        return scale * clipped / (1 - clipped / saturation)

    red_fnu = single_band(rho_red_corrected, a_red, c_red)
    nir_fnu = single_band(rho_nir_corrected, a_nir, c_nir)
    nir_weight = np.clip((rho_red_corrected - blend_low) / (blend_high - blend_low), 0, 1)
    turbidity = (1 - nir_weight) * red_fnu + nir_weight * nir_fnu
    saturated = water & (
        (rho_red_corrected > 0.9 * c_red)
        | (rho_nir_corrected > 0.9 * c_nir)
    )
    valid = water & ~saturated & np.isfinite(turbidity)
    selected = _finite_values(turbidity, valid)
    high_value = float(np.nanpercentile(selected, 98))
    median = float(np.nanmedian(selected))
    p95 = float(np.nanpercentile(selected, 95))
    return {
        "key": "turbidity_fnu",
        "label": "Turbidity (FNU)",
        "units": "FNU",
        "range": [0.0, high_value],
        "median": median,
        "p95": p95,
        "water_pixel_count": int(water.sum()),
        "valid_pixel_count": int(selected.size),
        "image": _render_product(turbidity, valid, "turbo", 0.0, high_value),
        "legend": _legend("turbo", 0.0, high_value, "0 to 98th percentile"),
    }


def coastal_from_root(product: dict, root, max_size: int, workflow: str, source_kind: str) -> dict:
    dataset_path = PRODUCTS["ortho_sr"]["dataset"]
    dataset = get_node(root, dataset_path)
    if dataset is None:
        raise SpectrumError(422, f"{dataset_path} is missing")
    wavelengths = np.asarray(attr_array(dataset.attrs, "wavelengths") or [], dtype=float)
    good_values = attr_array(dataset.attrs, "good_wavelengths")
    good = np.asarray(good_values, dtype=bool) if good_values is not None else np.ones(wavelengths.size, dtype=bool)
    if good.size != wavelengths.size:
        good = np.ones(wavelengths.size, dtype=bool)

    height, width = int(dataset.shape[1]), int(dataset.shape[2])
    georeferencing = _grid_georeferencing(root, height, width)
    stride = max(1, int(math.ceil(max(height, width) / max_size)))
    fill_value = clean_scalar(dataset.attrs.get("_FillValue"))
    preview_shape = (
        int(math.ceil(height / stride)),
        int(math.ceil(width / stride)),
    )

    if workflow == "fnu":
        nodata = _read_qa(root, "nodata_pixels", stride, preview_shape)
        cloud = _read_qa(root, "beta_cloud_mask", stride, preview_shape)
        cirrus = _read_qa(root, "beta_cirrus_mask", stride, preview_shape)
        product_result = _dogliotti_product(
            dataset, wavelengths, good, stride, product, fill_value, nodata, cloud, cirrus
        )
        water_pixels = product_result["water_pixel_count"]
        return {
            "source": source_kind,
            "products": [product_result],
            "georeferencing": georeferencing,
            "method_notes": [
                "Scene-specific colour scale; compare values, not colours, across scenes.",
                _preview_note(stride),
                "Cloud, cirrus, nodata and saturated retrievals are excluded.",
                "The NDWI mask can retain inland water and isolated water-like pixels.",
                "FNU follows the notebook Dogliotti workflow after SWIR de-glint correction.",
            ],
            "qa": {
                "water_pixel_count": water_pixels,
                "preview_pixel_count": int(nodata.size),
                "water_fraction": float(water_pixels / max(1, nodata.size)),
                "cloud_fraction": float(cloud.mean()),
                "cdom_turbidity_rank_correlation": None,
                "ndci_positive_fraction": None,
            },
            "matched_wavelengths_nm": [],
            "source_shape": [height, width],
            "preview_shape": [int(nodata.shape[0]), int(nodata.shape[1])],
            "stride": stride,
            "batched_band_read": True,
            "band_cache_hits": 0,
        }

    matches = [nearest_band_index(wavelengths, target) for target in RELATIVE_TARGETS]
    for index, wavelength in matches:
        if index < good.size and not good[index]:
            raise SpectrumError(422, f"{wavelength:.1f} nm is flagged as a bad surface-reflectance band")

    channels, batched, cache_hits = cached_preview_channels(
        dataset, matches, stride, product, dataset_path
    )
    blue, green, red, red_edge, nir = [
        _clean_channel(channel, fill_value) for channel in channels
    ]
    shape = green.shape
    nodata = _read_qa(root, "nodata_pixels", stride, shape)
    cloud = _read_qa(root, "beta_cloud_mask", stride, shape)
    cirrus = _read_qa(root, "beta_cirrus_mask", stride, shape)

    valid = ~nodata & np.isfinite(green) & np.isfinite(nir)
    clean = valid & ~cloud & ~cirrus
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / (green + nir)
        water = clean & (ndwi > 0)
        turbidity = red / (1.0 - red / 0.25)
        ndci = (red_edge - red) / (red_edge + red)
        cdom = green / blue

    if int(water.sum()) < 25:
        raise SpectrumError(422, "scene has too few clean water pixels for coastal analysis")

    definitions = (
        ("relative_turbidity", "Relative turbidity", turbidity, "turbo", "unitless"),
        ("relative_cdom", "Relative CDOM", cdom, "magma", "ρ(560)/ρ(443)"),
        ("ndci", "Algal-bloom likelihood (NDCI)", ndci, "viridis", "NDCI"),
    )
    products = []
    summaries = {}
    for key, label, values, cmap_name, units in definitions:
        product_mask = water & np.isfinite(values)
        summary = _summary(values, product_mask, 2.0, 98.0)
        summaries[key] = summary
        products.append({
            "key": key,
            "label": label,
            "units": units,
            **summary,
            "image": _render_product(
                values, product_mask, cmap_name, summary["range"][0], summary["range"][1]
            ),
            "legend": _legend(
                cmap_name,
                summary["range"][0],
                summary["range"][1],
                "2nd to 98th percentile",
            ),
        })

    cdom_correlation = _rank_correlation(turbidity, cdom, water)
    if cdom_correlation is not None and cdom_correlation >= 0.75:
        products[1]["note"] = "Closely follows the turbidity pattern."
    ndci_values = _finite_values(ndci, water)
    positive_fraction = float(np.mean(ndci_values > 0))
    if summaries["ndci"]["median"] <= 0 or positive_fraction < 0.1:
        products[2]["note"] = "Weak algal-bloom signal."

    return {
        "source": source_kind,
        "products": products,
        "georeferencing": georeferencing,
        "method_notes": [
            "Scene-specific 2nd–98th percentile colour scales; compare values, not colours, across scenes.",
            _preview_note(stride),
            "Cloud, cirrus and nodata pixels are excluded.",
            "The NDWI mask can retain inland water and isolated water-like pixels.",
            "Relative turbidity and CDOM are indicators, not concentration measurements.",
        ],
        "qa": {
            "water_pixel_count": int(water.sum()),
            "preview_pixel_count": int(water.size),
            "water_fraction": float(water.mean()),
            "cloud_fraction": float(cloud.mean()),
            "cdom_turbidity_rank_correlation": cdom_correlation,
            "ndci_positive_fraction": positive_fraction,
        },
        "matched_wavelengths_nm": [match[1] for match in matches],
        "source_shape": [height, width],
        "preview_shape": [int(shape[0]), int(shape[1])],
        "stride": stride,
        "batched_band_read": batched,
        "band_cache_hits": cache_hits,
    }


def create_coastal_analysis(product: dict, max_size: int, workflow: str) -> dict:
    with ExitStack() as stack:
        root = open_kerchunk_root(product, stack)
        if root is not None:
            return coastal_from_root(product, root, max_size, workflow, "kerchunk")
        root = open_hdf5(product, stack)
        return coastal_from_root(product, root, max_size, workflow, "hdf5")


def coastal_response(params: dict) -> dict:
    scene_id = query_value(params, "scene_id")
    workflow = query_value(params, "workflow", "").lower()
    if not workflow:
        include_fnu = query_value(params, "include_fnu", "false").lower() in {"1", "true", "yes"}
        workflow = "fnu" if include_fnu else "relative"
    if workflow not in {"relative", "fnu"}:
        raise SpectrumError(400, "workflow must be relative or fnu")
    max_size = max(MIN_PREVIEW_SIZE, min(MAX_PREVIEW_SIZE, query_int(params, "max_size", 480)))
    cache_key = (scene_id, workflow, max_size)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _CACHE.move_to_end(cache_key)
            return {**cached, "cache_hit": True}

    manifest = load_manifest()
    scene = manifest.get("scenes", {}).get(scene_id)
    if scene is None:
        raise SpectrumError(404, f"scene not found: {scene_id}")
    if not scene_is_eligible(scene):
        raise SpectrumError(403, "coastal analysis requires a coastal-water-bodies scene")
    product = scene.get("products", {}).get("ortho_sr_hdf5")
    if not product:
        raise SpectrumError(404, "ortho surface reflectance is not listed for this scene")

    result = {
        "scene_id": scene_id,
        "workflow": workflow,
        **create_coastal_analysis(product, max_size, workflow),
        "cache_hit": False,
    }
    with _CACHE_LOCK:
        _CACHE[cache_key] = result
        _CACHE.move_to_end(cache_key)
        while len(_CACHE) > MAX_CACHE_ITEMS:
            _CACHE.popitem(last=False)
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
            self.send_json(200, coastal_response(parse_qs(urlparse(self.path).query)))
        except SpectrumError as exc:
            self.send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - API guardrail
            self.send_json(500, {"error": f"unexpected coastal API error: {exc}"})
