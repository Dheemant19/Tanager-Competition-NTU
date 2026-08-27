"""Methane analysis for Tanager Workbench.

The methane calculations reuse ``scripts/ghg_methods.py``.  The API reads the
native basic-radiance methane window, runs the established two-pass CWMF or
MAG1C workflow, and bins the geolocated native result into a north-up preview.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import threading
from collections import OrderedDict
from contextlib import ExitStack
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from pyproj import Transformer

from api.colormaps import rgba
from api.composite import encode_rgba_png, query_int, query_value
from api.spectrum import SpectrumError, clean_scalar, fsspec, get_node, open_hdf5


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

from api.ghg_methods import (
    BasicRadianceScene,
    DATA_FIELDS as BASIC_DATA_FIELDS,
    LATITUDE_PATH,
    LONGITUDE_PATH,
    RADIANCE_PATH,
    build_hitran_ch4_target,
    run_columnwise_mag1c,
    run_iterative_columnwise_matched_filter,
    synthetic_target_test,
)


SCENE_MANIFEST_PATH = APP_ROOT / "data" / "ghg_scene_manifest.json"
REFERENCE_CACHE_DIR = APP_ROOT / "data" / "ghg_reference_cache"
HITRAN_CACHE = APP_ROOT / "data" / "hitran_cache"
MIN_PREVIEW_SIZE = 160
MAX_PREVIEW_SIZE = 640
MAX_CACHE_ITEMS = 4

PALETTES = {
    "cwmf": ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
    "artifact": ["#050816", "#293b8f", "#158f9c", "#8fd744", "#f7fcb9"],
    "reference": ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
}

_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_CACHE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_scene_manifest() -> dict:
    if not SCENE_MANIFEST_PATH.exists():
        raise SpectrumError(500, "ghg_scene_manifest.json has not been generated")
    return json.loads(SCENE_MANIFEST_PATH.read_text(encoding="utf-8"))


def scene_for_id(scene_id: str) -> dict:
    scene = load_scene_manifest().get("scenes", {}).get(scene_id)
    if scene is None:
        raise SpectrumError(403, "methane analysis requires a GHG scene or the reviewed HCMC scene")
    return scene


def scene_is_methane_eligible(scene_id: str) -> bool:
    return scene_id in load_scene_manifest().get("scenes", {})


@lru_cache(maxsize=None)
def _cached_reference(scene_id: str) -> dict | None:
    """Load the exact 480-pixel reference response used by the workbench UI."""

    path = REFERENCE_CACHE_DIR / f"{scene_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _local_path(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    candidate = PROJECT_ROOT / relative_path
    return candidate if candidate.exists() else None


def _basic_product(scene: dict) -> dict:
    product = scene.get("basic_radiance") or {}
    path = _local_path(product.get("local_path"))
    return {
        "url": product.get("url"),
        "local_path": str(path) if path is not None else None,
    }


def _read_basic_scene(root) -> BasicRadianceScene:
    dataset = get_node(root, RADIANCE_PATH)
    if dataset is None:
        raise SpectrumError(422, f"{RADIANCE_PATH} is missing")
    wavelengths = np.asarray(dataset.attrs.get("wavelengths"), dtype=float).squeeze()
    fwhm = np.asarray(dataset.attrs.get("fwhm"), dtype=float).squeeze()
    selected = np.flatnonzero((wavelengths >= 2104.0) & (wavelengths <= 2459.0))
    if selected.size == 0 or not np.all(np.diff(selected) == 1):
        raise SpectrumError(422, "the 2.3 micrometre methane bands are unavailable")

    band_slice = slice(int(selected[0]), int(selected[-1]) + 1)
    radiance = np.asarray(dataset[band_slice], dtype=np.float32)
    fill_value = float(clean_scalar(dataset.attrs.get("_FillValue", -9999.0)))
    units_value = clean_scalar(dataset.attrs.get("Unit", "unknown"))
    units = units_value.decode("utf-8") if isinstance(units_value, bytes) else str(units_value)

    latitude_node = get_node(root, LATITUDE_PATH)
    longitude_node = get_node(root, LONGITUDE_PATH)
    fields = get_node(root, BASIC_DATA_FIELDS)
    if latitude_node is None or longitude_node is None or fields is None:
        raise SpectrumError(422, "basic-product geolocation or QA fields are missing")

    latitude = np.asarray(latitude_node, dtype=float)
    longitude = np.asarray(longitude_node, dtype=float)

    def qa_field(name: str, default: float = 0.0) -> np.ndarray:
        node = get_node(root, f"{BASIC_DATA_FIELDS}/{name}")
        if node is None:
            return np.full(latitude.shape, default, dtype=np.float32)
        return np.asarray(node)

    cloud = qa_field("beta_cloud_mask")
    cirrus = qa_field("beta_cirrus_mask")
    nodata = qa_field("nodata_pixels")
    sensor_zenith = qa_field("sensor_zenith")
    invalid_radiance = (
        ~np.all(np.isfinite(radiance), axis=0)
        | np.any(radiance <= fill_value + 1.0, axis=0)
        | (np.nanmedian(radiance, axis=0) <= 0.0)
    )
    invalid_qa = (
        (cloud == 1)
        | (cirrus == 1)
        | (nodata == 1)
        | (cloud == 255)
        | (cirrus == 255)
        | (nodata == 255)
    )
    valid = ~(invalid_radiance | invalid_qa)
    return BasicRadianceScene(
        radiance=radiance,
        wavelengths_nm=wavelengths[band_slice],
        fwhm_nm=fwhm[band_slice],
        latitude=latitude,
        longitude=longitude,
        sensor_zenith_deg=sensor_zenith,
        cloud=cloud,
        cirrus=cirrus,
        nodata=nodata,
        valid=valid,
        radiance_units=units,
    )


def _target_for_scene(scene: BasicRadianceScene) -> np.ndarray:
    try:
        target = build_hitran_ch4_target(
            scene.wavelengths_nm,
            scene.fwhm_nm,
            HITRAN_CACHE,
        )
    except (FileNotFoundError, ImportError) as exc:
        raise SpectrumError(503, f"methane spectroscopy is unavailable: {exc}") from exc
    check = synthetic_target_test(target)
    if check["recovered_ppm_m"] <= 0 or abs(check["relative_error"]) > 1.0e-8:
        raise SpectrumError(500, "methane target sign/unit validation failed")
    return target


def _north_up_grid(
    values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    mask: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Bin irregular native pixels into a north-up geographic preview."""

    geolocated = np.isfinite(latitude) & np.isfinite(longitude)
    selected = mask & np.isfinite(values) & geolocated
    if int(selected.sum()) < 25:
        raise SpectrumError(422, "too few valid pixels for methane mapping")
    footprint_latitude = latitude[geolocated]
    footprint_longitude = longitude[geolocated]
    lat = latitude[selected]
    lon = longitude[selected]
    west, east = float(np.nanmin(footprint_longitude)), float(np.nanmax(footprint_longitude))
    south, north = float(np.nanmin(footprint_latitude)), float(np.nanmax(footprint_latitude))
    lon_span = max(east - west, np.finfo(float).eps)
    lat_span = max(north - south, np.finfo(float).eps)
    mean_latitude = (south + north) / 2.0
    aspect = lon_span * max(0.15, math.cos(math.radians(mean_latitude))) / lat_span
    if aspect >= 1:
        width = max_size
        height = max(MIN_PREVIEW_SIZE, int(round(max_size / aspect)))
    else:
        height = max_size
        width = max(MIN_PREVIEW_SIZE, int(round(max_size * aspect)))

    columns = np.clip(np.rint((lon - west) / lon_span * (width - 1)).astype(int), 0, width - 1)
    rows = np.clip(np.rint((north - lat) / lat_span * (height - 1)).astype(int), 0, height - 1)
    flat_indices = rows * width + columns
    sums = np.zeros(height * width, dtype=np.float64)
    counts = np.zeros(height * width, dtype=np.int32)
    np.add.at(sums, flat_indices, values[selected])
    np.add.at(counts, flat_indices, 1)
    grid = np.full(height * width, np.nan, dtype=np.float32)
    occupied = counts > 0
    grid[occupied] = (sums[occupied] / counts[occupied]).astype(np.float32)
    return grid.reshape(height, width), occupied.reshape(height, width), [west, south, east, north]


def _finite_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    selected = values[mask & np.isfinite(values)]
    if selected.size < 25:
        raise SpectrumError(422, "too few finite methane values")
    return float(np.nanpercentile(selected, percentile))


def _render(values: np.ndarray, mask: np.ndarray, cmap_name: str, low: float, high: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise SpectrumError(422, "methane result has no usable spatial contrast")
    scaled = np.clip((values - low) / (high - low), 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    pixels = rgba(cmap_name, scaled)
    pixels[..., 3] = np.where(mask & np.isfinite(values), 238, 0).astype(np.uint8)
    encoded = base64.b64encode(encode_rgba_png(pixels)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def methane_from_root(scene_id: str, scene_meta: dict, root, layer: str, max_size: int, source_kind: str) -> dict:
    """Run a notebook-derived methane layer from an already opened basic cube."""

    scene = _read_basic_scene(root)
    target = _target_for_scene(scene)
    if layer == "artifact":
        result = run_columnwise_mag1c(scene, target)
        values = result.enhancement_vertical_ppm_m
        analysis_mask = result.valid & np.isfinite(values)
        background_counts = result.background_spectra_per_detector
        label = "Artifact-suppressed methane response"
        cmap_name = "viridis"
        significance = None
    else:
        result = run_iterative_columnwise_matched_filter(scene, target)
        values = result.enhancement_vertical_ppm_m
        analysis_mask = result.valid & np.isfinite(values)
        background_counts = result.background_spectra_per_detector
        label = "CWMF methane analysis"
        cmap_name = "plasma"
        significance = result.significance

    preview, preview_mask, bounds = _north_up_grid(
        values,
        scene.latitude,
        scene.longitude,
        analysis_mask,
        max_size,
    )
    high = _finite_percentile(values, analysis_mask, 99.5)
    low = max(0.0, _finite_percentile(values, analysis_mask, 50.0))
    if high <= low:
        low = _finite_percentile(values, analysis_mask, 2.0)
    processed_columns = int(np.sum(np.asarray(background_counts) > 0))
    preferred_columns = int(np.sum(np.asarray(background_counts) >= 7 * scene.wavelengths_nm.size))
    valid_values = values[analysis_mask]
    metrics = {
        "median_ppm_m": float(np.nanmedian(valid_values)),
        "p95_ppm_m": float(np.nanpercentile(valid_values, 95)),
        "processed_columns": processed_columns,
        "preferred_support_columns": preferred_columns,
        "total_columns": int(values.shape[1]),
        "valid_pixel_count": int(analysis_mask.sum()),
    }
    if significance is not None:
        finite_significance = significance[analysis_mask & np.isfinite(significance)]
        metrics["peak_significance_sigma"] = float(np.nanpercentile(finite_significance, 99.9))
        finite_noise = result.noise_ppm_m_per_detector[np.isfinite(result.noise_ppm_m_per_detector)]
        metrics["median_robust_noise_ppm_m"] = float(np.nanmedian(finite_noise)) if finite_noise.size else None

    return {
        "scene_id": scene_id,
        "workflow": "methane",
        "layer": layer,
        "source": source_kind,
        "evidence_type": "independent_radiance_retrieval",
        "product": {
            "key": layer,
            "label": label,
            "units": "ppm m",
            "range": [low, high],
            "palette": PALETTES[layer],
            "image": _render(preview, preview_mask, cmap_name, low, high),
            "bounds": bounds,
            "metrics": metrics,
        },
        "qa": {
            "methane_window_nm": [float(scene.wavelengths_nm[0]), float(scene.wavelengths_nm[-1])],
            "band_count": int(scene.wavelengths_nm.size),
            "valid_fraction": float(scene.valid.mean()),
        },
        "comparison_available": bool((scene_meta.get("published_reference") or {}).get("url") or (scene_meta.get("published_reference") or {}).get("local_path")),
    }


_TIFF_TYPES = {
    1: ("B", 1),  # BYTE
    2: ("c", 1),  # ASCII
    3: ("H", 2),  # SHORT
    4: ("I", 4),  # LONG
    12: ("d", 8),  # DOUBLE
}


def _tiff_tags(payload: bytes) -> tuple[str, dict[int, tuple]]:
    """Read the baseline tags used by the small published LZW GeoTIFFs."""

    if payload[:2] == b"II":
        endian = "<"
    elif payload[:2] == b"MM":
        endian = ">"
    else:
        raise SpectrumError(422, "published methane reference is not a TIFF")
    if len(payload) < 8 or struct.unpack_from(endian + "H", payload, 2)[0] != 42:
        raise SpectrumError(422, "published methane reference uses an unsupported TIFF format")

    directory_offset = struct.unpack_from(endian + "I", payload, 4)[0]
    if directory_offset + 2 > len(payload):
        raise SpectrumError(422, "published methane reference has an invalid TIFF directory")
    entry_count = struct.unpack_from(endian + "H", payload, directory_offset)[0]
    tags: dict[int, tuple] = {}
    for index in range(entry_count):
        entry_offset = directory_offset + 2 + index * 12
        if entry_offset + 12 > len(payload):
            raise SpectrumError(422, "published methane reference has a truncated TIFF directory")
        tag, value_type, count, value_offset = struct.unpack_from(endian + "HHII", payload, entry_offset)
        parser = _TIFF_TYPES.get(value_type)
        if parser is None:
            continue
        fmt, width = parser
        byte_count = count * width
        if byte_count <= 4:
            raw = struct.pack(endian + "I", value_offset)[:byte_count]
        else:
            raw = payload[value_offset : value_offset + byte_count]
        if len(raw) != byte_count:
            raise SpectrumError(422, "published methane reference has a truncated TIFF value")
        tags[tag] = struct.unpack(endian + fmt * count, raw)
    return endian, tags


def _decode_tiff_lzw(payload: bytes) -> bytes:
    """Decode TIFF's MSB-first LZW stream without a GDAL-sized dependency."""

    bit_offset = 0

    def read_code(width: int) -> int | None:
        nonlocal bit_offset
        if bit_offset + width > len(payload) * 8:
            return None
        value = 0
        for _ in range(width):
            value = (value << 1) | ((payload[bit_offset // 8] >> (7 - bit_offset % 8)) & 1)
            bit_offset += 1
        return value

    output = bytearray()
    dictionary = {code: bytes([code]) for code in range(256)}
    width = 9
    next_code = 258
    previous: bytes | None = None
    while True:
        code = read_code(width)
        if code is None or code == 257:  # end-of-information
            break
        if code == 256:  # clear
            dictionary = {entry: bytes([entry]) for entry in range(256)}
            width = 9
            next_code = 258
            previous = None
            continue
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code and previous is not None:
            entry = previous + previous[:1]
        else:
            raise SpectrumError(422, "published methane reference has invalid LZW data")
        output.extend(entry)
        if previous is not None and next_code < 4096:
            dictionary[next_code] = previous + entry[:1]
            next_code += 1
            # TIFF LZW uses the early-change convention at each code-width boundary.
            if next_code == (1 << width) - 1 and width < 12:
                width += 1
        previous = entry
    return bytes(output)


def _resample_average(
    values: np.ndarray,
    height: int,
    width: int,
    nodata: float | None,
) -> np.ndarray:
    """Area-average a raster to the same bounded preview policy as before."""

    source_height, source_width = values.shape
    if (source_height, source_width) == (height, width):
        return values.astype(np.float32, copy=False)

    def weights(source_size: int, target_size: int) -> np.ndarray:
        starts = np.linspace(0.0, source_size, target_size, endpoint=False)
        stops = np.linspace(0.0, source_size, target_size + 1)[1:]
        source_starts = np.arange(source_size, dtype=float)
        source_stops = source_starts + 1.0
        overlap = np.maximum(
            0.0,
            np.minimum(stops[:, None], source_stops[None, :])
            - np.maximum(starts[:, None], source_starts[None, :]),
        )
        return overlap / (stops - starts)[:, None]

    row_weights = weights(source_height, height)
    column_weights = weights(source_width, width)
    valid = np.ones(values.shape, dtype=np.float32)
    if nodata is not None:
        valid = (~np.isclose(values, nodata)).astype(np.float32)
    numerator = row_weights @ (values.astype(np.float32) * valid) @ column_weights.T
    denominator = row_weights @ valid @ column_weights.T
    return np.divide(
        numerator,
        denominator,
        out=np.full((height, width), float(nodata or 0.0), dtype=np.float32),
        where=denominator > 0,
    ).astype(np.float32)


def _read_reference_raster(reference: dict, max_size: int) -> tuple[np.ndarray, float | None, list[float]]:
    """Read the published single-band LZW GeoTIFF and return a bounded preview."""

    local = _local_path(reference.get("local_path"))
    if local is not None:
        payload = local.read_bytes()
    else:
        url = reference.get("url")
        if not url:
            raise SpectrumError(404, "published methane reference is unavailable")
        if fsspec is None:
            raise SpectrumError(503, "remote methane rasters require fsspec")
        with fsspec.open(url, "rb") as remote:
            payload = remote.read()

    _, tags = _tiff_tags(payload)
    required = {256, 257, 258, 259, 277, 317, 322, 323, 324, 325, 33550, 33922, 34735}
    if not required.issubset(tags):
        raise SpectrumError(422, "published methane reference lacks required GeoTIFF metadata")
    image_width, image_height = int(tags[256][0]), int(tags[257][0])
    if tags[258][0] != 8 or tags[259][0] != 5 or tags[277][0] != 1 or tags[317][0] != 2:
        raise SpectrumError(422, "published methane reference uses an unsupported pixel encoding")

    tile_width, tile_height = int(tags[322][0]), int(tags[323][0])
    tile_offsets, tile_sizes = tags[324], tags[325]
    tile_columns = math.ceil(image_width / tile_width)
    values = np.zeros((image_height, image_width), dtype=np.uint8)
    for index, (offset, size) in enumerate(zip(tile_offsets, tile_sizes)):
        decoded = _decode_tiff_lzw(payload[int(offset) : int(offset) + int(size)])
        expected_size = tile_width * tile_height
        if len(decoded) != expected_size:
            raise SpectrumError(422, "published methane reference has an invalid tile size")
        tile = np.frombuffer(decoded, dtype=np.uint8).reshape(tile_height, tile_width)
        tile = np.cumsum(tile, axis=1, dtype=np.uint16).astype(np.uint8)
        tile_row, tile_column = divmod(index, tile_columns)
        top, left = tile_row * tile_height, tile_column * tile_width
        bottom, right = min(top + tile_height, image_height), min(left + tile_width, image_width)
        values[top:bottom, left:right] = tile[: bottom - top, : right - left]

    nodata = None
    if 42113 in tags:
        try:
            nodata = float(b"".join(tags[42113]).rstrip(b"\x00").decode("ascii"))
        except ValueError:
            pass
    scale = max(1.0, max(image_height, image_width) / max_size)
    preview_height = max(1, int(round(image_height / scale)))
    preview_width = max(1, int(round(image_width / scale)))
    preview = _resample_average(values, preview_height, preview_width, nodata)

    geo_keys = tags[34735]
    projected_epsg = None
    for index in range(4, len(geo_keys), 4):
        key_id, location, count, value = geo_keys[index : index + 4]
        if key_id in {2048, 3072} and location == 0 and count == 1:
            projected_epsg = int(value)
            break
    if projected_epsg is None:
        raise SpectrumError(422, "published methane reference has no supported CRS")
    scale_x, scale_y, _ = tags[33550][:3]
    _, _, _, west, north, _ = tags[33922][:6]
    east = west + image_width * scale_x
    south = north - image_height * scale_y
    transformer = Transformer.from_crs(f"EPSG:{projected_epsg}", "EPSG:4326", always_xy=True)
    bounds = [float(value) for value in transformer.transform_bounds(west, south, east, north, densify_pts=21)]
    return preview, nodata, bounds


def _load_geojson(reference: dict) -> list[dict]:
    local = _local_path(reference.get("plume_local_path"))
    if local is not None:
        payload = json.loads(local.read_text(encoding="utf-8"))
    elif reference.get("plume_url") and fsspec is not None:
        with fsspec.open(reference["plume_url"], "rt") as handle:
            payload = json.load(handle)
    else:
        return []
    markers = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "Point" or not coordinates or len(coordinates) < 2:
            continue
        properties = feature.get("properties") or {}
        markers.append({
            "longitude": float(coordinates[0]),
            "latitude": float(coordinates[1]),
            "label": str(properties.get("plume_id") or properties.get("name") or properties.get("id") or "Published plume"),
        })
    return markers


def reference_response(scene_id: str, scene_meta: dict, max_size: int) -> dict:
    """Render the official quicklook/reference with its own honest legend."""

    reviewed = None
    metrics_path = _local_path(scene_meta.get("reviewed_metrics_path"))
    if metrics_path is not None:
        reviewed = json.loads(metrics_path.read_text(encoding="utf-8")).get("carbon_mapper_comparison")

    cached = _cached_reference(scene_id) if max_size == 480 else None
    if cached is not None:
        return {
            "scene_id": scene_id,
            "workflow": "methane",
            "layer": "reference",
            "evidence_type": "published_provider_product",
            "product": {
                "key": "reference",
                "label": scene_meta.get("comparison_label", "Published methane reference"),
                "units": "ppm m",
                "range": cached["range"],
                "palette": PALETTES["reference"],
                "image": cached["image"],
                "bounds": cached["bounds"],
                "metrics": {
                    "valid_pixel_count": cached["valid_pixel_count"],
                    "reviewed_comparison": reviewed,
                },
            },
            "plumes": cached["plumes"],
            "comparison_available": True,
        }

    reference = scene_meta.get("published_reference") or {}
    values, nodata, bounds = _read_reference_raster(reference, max_size)
    mask = np.isfinite(values)
    if nodata is not None:
        mask &= ~np.isclose(values, nodata)
    mask &= values > 0
    if int(mask.sum()) < 25:
        mask = np.isfinite(values)
        if nodata is not None:
            mask &= ~np.isclose(values, nodata)
    low, high = (float(v) for v in np.nanpercentile(values[mask], [2, 98]))

    return {
        "scene_id": scene_id,
        "workflow": "methane",
        "layer": "reference",
        "evidence_type": "published_provider_product",
        "product": {
            "key": "reference",
            "label": scene_meta.get("comparison_label", "Published methane reference"),
            "units": "ppm m",
            "range": [low, high],
            "palette": PALETTES["reference"],
            "image": _render(values, mask, "magma", low, high),
            "bounds": bounds,
            "metrics": {"valid_pixel_count": int(mask.sum()), "reviewed_comparison": reviewed},
        },
        "plumes": _load_geojson(reference),
        "comparison_available": True,
    }


def create_methane_response(scene_id: str, layer: str, max_size: int) -> dict:
    scene_meta = scene_for_id(scene_id)
    if layer == "reference":
        return reference_response(scene_id, scene_meta, max_size)
    product = _basic_product(scene_meta)
    with ExitStack() as stack:
        root = open_hdf5(product, stack)
        source_kind = "local_hdf5" if product.get("local_path") else "remote_hdf5"
        return methane_from_root(scene_id, scene_meta, root, layer, max_size, source_kind)


def capabilities_response(scene_id: str) -> dict:
    scene = load_scene_manifest().get("scenes", {}).get(scene_id)
    if scene is None:
        raise SpectrumError(403, "methane analysis requires a GHG scene or the reviewed HCMC scene")
    reference = (scene or {}).get("published_reference") or {}
    return {
        "scene_id": scene_id,
        "methane_available": scene is not None,
        "artifact_available": scene is not None,
        "reference_available": bool(reference.get("url") or reference.get("local_path")),
        "comparison_label": (scene or {}).get("comparison_label"),
    }


def ghg_response(params: dict) -> dict:
    scene_id = query_value(params, "scene_id")
    workflow = query_value(params, "workflow", "capabilities").lower()
    if workflow == "capabilities":
        return capabilities_response(scene_id)
    if workflow != "methane":
        raise SpectrumError(400, "workflow must be capabilities or methane")

    layer = query_value(params, "layer", "cwmf").lower()
    if layer not in {"cwmf", "artifact", "reference"}:
        raise SpectrumError(400, "layer must be cwmf, artifact, or reference")
    max_size = max(MIN_PREVIEW_SIZE, min(MAX_PREVIEW_SIZE, query_int(params, "max_size", 480)))
    cache_key = (scene_id, layer, max_size)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _CACHE.move_to_end(cache_key)
            return {**cached, "cache_hit": True}
    result = {**create_methane_response(scene_id, layer, max_size), "cache_hit": False}
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
            self.send_json(200, ghg_response(parse_qs(urlparse(self.path).query)))
        except SpectrumError as exc:
            self.send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - API guardrail
            self.send_json(500, {"error": f"unexpected GHG API error: {exc}"})
