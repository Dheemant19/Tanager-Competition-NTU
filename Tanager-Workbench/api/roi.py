"""Polygon region-of-interest spectra for the Tanager workbench.

The endpoint reads only the rectangular HDF5 window surrounding the polygon,
then keeps pixel centres that fall inside the polygon. Large accepted windows
use a deterministic spatial sampling stride for spectral statistics while
area and QA counts remain exact.
"""

from __future__ import annotations

import json
import math
import warnings
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler

import numpy as np

from api.spectrum import (
    DATA_FIELDS,
    PRODUCTS,
    SpectrumError,
    attr_array,
    attr_text,
    calculate_indices,
    clean_scalar,
    get_node,
    grid_info,
    load_manifest,
    open_hdf5,
    open_kerchunk_root,
    transformer_for,
    values_to_json,
)


MAX_ROI_VERTICES = 200
MAX_ROI_WINDOW_PIXELS = 262_144
MAX_ROI_SPECTRAL_WINDOW_PIXELS = 32_768


def parse_polygon(payload: dict) -> list[tuple[float, float]]:
    """Validate a GeoJSON Polygon and return its exterior lon/lat ring."""

    geometry = payload.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        raise SpectrumError(400, "geometry must be a GeoJSON Polygon")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates or not isinstance(coordinates[0], list):
        raise SpectrumError(400, "polygon exterior ring is missing")
    raw_ring = coordinates[0]
    if len(raw_ring) < 4:
        raise SpectrumError(400, "polygon needs at least three vertices")
    if len(raw_ring) > MAX_ROI_VERTICES:
        raise SpectrumError(413, f"polygon has more than {MAX_ROI_VERTICES} vertices")

    ring = []
    for coordinate in raw_ring:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            raise SpectrumError(400, "each polygon coordinate must contain longitude and latitude")
        try:
            lon, lat = float(coordinate[0]), float(coordinate[1])
        except (TypeError, ValueError) as exc:
            raise SpectrumError(400, "polygon coordinates must be numbers") from exc
        if not math.isfinite(lon) or not math.isfinite(lat) or not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
            raise SpectrumError(400, "polygon contains an invalid longitude or latitude")
        ring.append((lon, lat))

    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def polygon_centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    """Return a stable display centroid; fall back to the vertex mean."""

    points = ring[:-1]
    cross_sum = 0.0
    lon_sum = 0.0
    lat_sum = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(ring[:-1], ring[1:]):
        cross = lon1 * lat2 - lon2 * lat1
        cross_sum += cross
        lon_sum += (lon1 + lon2) * cross
        lat_sum += (lat1 + lat2) * cross
    if abs(cross_sum) < 1e-12:
        return (
            sum(lon for lon, _ in points) / len(points),
            sum(lat for _, lat in points) / len(points),
        )
    factor = 1.0 / (3.0 * cross_sum)
    return lon_sum * factor, lat_sum * factor


def polygon_mask(x_values: np.ndarray, y_values: np.ndarray, ring_xy: np.ndarray) -> np.ndarray:
    """Vectorised even/odd point-in-polygon test for pixel-centre grids."""

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    inside = np.zeros(x_grid.shape, dtype=bool)
    x1, y1 = ring_xy[-1]
    for x2, y2 in ring_xy:
        crosses = (y2 > y_grid) != (y1 > y_grid)
        denominator = y1 - y2
        if abs(denominator) < 1e-15:
            denominator = 1e-15
        boundary_x = (x1 - x2) * (y_grid - y2) / denominator + x2
        inside ^= crosses & (x_grid < boundary_x)
        x1, y1 = x2, y2
    return inside


def roi_window(info: dict, ring: list[tuple[float, float]]) -> tuple[int, int, int, int, np.ndarray]:
    transformer = transformer_for(info["epsg"])
    projected = np.asarray([transformer.transform(lon, lat) for lon, lat in ring], dtype=float)
    col_positions = (projected[:, 0] - info["ulx"]) / info["xres"]
    row_positions = (info["uly"] - projected[:, 1]) / info["yres"]

    c0 = max(0, int(math.floor(np.min(col_positions))))
    c1 = min(info["xdim"], int(math.ceil(np.max(col_positions))) + 1)
    r0 = max(0, int(math.floor(np.min(row_positions))))
    r1 = min(info["ydim"], int(math.ceil(np.max(row_positions))) + 1)
    if c0 >= c1 or r0 >= r1:
        raise SpectrumError(422, "polygon is outside the HDF5 grid")

    window_pixels = (r1 - r0) * (c1 - c0)
    if window_pixels > MAX_ROI_WINDOW_PIXELS:
        raise SpectrumError(
            413,
            f"area is too large ({window_pixels:,} candidate pixels); "
            f"draw a smaller polygon below {MAX_ROI_WINDOW_PIXELS:,} candidate pixels",
        )
    return r0, r1, c0, c1, projected


def qa_window(root, r0: int, r1: int, c0: int, c1: int, roi_mask: np.ndarray) -> tuple[dict, np.ndarray]:
    """Summarise QA flags and return pixels that are not marked nodata."""

    summary = {}
    nodata = np.zeros(roi_mask.shape, dtype=bool)
    names = {
        "beta_cloud_mask": "cloud",
        "beta_cirrus_mask": "cirrus",
        "nodata_pixels": "nodata",
    }
    selected_count = int(roi_mask.sum())
    for dataset_name, short_name in names.items():
        node = get_node(root, f"{DATA_FIELDS}/{dataset_name}")
        if node is None:
            summary[f"{short_name}_fraction"] = None
            continue
        values = np.asarray(node[r0:r1, c0:c1])
        flagged = values != 0
        if short_name == "nodata":
            nodata = flagged
        count = int(np.count_nonzero(flagged & roi_mask))
        summary[f"{short_name}_count"] = count
        summary[f"{short_name}_fraction"] = count / selected_count if selected_count else None
    return summary, roi_mask & ~nodata


def band_statistics(cube: np.ndarray) -> dict:
    """Calculate the two displayed per-band summaries while keeping JSON NaN-free."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(cube, axis=1)
        mean = np.nanmean(cube, axis=1)
    return {
        "median": values_to_json(median, None),
        "mean": values_to_json(mean, None),
        "valid_count": np.count_nonzero(np.isfinite(cube), axis=1).astype(int).tolist(),
    }


def spectral_sampling_grid(mask: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    """Choose a deterministic grid phase that retains the most usable pixels."""

    stride = max(
        1,
        int(math.ceil(math.sqrt(mask.size / MAX_ROI_SPECTRAL_WINDOW_PIXELS))),
    )
    if stride == 1:
        return stride, 0, 0, mask

    best_row = best_col = 0
    best_mask = mask[::stride, ::stride]
    best_count = int(best_mask.sum())
    for row_offset in range(stride):
        for col_offset in range(stride):
            candidate = mask[row_offset::stride, col_offset::stride]
            count = int(candidate.sum())
            if count > best_count:
                best_row, best_col = row_offset, col_offset
                best_mask, best_count = candidate, count
    return stride, best_row, best_col, best_mask


def extract_roi_product_from_root(
    product_name: str,
    product: dict,
    root,
    ring: list[tuple[float, float]],
    source_kind: str,
) -> dict:
    spec = PRODUCTS[product_name]
    dataset = get_node(root, spec["dataset"])
    if dataset is None:
        raise SpectrumError(422, f"{spec['dataset']} is missing")

    info = grid_info(root)
    r0, r1, c0, c1, ring_xy = roi_window(info, ring)
    x_values = info["ulx"] + (np.arange(c0, c1) + 0.5) * info["xres"]
    y_values = info["uly"] - (np.arange(r0, r1) + 0.5) * info["yres"]
    mask = polygon_mask(x_values, y_values, ring_xy)
    selected_count = int(mask.sum())
    if selected_count == 0:
        raise SpectrumError(422, "polygon does not include a pixel centre in this product")

    qa, usable_mask = qa_window(root, r0, r1, c0, c1, mask)
    analysis_stride, row_offset, col_offset, spectral_mask = spectral_sampling_grid(usable_mask)
    cube_window = np.asarray(
        dataset[
            :,
            r0 + row_offset:r1:analysis_stride,
            c0 + col_offset:c1:analysis_stride,
        ],
        dtype=np.float32,
    )
    cube = cube_window[:, spectral_mask]
    if cube.shape[1] == 0:
        raise SpectrumError(422, "polygon has no usable spectral pixels")
    fill_value = clean_scalar(dataset.attrs.get("_FillValue"))
    if fill_value is not None:
        cube[np.isclose(cube, fill_value)] = np.nan

    good_wavelengths = attr_array(dataset.attrs, "good_wavelengths")
    if product_name == "ortho_sr" and good_wavelengths is not None:
        good = np.asarray(good_wavelengths, dtype=bool)
        if good.size == cube.shape[0]:
            cube[~good, :] = np.nan

    statistics = band_statistics(cube)
    return {
        "available": True,
        "label": spec["label"],
        "dataset": spec["dataset"],
        "source": source_kind,
        "source_url": product.get("url"),
        "ref_path": product.get("ref_path") if source_kind == "kerchunk" else None,
        "units": attr_text(dataset.attrs, "Unit"),
        "wavelength_units": attr_text(dataset.attrs, "wavelengths_units") or "nm",
        "wavelengths": attr_array(dataset.attrs, "wavelengths"),
        "fwhm": attr_array(dataset.attrs, "fwhm"),
        "good_wavelengths": good_wavelengths,
        "values": statistics["median"],
        "statistics": statistics,
        "qa": qa,
        "roi": {
            "selected_pixel_count": selected_count,
            "data_pixel_count": int(usable_mask.sum()),
            "spectral_pixel_count": int(spectral_mask.sum()),
            "candidate_pixel_count": int(mask.size),
            "analysis_stride": analysis_stride,
            "statistics_are_sampled": analysis_stride > 1,
            "area_m2": float(selected_count * abs(info["xres"] * info["yres"])),
            "pixel_size_m": [float(info["xres"]), float(info["yres"])],
        },
    }


def extract_roi_product(
    product_name: str,
    product: dict,
    ring: list[tuple[float, float]],
) -> dict:
    with ExitStack() as stack:
        root = open_kerchunk_root(product, stack)
        if root is not None:
            return extract_roi_product_from_root(product_name, product, root, ring, "kerchunk")
        root = open_hdf5(product, stack)
        return extract_roi_product_from_root(product_name, product, root, ring, "hdf5")


def roi_response(payload: dict) -> dict:
    scene_id = payload.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise SpectrumError(400, "scene_id is required")
    ring = parse_polygon(payload)
    product_names = payload.get("products", ["ortho_radiance", "ortho_sr"])
    if not isinstance(product_names, list) or not product_names:
        raise SpectrumError(400, "products must be a non-empty list")
    unknown = [name for name in product_names if name not in PRODUCTS]
    if unknown:
        raise SpectrumError(400, f"unknown product(s): {', '.join(unknown)}")

    manifest = load_manifest()
    scene = manifest.get("scenes", {}).get(scene_id)
    if scene is None:
        raise SpectrumError(404, f"scene not found: {scene_id}")

    products = {}
    errors = {}
    for product_name in product_names:
        manifest_key = PRODUCTS[product_name]["manifest_key"]
        product = scene.get("products", {}).get(manifest_key)
        if not product:
            products[product_name] = {
                "available": False,
                "reason": f"{manifest_key} is not listed for this scene",
            }
            continue
        try:
            products[product_name] = extract_roi_product(product_name, product, ring)
        except SpectrumError as exc:
            products[product_name] = {"available": False, "reason": exc.message}
            errors[product_name] = exc.message

    if not any(product.get("available") for product in products.values()):
        message = "; ".join(dict.fromkeys(errors.values())) or "no requested products are available"
        status = 413 if any("too large" in error for error in errors.values()) else 503
        raise SpectrumError(status, message)

    centroid_lon, centroid_lat = polygon_centroid(ring)
    return {
        "scene_id": scene_id,
        "collections": scene.get("collections", []),
        "sample_type": "roi",
        "clicked": {"lat": centroid_lat, "lon": centroid_lon},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[lon, lat] for lon, lat in ring]],
        },
        "products": products,
        "indices": calculate_indices(products.get("ortho_sr", {})),
    }


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
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise SpectrumError(400, "request body is missing or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise SpectrumError(400, "request body must be a JSON object")
            self.send_json(200, roi_response(payload))
        except json.JSONDecodeError:
            self.send_json(400, {"error": "request body is not valid JSON"})
        except SpectrumError as exc:
            self.send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - API guardrail
            self.send_json(500, {"error": f"unexpected ROI API error: {exc}"})
