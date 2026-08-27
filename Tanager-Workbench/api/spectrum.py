from __future__ import annotations

import gzip
import json
import math
import os
import re
from contextlib import ExitStack
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np
from pyproj import Transformer

try:
    import fsspec
except Exception:  # pragma: no cover - local-file tests do not need fsspec
    fsspec = None

try:
    import zarr
except Exception:  # pragma: no cover - direct HDF5 fallback does not need zarr
    zarr = None

try:
    from aiohttp import ClientTimeout
except Exception:  # pragma: no cover - older fsspec stacks may not use aiohttp
    ClientTimeout = None


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
MANIFEST_PATH = APP_ROOT / "data" / "spectral_manifest.json"
DATA_FIELDS = "HDFEOS/GRIDS/HYP/Data Fields"
STRUCT_METADATA = "HDFEOS INFORMATION/StructMetadata.0"
HTTP_BLOCK_SIZE = 8 * 1024 * 1024
HEADERS = {"User-Agent": "tanager-spectrum-api/0.1"}

PRODUCTS = {
    "ortho_radiance": {
        "manifest_key": "ortho_radiance_hdf5",
        "dataset": f"{DATA_FIELDS}/toa_radiance",
        "label": "TOA radiance",
    },
    "ortho_sr": {
        "manifest_key": "ortho_sr_hdf5",
        "dataset": f"{DATA_FIELDS}/surface_reflectance",
        "label": "Surface reflectance",
    },
}

INDEX_DEFINITIONS = {
    "NDVI": {
        "formula": "(NIR - red) / (NIR + red)",
        "bands": {"red": 665.0, "nir": 865.0},
    },
    "NDWI": {
        "formula": "(green - NIR) / (green + NIR)",
        "bands": {"green": 560.0, "nir": 865.0},
    },
    "MNDWI": {
        "formula": "(green - SWIR1) / (green + SWIR1)",
        "bands": {"green": 560.0, "swir1": 1610.0},
    },
    "Red-edge slope": {
        "formula": "(NIR - red-edge) / (lambda_NIR - lambda_red_edge)",
        "bands": {"red_edge": 705.0, "nir_red_edge": 740.0},
    },
    "SWIR/NIR": {
        "formula": "SWIR1 / NIR",
        "bands": {"swir1": 1610.0, "nir": 865.0},
    },
}


class SpectrumError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SpectrumError(500, "spectral_manifest.json has not been generated")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def parse_number(params: dict, name: str) -> float:
    raw = params.get(name, [None])[0]
    if raw is None:
        raise SpectrumError(400, f"missing required query parameter: {name}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise SpectrumError(400, f"invalid {name}: {raw}") from exc
    if not math.isfinite(value):
        raise SpectrumError(400, f"invalid {name}: {raw}")
    return value


def parse_radius(params: dict) -> int:
    raw = params.get("radius", ["0"])[0]
    try:
        radius = int(raw)
    except ValueError as exc:
        raise SpectrumError(400, "radius must be 0 or 1") from exc
    if radius not in (0, 1):
        raise SpectrumError(400, "radius must be 0 or 1")
    return radius


def parse_products(params: dict) -> list[str]:
    raw = params.get("products", ["ortho_radiance,ortho_sr"])[0]
    names = [name.strip().replace("_hdf5", "") for name in raw.split(",") if name.strip()]
    if not names:
        raise SpectrumError(400, "products must name at least one product")
    unknown = [name for name in names if name not in PRODUCTS]
    if unknown:
        raise SpectrumError(400, f"unknown product(s): {', '.join(unknown)}")
    return names


def bbox_contains(bbox: list[float], lat: float, lon: float) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def clean_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def attr_array(attrs: h5py.AttributeManager, name: str) -> list | None:
    if name not in attrs:
        return None
    value = np.asarray(attrs[name])
    if value.dtype.kind in ("S", "O", "U"):
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in value.tolist()]
    # Kerchunk serialises this Boolean product flag as numeric 0/1.  Normalise
    # it here so every API consumer reliably removes bands flagged as invalid.
    if name == "good_wavelengths" or value.dtype.kind == "b":
        return value.astype(bool).tolist()
    return value.astype(float).tolist()


def attr_text(attrs: h5py.AttributeManager, name: str) -> str | None:
    if name not in attrs:
        return None
    value = clean_scalar(attrs[name])
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def local_path(product: dict) -> Path | None:
    rel = product.get("local_path")
    if not rel:
        return None
    candidates = [PROJECT_ROOT / rel, APP_ROOT / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def local_ref_path(product: dict) -> Path | None:
    rel = product.get("ref_path")
    if not rel:
        return None
    candidates = [APP_ROOT / rel, PROJECT_ROOT / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def open_kerchunk_root(product: dict, stack: ExitStack):
    path = local_ref_path(product)
    reference_url = None
    if path is None:
        relative_path = product.get("ref_path")
        deployment_host = os.environ.get("VERCEL_URL")
        if relative_path and deployment_host:
            reference_url = f"https://{deployment_host}/{relative_path.lstrip('/')}"
        else:
            return None
    if fsspec is None or zarr is None:
        raise SpectrumError(503, "Kerchunk refs require fsspec and zarr")

    if path is not None:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            refs = json.load(handle)
    else:
        with stack.enter_context(fsspec.open(reference_url, "rb", headers=HEADERS)) as remote:
            with gzip.GzipFile(fileobj=remote, mode="rb") as compressed:
                refs = json.loads(compressed.read().decode("utf-8"))
    zarr_major = int(str(getattr(zarr, "__version__", "2")).split(".", maxsplit=1)[0])
    if zarr_major >= 3:
        fs = fsspec.filesystem(
            "reference",
            fo=refs,
            remote_protocol="https",
            remote_options={"headers": HEADERS, "asynchronous": True},
            asynchronous=True,
        )
    else:
        fs = fsspec.filesystem(
            "reference",
            fo=refs,
            remote_protocol="https",
            remote_options={"headers": HEADERS},
        )
    close = getattr(fs, "close", None)
    if close is not None:
        stack.callback(close)
    mapper = fs.get_mapper("")
    return zarr.open_group(mapper, mode="r")


def open_hdf5(product: dict, stack: ExitStack) -> h5py.File:
    path = local_path(product)
    if path is not None:
        return stack.enter_context(h5py.File(path, "r"))

    url = product.get("url")
    if not url:
        raise SpectrumError(404, "product has no HDF5 URL")
    if fsspec is None:
        raise SpectrumError(503, "remote HDF5 access requires fsspec")

    timeout = ClientTimeout(total=90) if ClientTimeout is not None else 90
    fs = fsspec.filesystem(
        "https",
        block_size=HTTP_BLOCK_SIZE,
        headers=HEADERS,
        client_kwargs={"timeout": timeout},
    )
    remote = stack.enter_context(fs.open(url, "rb", block_size=HTTP_BLOCK_SIZE))
    return stack.enter_context(h5py.File(remote, "r"))


def get_node(root, path: str):
    try:
        return root[path]
    except (KeyError, ValueError):
        return None


def struct_metadata_text(root) -> str:
    node = get_node(root, STRUCT_METADATA)
    if node is None:
        raise SpectrumError(422, "HDF-EOS StructMetadata.0 is missing")
    raw = node[()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, np.bytes_):
        return bytes(raw).decode("utf-8", errors="replace")
    return str(raw)


def match_float_pair(text: str, key: str) -> tuple[float, float]:
    match = re.search(rf"{key}=\(([-+0-9.]+),([-+0-9.]+)\)", text)
    if not match:
        raise SpectrumError(422, f"HDF-EOS {key} is missing")
    return float(match.group(1)), float(match.group(2))


def match_int(text: str, key: str) -> int:
    match = re.search(rf"{key}=([-+0-9]+)", text)
    if not match:
        raise SpectrumError(422, f"HDF-EOS {key} is missing")
    return int(match.group(1))


def grid_info(root) -> dict:
    hyp = get_node(root, "HDFEOS/GRIDS/HYP")
    epsg = clean_scalar(hyp.attrs.get("epsg_code")) if hyp is not None else None
    text = struct_metadata_text(root)
    ulx, uly = match_float_pair(text, "UpperLeftPointMtrs")
    lrx, lry = match_float_pair(text, "LowerRightMtrs")
    xdim = match_int(text, "XDim")
    ydim = match_int(text, "YDim")
    if epsg is None:
        zone = match_int(text, "ZoneCode")
        epsg = 32600 + zone
    return {
        "epsg": int(epsg),
        "ulx": ulx,
        "uly": uly,
        "lrx": lrx,
        "lry": lry,
        "xdim": xdim,
        "ydim": ydim,
        "xres": (lrx - ulx) / xdim,
        "yres": (uly - lry) / ydim,
    }


@lru_cache(maxsize=64)
def transformer_for(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def lonlat_to_pixel(info: dict, lon: float, lat: float) -> tuple[int, int, float, float]:
    x, y = transformer_for(info["epsg"]).transform(lon, lat)
    col = math.floor((x - info["ulx"]) / info["xres"])
    row = math.floor((info["uly"] - y) / info["yres"])
    if row < 0 or col < 0 or row >= info["ydim"] or col >= info["xdim"]:
        raise SpectrumError(422, "clicked point is outside the HDF5 grid")
    return int(row), int(col), float(x), float(y)


def read_qa(root, row: int, col: int) -> dict:
    qa = {}
    for name in ("beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels"):
        path = f"{DATA_FIELDS}/{name}"
        node = get_node(root, path)
        if node is not None:
            value = clean_scalar(node[row, col])
            qa[name] = int(value)
    if qa:
        qa["is_cloud"] = bool(qa.get("beta_cloud_mask", 0))
        qa["is_cirrus"] = bool(qa.get("beta_cirrus_mask", 0))
        qa["is_nodata"] = bool(qa.get("nodata_pixels", 0))
    return qa


def values_to_json(values: np.ndarray, fill_value: float | None) -> list:
    arr = np.asarray(values, dtype=float)
    if fill_value is not None:
        arr[np.isclose(arr, fill_value)] = np.nan
    return [None if not np.isfinite(v) else float(v) for v in arr]


def nearest_band(wavelengths: list | None, target: float) -> dict | None:
    if not wavelengths:
        return None
    arr = np.asarray(wavelengths, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    index = int(np.nanargmin(np.abs(arr - target)))
    return {
        "index": index,
        "target_nm": target,
        "wavelength_nm": float(arr[index]),
        "delta_nm": float(abs(arr[index] - target)),
    }


def value_at(values: list, band: dict | None) -> float | None:
    if band is None:
        return None
    index = band["index"]
    if index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def reflectance_value_at(values: list, good_wavelengths: list | None, band: dict | None) -> float | None:
    value = value_at(values, band)
    if value is None or band is None:
        return None
    index = band["index"]
    if good_wavelengths is not None and index < len(good_wavelengths):
        try:
            if not bool(good_wavelengths[index]):
                return None
        except TypeError:
            return None
    if value < 0:
        return None
    return value


def normalized_difference(a: float | None, b: float | None) -> tuple[float | None, str | None]:
    if a is None or b is None:
        return None, "one or more required bands are masked or unavailable"
    denom = a + b
    if abs(denom) < 1e-12:
        return None, "division by zero"
    return (a - b) / denom, None


def ratio(a: float | None, b: float | None) -> tuple[float | None, str | None]:
    if a is None or b is None:
        return None, "one or more required bands are masked or unavailable"
    if abs(b) < 1e-12:
        return None, "division by zero"
    return a / b, None


def calculate_indices(product: dict) -> dict:
    if not product.get("available"):
        return {}

    wavelengths = product.get("wavelengths") or []
    values = product.get("values") or []
    bands = {
        name: nearest_band(wavelengths, target)
        for definition in INDEX_DEFINITIONS.values()
        for name, target in definition["bands"].items()
    }
    good_wavelengths = product.get("good_wavelengths")
    band_values = {
        name: reflectance_value_at(values, good_wavelengths, band)
        for name, band in bands.items()
    }

    computed = {}
    ndvi, ndvi_reason = normalized_difference(band_values.get("nir"), band_values.get("red"))
    ndwi, ndwi_reason = normalized_difference(band_values.get("green"), band_values.get("nir"))
    mndwi, mndwi_reason = normalized_difference(band_values.get("green"), band_values.get("swir1"))
    swir_nir, swir_nir_reason = ratio(band_values.get("swir1"), band_values.get("nir"))

    red_edge = band_values.get("red_edge")
    nir740 = band_values.get("nir_red_edge")
    red_edge_slope = None
    red_edge_reason = None
    if red_edge is None or nir740 is None or bands.get("red_edge") is None or bands.get("nir_red_edge") is None:
        red_edge_reason = "one or more required bands are masked or unavailable"
    else:
        delta = bands["nir_red_edge"]["wavelength_nm"] - bands["red_edge"]["wavelength_nm"]
        if abs(delta) < 1e-12:
            red_edge_reason = "matched wavelengths are identical"
        else:
            red_edge_slope = (nir740 - red_edge) / delta

    raw_values = {
        "NDVI": (ndvi, ndvi_reason),
        "NDWI": (ndwi, ndwi_reason),
        "MNDWI": (mndwi, mndwi_reason),
        "Red-edge slope": (red_edge_slope, red_edge_reason),
        "SWIR/NIR": (swir_nir, swir_nir_reason),
    }

    for name, definition in INDEX_DEFINITIONS.items():
        used = {
            band_name: bands.get(band_name)
            for band_name in definition["bands"]
        }
        value, reason = raw_values[name]
        computed[name] = {
            "value": None if value is None else float(value),
            "formula": definition["formula"],
            "bands": list(definition["bands"]),
            "wavelength_matches": used,
            "invalid": value is None,
            "reason": reason,
        }
    return computed


def extract_values(dataset, row: int, col: int, radius: int, fill_value) -> list:
    if radius == 0:
        return values_to_json(dataset[:, row, col], fill_value)

    r0 = max(0, row - radius)
    r1 = min(dataset.shape[1], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(dataset.shape[2], col + radius + 1)
    cube = np.asarray(dataset[:, r0:r1, c0:c1], dtype=float).reshape(dataset.shape[0], -1)
    if fill_value is not None:
        cube[np.isclose(cube, fill_value)] = np.nan
    return values_to_json(np.nanmedian(cube, axis=1), None)


def extract_product_from_root(
    product_name: str,
    product: dict,
    root,
    lat: float,
    lon: float,
    radius: int,
    source_kind: str,
) -> dict:
    spec = PRODUCTS[product_name]
    path = spec["dataset"]
    dataset = get_node(root, path)
    if dataset is None:
        raise SpectrumError(422, f"{path} is missing")
    info = grid_info(root)
    row, col, x, y = lonlat_to_pixel(info, lon, lat)
    fill_value = clean_scalar(dataset.attrs.get("_FillValue"))
    attrs = dataset.attrs
    good_wavelengths = attr_array(attrs, "good_wavelengths")
    values = extract_values(dataset, row, col, radius, fill_value)
    if product_name == "ortho_sr" and good_wavelengths is not None:
        values = [
            value if index < len(good_wavelengths) and good_wavelengths[index] else None
            for index, value in enumerate(values)
        ]
    return {
        "available": True,
        "label": spec["label"],
        "dataset": path,
        "source": source_kind,
        "source_url": product.get("url"),
        "ref_path": product.get("ref_path") if source_kind == "kerchunk" else None,
        "row": row,
        "col": col,
        "projected_x": x,
        "projected_y": y,
        "grid": {
            "epsg": info["epsg"],
            "shape": [int(dataset.shape[1]), int(dataset.shape[2])],
            "pixel_size_m": [info["xres"], info["yres"]],
        },
        "qa": read_qa(root, row, col),
        "units": attr_text(attrs, "Unit"),
        "wavelength_units": attr_text(attrs, "wavelengths_units") or "nm",
        "wavelengths": attr_array(attrs, "wavelengths"),
        "fwhm": attr_array(attrs, "fwhm"),
        "good_wavelengths": good_wavelengths,
        "values": values,
    }


def extract_product(product_name: str, product: dict, lat: float, lon: float, radius: int) -> dict:
    with ExitStack() as stack:
        root = open_kerchunk_root(product, stack)
        if root is not None:
            return extract_product_from_root(product_name, product, root, lat, lon, radius, "kerchunk")
        h5 = open_hdf5(product, stack)
        return extract_product_from_root(product_name, product, h5, lat, lon, radius, "hdf5")


def spectrum_response(params: dict) -> dict:
    scene_id = params.get("scene_id", [None])[0]
    if not scene_id:
        raise SpectrumError(400, "missing required query parameter: scene_id")
    lat = parse_number(params, "lat")
    lon = parse_number(params, "lon")
    radius = parse_radius(params)
    product_names = parse_products(params)

    manifest = load_manifest()
    scene = manifest.get("scenes", {}).get(scene_id)
    if scene is None:
        raise SpectrumError(404, f"scene not found: {scene_id}")
    bbox = scene.get("bbox")
    if bbox and not bbox_contains(bbox, lat, lon):
        raise SpectrumError(422, "clicked point is outside the scene bbox")

    products = {}
    row = col = None
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
            extracted = extract_product(product_name, product, lat, lon, radius)
            products[product_name] = extracted
            row = extracted["row"] if row is None else row
            col = extracted["col"] if col is None else col
        except SpectrumError as exc:
            products[product_name] = {"available": False, "reason": exc.message}
            errors[product_name] = exc.message

    if not any(product.get("available") for product in products.values()):
        message = "; ".join(errors.values()) or "no requested products are available"
        raise SpectrumError(503, message)

    return {
        "scene_id": scene_id,
        "collections": scene.get("collections", []),
        "clicked": {"lat": lat, "lon": lon},
        "pixel": {"row": row, "col": col, "radius": radius},
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
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            params = parse_qs(urlparse(self.path).query)
            self.send_json(200, spectrum_response(params))
        except SpectrumError as exc:
            self.send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - API guardrail
            self.send_json(500, {"error": f"unexpected spectrum API error: {exc}"})
