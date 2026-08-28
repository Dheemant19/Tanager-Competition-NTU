"""Build the one-time Tanager-only Sangatta exploratory evidence pack.

This data-preparation pipeline is not part of the numbered notebook sequence.
Its useful GeoJSON outputs, `gmw_clipped.geojson` and `mouth_seed.geojson`,
are already shipped in `data/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Prevent a joblib core-count warning on Windows systems without WMIC.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import xy
import requests
from scipy import ndimage
from shapely.geometry import Point, shape
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SCENE_ID = "20250302_030003_92_4001"
SITE_NAME = "Sangatta, Indonesia"
FEATURE_LABEL = "Sangatta river mouth"
OUTPUT_NAME = "coastal_tanager_evidence"
HERO_SLUG = "sangatta"
ACQUISITION_DATE = "2025-03-02"
CONTEXT_EDGE_LABEL = "mapped water-side mangrove edge"
CONTEXT_MODE = "gmw"
DATA_FIELD_ROOT = "HDFEOS/GRIDS/HYP/Data Fields"
FILL_VALUE = -9999.0
RANDOM_SEED = 17
TRAIN_PIXELS = 20_000
SILHOUETTE_PIXELS = 5_000
PCA_COMPONENTS = 5

# Fixed after visual inspection of the georeferenced Tanager RGB image.
MOUTH_SEED_ROW = 155
MOUTH_SEED_COL = 595
MOUTH_SEED_RADIUS_PIXELS = 6

CORE_THRESHOLD = 0.80
BOUNDARY_LOW = 0.20

GMW_URL = (
    "https://data-gis.unep-wcmc.org/server/rest/services/Hosted/"
    "Global_Mangrove_Watch/FeatureServer/0/query"
)
BIG_URL = (
    "https://geoservices.big.go.id/gis/rest/services/PTRA/"
    "Atlas_250K_KawasanKonservasi/MapServer/7/query"
)

COLORS = {
    "ink": "#112331",
    "muted": "#5b6b75",
    "ocean": "#0b3c5d",
    "blue": "#1f78a8",
    "cyan": "#39a7b8",
    "gold": "#e0a82e",
    "orange": "#d85f2f",
    "green": "#2f7d61",
    "boundary": "#f4c86a",
    "background": "#f6f3eb",
}


@dataclass
class SceneData:
    """Tanager spectra, quality masks, and geospatial metadata."""

    wavelengths: np.ndarray
    spectra: np.ndarray
    uncertainty: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    red_reflectance: np.ndarray
    marine_mask: np.ndarray
    vegetation_mask: np.ndarray
    shore_distance: np.ndarray
    profile: dict
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    pixel_size_m: float
    pixel_area_km2: float
    rgb: np.ndarray
    counts: dict[str, int]


@dataclass(frozen=True)
class RunSpec:
    """One planned processing variation."""

    regime_count: int
    shore_buffer_pixels: int
    random_seed: int
    wavelength_mode: str
    uncertainty_scale: float


@dataclass
class RunResult:
    """Mapped river-connected component and diagnostics for one run."""

    spec: RunSpec
    component: np.ndarray
    area_km2: float
    reach_km: float
    silhouette: float
    spatial_agreement: float
    target_seed_fraction: float
    labels_map: np.ndarray
    regime_medians: np.ndarray
    active_pixel_count: int


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a script or notebook working directory."""

    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "AGENTS.md").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root.")


def nearest_band(
    dataset: h5py.Dataset,
    wavelengths: np.ndarray,
    target_nm: float,
) -> np.ndarray:
    """Read the band nearest a requested wavelength."""

    index = int(np.argmin(np.abs(wavelengths - target_nm)))
    band = dataset[index].astype("float32")
    band[band == FILL_VALUE] = np.nan
    return band


def marine_component(water: np.ndarray) -> np.ndarray:
    """Keep the open-water component connected to the eastern scene edge."""

    labels, count = ndimage.label(water)
    if count == 0:
        raise ValueError("The water mask contains no connected component.")
    edge_labels = np.unique(labels[:, -3:])
    edge_labels = edge_labels[edge_labels > 0]
    if edge_labels.size:
        sizes = ndimage.sum(water, labels, edge_labels)
        selected = int(edge_labels[int(np.argmax(sizes))])
    else:
        sizes = ndimage.sum(water, labels, np.arange(1, count + 1))
        selected = int(np.argmax(sizes) + 1)
    return labels == selected


def load_scene(scene_dir: Path) -> SceneData:
    """Load quality-screened Tanager spectra and uncertainty."""

    sr_path = scene_dir / "ortho_sr_hdf5.h5"
    visual_path = scene_dir / "ortho_visual.tif"
    if not sr_path.exists():
        matches = list(scene_dir.glob("*_ortho_sr_hdf5.h5"))
        if len(matches) == 1:
            sr_path = matches[0]
    if not visual_path.exists():
        matches = list(scene_dir.glob("*_ortho_visual.tif"))
        if len(matches) == 1:
            visual_path = matches[0]
    if not sr_path.exists() or not visual_path.exists():
        raise FileNotFoundError(f"Missing Tanager inputs under {scene_dir}")

    with h5py.File(sr_path, "r") as handle:
        fields = handle[DATA_FIELD_ROOT]
        reflectance = fields["surface_reflectance"]
        reflectance_uncertainty = fields["surface_reflectance_uncertainty"]
        wavelengths = np.asarray(reflectance.attrs["wavelengths"], dtype=float)
        good = np.asarray(reflectance.attrs["good_wavelengths"], dtype=bool)

        green = nearest_band(reflectance, wavelengths, 560)
        red = nearest_band(reflectance, wavelengths, 665)
        nir = nearest_band(reflectance, wavelengths, 865)
        swir1 = nearest_band(reflectance, wavelengths, 1_610)
        swir2 = nearest_band(reflectance, wavelengths, 2_200)

        nodata = fields["nodata_pixels"][:]
        cloud = fields["beta_cloud_mask"][:]
        cirrus = fields["beta_cirrus_mask"][:]
        valid = (
            (nodata == 0)
            & (cloud == 0)
            & (cirrus == 0)
            & np.isfinite(green)
            & np.isfinite(nir)
            & np.isfinite(swir1)
            & np.isfinite(swir2)
        )
        ndwi = (green - nir) / (green + nir + 1e-9)
        ndvi = (nir - red) / (nir + red + 1e-9)
        initial_water = valid & (ndwi > 0.0) & (swir1 < 0.03) & (swir2 < 0.02)
        marine = marine_component(initial_water)
        vegetation = valid & (ndvi > 0.35) & (nir > 0.08)
        shore_distance = ndimage.distance_transform_edt(marine)
        analysis_mask = marine & (shore_distance >= 2.0)
        rows, cols = np.where(analysis_mask)

        selected = np.where(
            good & (wavelengths >= 430.0) & (wavelengths <= 800.0)
        )[0]
        spectra = np.empty((rows.size, selected.size), dtype="float32")
        uncertainty = np.empty_like(spectra)
        for output_index, band_index in enumerate(selected):
            band = reflectance[band_index].astype("float32")
            band_uncertainty = reflectance_uncertainty[band_index].astype("float32")
            spectra[:, output_index] = band[rows, cols]
            uncertainty[:, output_index] = band_uncertainty[rows, cols]

    if not np.isfinite(spectra).all():
        raise ValueError("Non-finite values remain in the Tanager water spectra.")
    uncertainty = np.nan_to_num(
        uncertainty,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    with rasterio.open(visual_path) as visual:
        profile = visual.profile.copy()
        rgb = np.moveaxis(visual.read([1, 2, 3]), 0, -1)
        transform = visual.transform
        crs = visual.crs
        pixel_size_m = float(abs(transform.a))
        pixel_area_km2 = abs(transform.a * transform.e) / 1_000_000

    return SceneData(
        wavelengths=wavelengths[selected],
        spectra=spectra,
        uncertainty=uncertainty,
        rows=rows,
        cols=cols,
        red_reflectance=red[rows, cols],
        marine_mask=marine,
        vegetation_mask=vegetation,
        shore_distance=shore_distance,
        profile=profile,
        transform=transform,
        crs=crs,
        pixel_size_m=pixel_size_m,
        pixel_area_km2=pixel_area_km2,
        rgb=rgb,
        counts={
            "clean_pixels": int(valid.sum()),
            "initial_water_pixels": int(initial_water.sum()),
            "marine_water_pixels": int(marine.sum()),
            "analysis_water_pixels": int(analysis_mask.sum()),
            "spectral_bands": int(selected.size),
        },
    )


def mouth_seed_mask(shape_: tuple[int, int]) -> np.ndarray:
    """Return the fixed circular mouth seed in raster coordinates."""

    row_grid, col_grid = np.ogrid[: shape_[0], : shape_[1]]
    return (
        (row_grid - MOUTH_SEED_ROW) ** 2
        + (col_grid - MOUTH_SEED_COL) ** 2
        <= MOUTH_SEED_RADIUS_PIXELS**2
    )


def stratified_indices(
    values: np.ndarray,
    maximum: int,
    random_seed: int,
) -> np.ndarray:
    """Sample across reflectance deciles so bright coastal water is retained."""

    rng = np.random.default_rng(random_seed)
    edges = np.quantile(values, np.linspace(0, 1, 11))
    selected: list[np.ndarray] = []
    per_bin = max(1, maximum // 10)
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        if index == 9:
            members = np.where((values >= lower) & (values <= upper))[0]
        else:
            members = np.where((values >= lower) & (values < upper))[0]
        take = min(per_bin, members.size)
        if take:
            selected.append(rng.choice(members, take, replace=False))
    return np.sort(np.concatenate(selected))


def wavelength_indices(wavelengths: np.ndarray, mode: str) -> np.ndarray:
    """Select one approved Tanager wavelength subset."""

    if mode == "full":
        return np.arange(wavelengths.size)
    if mode == "trimmed":
        return np.where((wavelengths >= 450.0) & (wavelengths <= 780.0))[0]
    if mode == "thinned":
        return np.arange(0, wavelengths.size, 2)
    raise ValueError(f"Unknown wavelength mode: {mode}")


def order_labels_by_red(
    labels: np.ndarray,
    red_reflectance: np.ndarray,
    regime_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Order regimes from lowest to highest median red reflectance."""

    medians = np.array(
        [np.median(red_reflectance[labels == value]) for value in range(regime_count)]
    )
    order = np.argsort(medians)
    lookup = np.empty(regime_count, dtype=np.uint8)
    lookup[order] = np.arange(regime_count, dtype=np.uint8)
    return lookup[labels], medians[order]


def spatial_neighbor_agreement(labels_map: np.ndarray) -> float:
    """Measure local class coherence among adjacent valid pixels."""

    right_valid = (labels_map[:, :-1] >= 0) & (labels_map[:, 1:] >= 0)
    down_valid = (labels_map[:-1, :] >= 0) & (labels_map[1:, :] >= 0)
    agreements = np.concatenate(
        [
            labels_map[:, :-1][right_valid] == labels_map[:, 1:][right_valid],
            labels_map[:-1, :][down_valid] == labels_map[1:, :][down_valid],
        ]
    )
    return float(np.mean(agreements))


def select_seed_component(
    ordered_labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    active_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Retain the class component connected to the fixed mouth seed."""

    labels_map = np.full(active_mask.shape, -1, dtype=np.int16)
    labels_map[rows, cols] = ordered_labels
    seed_values = labels_map[seed_mask & active_mask]
    seed_values = seed_values[seed_values >= 0]
    if seed_values.size == 0:
        raise ValueError("The fixed mouth seed contains no active water pixels.")
    counts = np.bincount(seed_values)
    target_label = int(np.argmax(counts))
    target_seed_fraction = float(counts[target_label] / seed_values.size)

    target = (labels_map == target_label) & active_mask
    target = ndimage.binary_closing(
        target,
        structure=np.ones((3, 3), dtype=bool),
    )
    components, count = ndimage.label(target)
    component_ids = np.unique(components[seed_mask])
    component_ids = component_ids[component_ids > 0]
    if component_ids.size == 0:
        raise ValueError("No classified component intersects the mouth seed.")
    overlaps = ndimage.sum(seed_mask, components, component_ids)
    selected = int(component_ids[int(np.argmax(overlaps))])
    return components == selected, target_seed_fraction


def run_specification(
    scene: SceneData,
    spec: RunSpec,
) -> RunResult:
    """Fit one Tanager optical-regime model and select its mouth component."""

    active_global = scene.shore_distance[scene.rows, scene.cols] >= float(
        spec.shore_buffer_pixels
    )
    rows = scene.rows[active_global]
    cols = scene.cols[active_global]
    red = scene.red_reflectance[active_global]

    band_indices = wavelength_indices(scene.wavelengths, spec.wavelength_mode)
    features = scene.spectra[active_global][:, band_indices].copy()
    if spec.uncertainty_scale:
        features += (
            spec.uncertainty_scale
            * scene.uncertainty[active_global][:, band_indices]
        )

    training = stratified_indices(
        red,
        maximum=TRAIN_PIXELS,
        random_seed=spec.random_seed,
    )
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(features[training])
    all_scaled = scaler.transform(features)

    pca = PCA(
        n_components=min(PCA_COMPONENTS, features.shape[1]),
        random_state=spec.random_seed,
    )
    training_scores = pca.fit_transform(training_scaled)
    all_scores = pca.transform(all_scaled)

    model = KMeans(
        n_clusters=spec.regime_count,
        n_init=12,
        random_state=spec.random_seed,
        algorithm="lloyd",
    )
    model.fit(training_scores)
    raw_labels = model.predict(all_scores)
    ordered_labels, regime_medians = order_labels_by_red(
        raw_labels,
        red,
        spec.regime_count,
    )

    active_mask = np.zeros(scene.marine_mask.shape, dtype=bool)
    active_mask[rows, cols] = True
    seed = mouth_seed_mask(active_mask.shape)
    component, target_seed_fraction = select_seed_component(
        ordered_labels,
        rows,
        cols,
        active_mask,
        seed,
    )

    labels_map = np.full(active_mask.shape, -1, dtype=np.int16)
    labels_map[rows, cols] = ordered_labels
    rng = np.random.default_rng(spec.random_seed)
    sample = rng.choice(
        training_scores.shape[0],
        size=min(SILHOUETTE_PIXELS, training_scores.shape[0]),
        replace=False,
    )
    silhouette = silhouette_score(
        training_scores[sample],
        model.labels_[sample],
        metric="euclidean",
    )

    component_rows, component_cols = np.where(component)
    reach_km = float(
        np.hypot(
            component_rows - MOUTH_SEED_ROW,
            component_cols - MOUTH_SEED_COL,
        ).max()
        * scene.pixel_size_m
        / 1_000
    )
    return RunResult(
        spec=spec,
        component=component,
        area_km2=float(component.sum() * scene.pixel_area_km2),
        reach_km=reach_km,
        silhouette=float(silhouette),
        spatial_agreement=spatial_neighbor_agreement(labels_map),
        target_seed_fraction=target_seed_fraction,
        labels_map=labels_map,
        regime_medians=regime_medians,
        active_pixel_count=int(active_mask.sum()),
    )


def build_run_specs() -> list[RunSpec]:
    """Create a balanced, deterministic uncertainty ensemble."""

    specs: list[RunSpec] = []
    modes = ("full", "trimmed", "thinned")
    scales = (-0.25, 0.25)
    for regime_count in (3, 4, 5):
        for shore_buffer in (2, 3, 5):
            for replicate in range(2):
                mode_index = (
                    regime_count + shore_buffer + replicate
                ) % len(modes)
                spec = RunSpec(
                    regime_count=regime_count,
                    shore_buffer_pixels=shore_buffer,
                    random_seed=RANDOM_SEED + 11 * replicate + regime_count,
                    wavelength_mode=modes[mode_index],
                    uncertainty_scale=scales[replicate],
                )
                specs.append(spec)

    # Ensure one run is the plain, visually audited reference configuration.
    specs[6] = RunSpec(
        regime_count=4,
        shore_buffer_pixels=2,
        random_seed=RANDOM_SEED,
        wavelength_mode="full",
        uncertainty_scale=0.0,
    )
    return specs


def raster_to_geometry(mask: np.ndarray, scene: SceneData):
    """Convert a binary raster mask to a dissolved geometry."""

    geometries = [
        shape(geometry)
        for geometry, value in shapes(
            mask.astype("uint8"),
            mask=mask,
            transform=scene.transform,
        )
        if value == 1
    ]
    if not geometries:
        return None
    return gpd.GeoSeries(geometries, crs=scene.crs).union_all()


def write_mask_geojson(
    path: Path,
    mask: np.ndarray,
    scene: SceneData,
    properties: dict,
) -> None:
    """Write one dissolved raster-derived feature."""

    geometry = raster_to_geometry(mask, scene)
    if geometry is None:
        collection = {"type": "FeatureCollection", "features": []}
    else:
        frame = gpd.GeoDataFrame(
            [properties],
            geometry=[geometry],
            crs=scene.crs,
        ).to_crs("EPSG:4326")
        collection = json.loads(frame.to_json())
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def save_raster(
    path: Path,
    values: np.ndarray,
    scene: SceneData,
    dtype: str,
    nodata,
) -> None:
    """Save one georeferenced analytical raster."""

    profile = scene.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        photometric="minisblack",
    )
    with rasterio.open(path, "w", **profile) as output:
        output.write(values.astype(dtype), 1)


def query_public_layers(scene: SceneData, data_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    """Download clipped public context layers and save exact responses."""

    scene_bounds = rasterio.warp.transform_bounds(
        scene.crs,
        "EPSG:4326",
        *rasterio.transform.array_bounds(
            scene.marine_mask.shape[0],
            scene.marine_mask.shape[1],
            scene.transform,
        ),
    )
    envelope = ",".join(f"{value:.8f}" for value in scene_bounds)
    common = {
        "geometry": envelope,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "true",
        "outSR": str(scene.crs.to_epsg()),
        "f": "geojson",
    }
    queries = {
        "gmw": (
            GMW_URL,
            common | {"where": "1=1", "outFields": "*"},
        ),
        "tn_kutai": (
            BIG_URL,
            common
            | {
                "where": "Jenis='Taman Nasional'",
                "outFields": "OBJECTID,NAME,Jenis",
            },
        ),
    }

    outputs: dict[str, gpd.GeoDataFrame] = {}
    provenance_path = data_dir / "public_layer_provenance.json"
    existing_provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {}
    )
    provenance = {}
    for name, (url, params) in queries.items():
        raw_path = data_dir / f"{name}_source.geojson"
        if raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            request_url = existing_provenance.get(name, {}).get(
                "request_url",
                "cached response",
            )
        else:
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            payload = response.json()
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
            request_url = response.url
        frame = gpd.GeoDataFrame.from_features(payload.get("features", []))
        frame.set_crs(scene.crs, inplace=True)
        frame = frame[frame.geometry.notna()].copy()
        frame.geometry = frame.geometry.make_valid()
        frame = frame[~frame.geometry.is_empty].copy()
        frame.to_file(data_dir / f"{name}_clipped.geojson", driver="GeoJSON")
        outputs[name] = frame
        provenance[name] = {
            "service": url,
            "request_url": request_url,
            "query_parameters": params,
            "feature_count": int(len(frame)),
            "crs": str(scene.crs),
        }
    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    return outputs


def mangrove_adjacency(
    layers: dict[str, gpd.GeoDataFrame],
    scene: SceneData,
    core: np.ndarray,
    boundary: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Estimate water-side mangrove-edge proximity on the 30 m grid."""

    mangroves = layers.get("gmw")
    if mangroves is None or mangroves.empty:
        return pd.DataFrame(), np.zeros(scene.marine_mask.shape, dtype=bool)
    mangrove_mask = rasterize(
        [(geometry, 1) for geometry in mangroves.geometry],
        out_shape=scene.marine_mask.shape,
        transform=scene.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    water = scene.marine_mask
    edge = mangrove_mask & ndimage.binary_dilation(water, iterations=1)

    rows = []
    for distance_m in (30, 60, 90):
        iterations = max(1, int(round(distance_m / scene.pixel_size_m)))
        core_near = ndimage.binary_dilation(core, iterations=iterations)
        boundary_near = ndimage.binary_dilation(boundary, iterations=iterations)
        rows.append(
            {
                "distance_m": distance_m,
                "mangrove_edge_km_near_core": float(
                    (edge & core_near).sum() * scene.pixel_size_m / 1_000
                ),
                "mangrove_edge_km_near_uncertain_boundary": float(
                    (edge & boundary_near).sum() * scene.pixel_size_m / 1_000
                ),
                "total_mapped_water_side_edge_km": float(
                    edge.sum() * scene.pixel_size_m / 1_000
                ),
            }
        )

    for shift_name, shifted in (
        ("contracted_30m", ndimage.binary_erosion(mangrove_mask, iterations=1)),
        ("nominal", mangrove_mask),
        ("expanded_30m", ndimage.binary_dilation(mangrove_mask, iterations=1)),
    ):
        shifted_edge = shifted & ndimage.binary_dilation(water, iterations=1)
        near = ndimage.binary_dilation(core, iterations=3)
        rows.append(
            {
                "distance_m": 90,
                "boundary_alignment_case": shift_name,
                "mangrove_edge_km_near_core": float(
                    (shifted_edge & near).sum() * scene.pixel_size_m / 1_000
                ),
                "mangrove_edge_km_near_uncertain_boundary": np.nan,
                "total_mapped_water_side_edge_km": float(
                    shifted_edge.sum() * scene.pixel_size_m / 1_000
                ),
            }
        )
    return pd.DataFrame(rows), edge


def local_vegetated_shore_adjacency(
    scene: SceneData,
    core: np.ndarray,
    boundary: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Estimate proximity to a Tanager-derived vegetated shoreline edge."""

    edge = scene.vegetation_mask & ndimage.binary_dilation(
        scene.marine_mask,
        iterations=1,
    )
    rows = []
    for distance_m in (30, 60, 90):
        iterations = max(1, int(round(distance_m / scene.pixel_size_m)))
        core_near = ndimage.binary_dilation(core, iterations=iterations)
        boundary_near = ndimage.binary_dilation(boundary, iterations=iterations)
        rows.append(
            {
                "distance_m": distance_m,
                "boundary_alignment_case": np.nan,
                "shoreline_edge_km_near_core": float(
                    (edge & core_near).sum() * scene.pixel_size_m / 1_000
                ),
                "shoreline_edge_km_near_uncertain_boundary": float(
                    (edge & boundary_near).sum() * scene.pixel_size_m / 1_000
                ),
                "total_mapped_water_side_edge_km": float(
                    edge.sum() * scene.pixel_size_m / 1_000
                ),
            }
        )
    return pd.DataFrame(rows), edge


def scene_view_limits(
    scene: SceneData,
    padding_pixels: int = 20,
    mask: np.ndarray | None = None,
    target_ratio: float | None = None,
) -> tuple:
    """Return a padded, optionally landscape-shaped view around a mask."""

    rows, cols = np.where(scene.marine_mask if mask is None else mask)
    if not rows.size:
        return (0, scene.rgb.shape[1]), (scene.rgb.shape[0], 0)
    left = max(0, int(cols.min()) - padding_pixels)
    right = min(scene.rgb.shape[1], int(cols.max()) + padding_pixels)
    top = max(0, int(rows.min()) - padding_pixels)
    bottom = min(scene.rgb.shape[0], int(rows.max()) + padding_pixels)
    if target_ratio is not None:
        width = max(1, right - left)
        height = max(1, bottom - top)
        if width / height < target_ratio:
            width = int(np.ceil(height * target_ratio))
        else:
            height = int(np.ceil(width / target_ratio))

        def centered_interval(start: int, end: int, size: int, limit: int) -> tuple:
            size = min(size, limit)
            center = (start + end) / 2
            new_start = int(round(center - size / 2))
            new_start = max(0, min(new_start, limit - size))
            return new_start, new_start + size

        left, right = centered_interval(left, right, width, scene.rgb.shape[1])
        top, bottom = centered_interval(top, bottom, height, scene.rgb.shape[0])
    return (left, right), (bottom, top)


def geometry_table(
    selection_frequency: np.ndarray,
    scene: SceneData,
) -> pd.DataFrame:
    """Summarize mapped geometry across transparent stability thresholds."""

    seed_x, seed_y = xy(
        scene.transform,
        MOUTH_SEED_ROW,
        MOUTH_SEED_COL,
        offset="center",
    )
    rows = []
    for threshold in np.arange(0.2, 1.0, 0.1):
        mask = selection_frequency >= threshold
        component_rows, component_cols = np.where(mask)
        if component_rows.size:
            reach = (
                np.hypot(
                    component_rows - MOUTH_SEED_ROW,
                    component_cols - MOUTH_SEED_COL,
                ).max()
                * scene.pixel_size_m
                / 1_000
            )
            centroid_row = float(component_rows.mean())
            centroid_col = float(component_cols.mean())
            centroid_x, centroid_y = xy(
                scene.transform,
                centroid_row,
                centroid_col,
                offset="center",
            )
            displacement = np.hypot(centroid_x - seed_x, centroid_y - seed_y) / 1_000
        else:
            reach = np.nan
            displacement = np.nan
        rows.append(
            {
                "selection_frequency_threshold": round(float(threshold), 1),
                "area_km2": float(mask.sum() * scene.pixel_area_km2),
                "maximum_mouth_distance_km": float(reach),
                "centroid_distance_from_mouth_km": float(displacement),
                "pixel_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def baseline_spectra(
    baseline: RunResult,
    scene: SceneData,
) -> pd.DataFrame:
    """Calculate class-mean spectra and within-class spread."""

    records = []
    labels = baseline.labels_map[scene.rows, scene.cols]
    valid = labels >= 0
    spectra = scene.spectra[valid]
    labels = labels[valid]
    for regime in sorted(np.unique(labels)):
        subset = spectra[labels == regime]
        mean = np.mean(subset, axis=0)
        p10 = np.quantile(subset, 0.10, axis=0)
        p90 = np.quantile(subset, 0.90, axis=0)
        for wavelength, mean_value, low, high in zip(
            scene.wavelengths,
            mean,
            p10,
            p90,
        ):
            records.append(
                {
                    "regime": int(regime) + 1,
                    "wavelength_nm": float(wavelength),
                    "mean_surface_reflectance": float(mean_value),
                    "p10_surface_reflectance": float(low),
                    "p90_surface_reflectance": float(high),
                    "pixel_count": int(subset.shape[0]),
                }
            )
    return pd.DataFrame(records)


def plot_hero(
    path: Path,
    scene: SceneData,
    core: np.ndarray,
    boundary: np.ndarray,
) -> None:
    """Create the opening image: visible plume and stability footprint."""

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.imshow(scene.rgb)
    axis.contour(
        boundary,
        levels=[0.5],
        colors=[COLORS["boundary"]],
        linewidths=2.2,
    )
    axis.contour(
        core,
        levels=[0.5],
        colors=[COLORS["orange"]],
        linewidths=3.2,
    )
    axis.scatter(
        [MOUTH_SEED_COL],
        [MOUTH_SEED_ROW],
        s=95,
        color="white",
        edgecolor=COLORS["ink"],
        linewidth=1.4,
        zorder=5,
    )
    x_limits, y_limits = scene_view_limits(
        scene,
        padding_pixels=70,
        mask=core | boundary,
        target_ratio=1.55,
    )
    label_x = max(x_limits[0], MOUTH_SEED_COL - 0.28 * (x_limits[1] - x_limits[0]))
    label_y = min(y_limits[0], MOUTH_SEED_ROW + 0.30 * (y_limits[0] - y_limits[1]))
    axis.annotate(
        FEATURE_LABEL,
        xy=(MOUTH_SEED_COL, MOUTH_SEED_ROW),
        xytext=(label_x, label_y),
        color="white",
        fontsize=13,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": "white", "lw": 1.5},
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.axis("off")
    axis.set_title(
        "Tanager isolates the seed-connected coastal optical footprint",
        loc="left",
        fontsize=21,
        weight="bold",
        color=COLORS["ink"],
        pad=14,
    )
    axis.text(
        0.02,
        0.025,
        "ORANGE  stable core ≥0.80     GOLD  uncertainty envelope ≥0.20",
        transform=axis.transAxes,
        fontsize=10.5,
        color="white",
        weight="bold",
        bbox={"facecolor": COLORS["ink"], "alpha": 0.78, "edgecolor": "none", "pad": 6},
    )
    figure.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_regimes_and_spectra(
    path: Path,
    baseline: RunResult,
    spectra: pd.DataFrame,
    scene: SceneData,
) -> None:
    """Show spatial regimes and their Tanager spectral signatures."""

    palette = ["#153b5b", "#267b91", "#66b39a", "#e0a82e", "#d85f2f"]
    cmap = ListedColormap(palette[: baseline.spec.regime_count])
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1.05, 1]},
        constrained_layout=True,
    )
    axes[0].imshow(scene.rgb, alpha=0.28)
    regime_map = baseline.labels_map.astype(float)
    regime_map[regime_map < 0] = np.nan
    axes[0].imshow(
        regime_map,
        cmap=cmap,
        vmin=0,
        vmax=baseline.spec.regime_count - 1,
        alpha=0.88,
    )
    axes[0].scatter(
        [MOUTH_SEED_COL],
        [MOUTH_SEED_ROW],
        s=75,
        color="white",
        edgecolor=COLORS["ink"],
        linewidth=1.2,
    )
    x_limits, y_limits = scene_view_limits(scene)
    axes[0].set_xlim(*x_limits)
    axes[0].set_ylim(*y_limits)
    axes[0].axis("off")
    axes[0].set_title(
        f"{baseline.spec.regime_count} constituent-neutral optical regimes",
        loc="left",
        fontsize=17,
        weight="bold",
    )

    for regime, group in spectra.groupby("regime"):
        color = palette[int(regime) - 1]
        axes[1].plot(
            group["wavelength_nm"],
            group["mean_surface_reflectance"],
            color=color,
            linewidth=2.6,
            label=f"Regime {regime}",
        )
        axes[1].fill_between(
            group["wavelength_nm"],
            group["p10_surface_reflectance"],
            group["p90_surface_reflectance"],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    axes[1].set_title(
        "Each mapped regime has a distinct spectral pattern",
        loc="left",
        fontsize=17,
        weight="bold",
    )
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Surface reflectance")
    axes[1].grid(axis="y", color="#d8dde0", linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    figure.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_stability(
    path: Path,
    selection_frequency: np.ndarray,
    core: np.ndarray,
    boundary: np.ndarray,
    scene: SceneData,
) -> None:
    """Show the continuous stability surface and decision thresholds."""

    figure, axis = plt.subplots(figsize=(11.5, 7), constrained_layout=True)
    axis.imshow(scene.rgb, alpha=0.25)
    frequency = selection_frequency.copy()
    frequency[frequency == 0] = np.nan
    image = axis.imshow(
        frequency,
        cmap="YlOrRd",
        vmin=0.2,
        vmax=1.0,
        alpha=0.9,
    )
    axis.contour(
        boundary,
        levels=[0.5],
        colors=[COLORS["gold"]],
        linewidths=1.8,
    )
    axis.contour(
        core,
        levels=[0.5],
        colors=["white"],
        linewidths=2.8,
    )
    axis.scatter(
        [MOUTH_SEED_COL],
        [MOUTH_SEED_ROW],
        s=85,
        color=COLORS["cyan"],
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
    )
    x_limits, y_limits = scene_view_limits(
        scene,
        padding_pixels=70,
        mask=core | boundary,
        target_ratio=1.55,
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.axis("off")
    axis.set_title(
        "The uncertainty ensemble separates a stable core from a shifting edge",
        loc="left",
        fontsize=21,
        weight="bold",
        color=COLORS["ink"],
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Selection frequency", fontsize=11)
    axis.text(
        0.025,
        0.035,
        f"STABLE CORE  {core.sum() * scene.pixel_area_km2:.2f} km²\n"
        f"UNCERTAINTY ENVELOPE  "
        f"{(core | boundary).sum() * scene.pixel_area_km2:.2f} km²",
        transform=axis.transAxes,
        color="white",
        fontsize=11,
        weight="bold",
        linespacing=1.5,
        bbox={"facecolor": COLORS["ink"], "alpha": 0.78, "edgecolor": "none", "pad": 7},
    )
    figure.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_geometry_sensitivity(path: Path, table: pd.DataFrame) -> None:
    """Show how area and reach change as stability requirements tighten."""

    figure, axis_area = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    axis_reach = axis_area.twinx()
    axis_area.plot(
        table["selection_frequency_threshold"],
        table["area_km2"],
        marker="o",
        color=COLORS["orange"],
        linewidth=2.8,
        label="Area",
    )
    axis_reach.plot(
        table["selection_frequency_threshold"],
        table["maximum_mouth_distance_km"],
        marker="s",
        color=COLORS["blue"],
        linewidth=2.4,
        label="Maximum seed distance",
    )
    axis_area.axvline(
        CORE_THRESHOLD,
        color=COLORS["ink"],
        linestyle="--",
        linewidth=1.2,
    )
    axis_area.text(
        CORE_THRESHOLD + 0.01,
        axis_area.get_ylim()[1] * 0.93,
        "stable-core threshold",
        color=COLORS["ink"],
        fontsize=10,
    )
    axis_area.set_xlabel("Minimum selection frequency")
    axis_area.set_ylabel("Mapped area (km²)", color=COLORS["orange"])
    axis_reach.set_ylabel(
        "Maximum distance from seed (km)",
        color=COLORS["blue"],
    )
    axis_area.set_title(
        "Geometry remains transparent as the stability threshold tightens",
        loc="left",
        fontsize=19,
        weight="bold",
        color=COLORS["ink"],
    )
    axis_area.grid(axis="y", color="#d8dde0", linewidth=0.8)
    axis_area.spines["top"].set_visible(False)
    axis_reach.spines["top"].set_visible(False)
    figure.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_context(
    path: Path,
    scene: SceneData,
    core: np.ndarray,
    boundary: np.ndarray,
    mangrove_edge: np.ndarray,
    layers: dict[str, gpd.GeoDataFrame],
) -> None:
    """Map stable optical structure beside screened shoreline context."""

    figure, axis = plt.subplots(figsize=(12, 7.5), constrained_layout=True)
    axis.imshow(scene.rgb)
    axis.contour(
        mangrove_edge,
        levels=[0.5],
        colors=[COLORS["green"]],
        linewidths=2.4,
    )
    axis.contour(
        boundary,
        levels=[0.5],
        colors=[COLORS["boundary"]],
        linewidths=1.8,
    )
    axis.contour(
        core,
        levels=[0.5],
        colors=[COLORS["orange"]],
        linewidths=3.0,
    )
    park = layers.get("tn_kutai")
    if park is not None and not park.empty:
        inverse = ~scene.transform
        for geometry in park.geometry:
            boundary_geometry = geometry.boundary
            for line in getattr(boundary_geometry, "geoms", [boundary_geometry]):
                coordinates = np.asarray(line.coords)
                pixels = np.array([inverse * tuple(point) for point in coordinates])
                axis.plot(
                    pixels[:, 0],
                    pixels[:, 1],
                    color="white",
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.9,
                )
    x_limits, y_limits = scene_view_limits(
        scene,
        padding_pixels=100,
        mask=core | boundary,
        target_ratio=1.55,
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.axis("off")
    axis.set_title(
        "The mapped footprint identifies shoreline segments for follow-up",
        loc="left",
        fontsize=21,
        weight="bold",
        color=COLORS["ink"],
        pad=14,
    )
    axis.text(
        0.02,
        0.025,
        "ORANGE  stable core     GOLD  uncertainty envelope\n"
        f"GREEN  {CONTEXT_EDGE_LABEL}",
        transform=axis.transAxes,
        fontsize=10.5,
        color="white",
        weight="bold",
        linespacing=1.45,
        bbox={"facecolor": COLORS["ink"], "alpha": 0.78, "edgecolor": "none", "pad": 6},
    )
    figure.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def write_story_brief(
    path: Path,
    summary: dict,
    adjacency: pd.DataFrame,
) -> None:
    """Write a compact evidence-led narrative for later deck authoring."""

    adjacency_row = adjacency[
        (adjacency["distance_m"] == 90)
        & adjacency["boundary_alignment_case"].isna()
    ]
    edge_prefix = "mangrove" if CONTEXT_MODE == "gmw" else "shoreline"
    boundary_column = f"{edge_prefix}_edge_km_near_uncertain_boundary"
    core_column = f"{edge_prefix}_edge_km_near_core"
    adjacent_km = (
        float(adjacency_row.iloc[0][boundary_column])
        if not adjacency_row.empty
        else np.nan
    )
    core_km = (
        float(adjacency_row.iloc[0][core_column])
        if not adjacency_row.empty
        else np.nan
    )
    core_text = (
        "the mapped edge does not touch the stable core"
        if np.isfinite(core_km) and core_km == 0
        else f"{core_km:.1f} km also lies near the stable core"
    )
    adjacent_text = (
        f"{adjacent_km:.1f} km of {CONTEXT_EDGE_LABEL} lies within "
        f"90 m of the uncertainty envelope; {core_text}."
        if np.isfinite(adjacent_km)
        else "The shoreline-context statistic remains pending."
    )
    text = f"""# Tanager at {SITE_NAME}

## Communication job

By the end, the audience should understand that Tanager can turn one complex
coastal acquisition into an uncertainty-aware monitoring target because its
dense spectra separate coherent water regimes and its georeferenced footprint
can be tested, measured, and connected to shoreline priorities.

## Six-beat story

1. **A tropical river or estuary meets an ecologically important coast.**
   Open with the Tanager RGB scene and the visible river-connected water mass.
2. **Tanager separates the water into inspectable optical regimes.**
   Show the regime map and class-mean spectra without assigning unsupported
   constituent concentrations.
3. **The seed-connected water regime remains spatially coherent.**
   The reference model classified {summary['analysis_water_pixels']:,} water
   pixels using {summary['spectral_bands']} quality-approved Tanager bands.
4. **Uncertainty becomes part of the product.**
   Across {summary['ensemble_runs']} planned variations, the stable core covers
   {summary['stable_core_area_km2']:.2f} km² and reaches
   {summary['stable_core_maximum_mouth_distance_km']:.2f} km from the fixed
   water seed.
5. **The footprint points to shoreline segments worth checking.**
   {adjacent_text} This is proximity for monitoring, not evidence of damage.
6. **The deliverable is a reusable coastal triage product.**
   End with the stability raster, vector boundaries, geometry table, provenance,
   and a clear request for repeat observation or field follow-up.

## Claim language

Use: **river-connected optical regime**, **stable core**, **uncertain boundary**,
**mapped shoreline proximity**, and **follow-up priority**.

Do not claim constituent concentration, source attribution, legal jurisdiction,
or ecological damage from this single acquisition.

## Primary visual assets

- `figures/01_tanager_{HERO_SLUG}_hero.png`
- `figures/02_optical_regimes_and_spectra.png`
- `figures/03_stability_surface.png`
- `figures/04_geometry_threshold_sensitivity.png`
- `figures/05_shoreline_context.png`
"""
    path.write_text(text, encoding="utf-8")


def write_evidence_readme(
    path: Path,
    summary: dict,
    files: list[Path],
) -> None:
    """Document the evidence pack and its reproducibility."""

    file_lines = "\n".join(f"- `{item.relative_to(path.parent)}`" for item in files)
    context_note = (
        "Public shoreline and management context is queried from the exact "
        "service URLs saved in `data/public_layer_provenance.json`."
        if CONTEXT_MODE == "gmw"
        else "Shoreline context is derived conservatively from vegetation in "
        "the same Tanager scene; no external shoreline layer is used."
    )
    text = f"""# {SITE_NAME} Tanager coastal evidence pack

This folder contains the reviewable evidence for a later 5–6 slide
presentation. It is not the presentation itself.

## Headline result

The {ACQUISITION_DATE} Tanager acquisition contains a coherent, seed-connected
coastal optical regime at {FEATURE_LABEL}. An {summary['ensemble_runs']}-run
uncertainty ensemble separates a {summary['stable_core_area_km2']:.2f} km²
stable core from a wider uncertain boundary. The stable core extends
{summary['stable_core_maximum_mouth_distance_km']:.2f} km from the fixed mouth
seed.

The result is constituent-neutral. It supports coastal triage and follow-up
monitoring; it does not provide a calibrated concentration, a source diagnosis,
or evidence of ecological damage.

## Reproduce

From the repository root:

```powershell
python scripts\\coastal_tanager_story.py --scene-id {SCENE_ID} --output-name {OUTPUT_NAME} --site-name "{SITE_NAME}" --feature-label "{FEATURE_LABEL}" --hero-slug {HERO_SLUG} --acquisition-date {ACQUISITION_DATE} --seed-row {MOUTH_SEED_ROW} --seed-col {MOUTH_SEED_COL} --seed-radius {MOUTH_SEED_RADIUS_PIXELS} --context-mode {CONTEXT_MODE}
```

The script reads the local Tanager scene under
`data/coastal/{SCENE_ID}/` and refreshes this evidence pack.
{context_note}

## Files

{file_lines}
"""
    path.write_text(text, encoding="utf-8")


def run_analysis(
    project_root: Path | None = None,
    *,
    scene_id: str = SCENE_ID,
    output_name: str = OUTPUT_NAME,
    site_name: str = SITE_NAME,
    feature_label: str = FEATURE_LABEL,
    hero_slug: str = HERO_SLUG,
    acquisition_date: str = ACQUISITION_DATE,
    seed_row: int = MOUTH_SEED_ROW,
    seed_col: int = MOUTH_SEED_COL,
    seed_radius: int = MOUTH_SEED_RADIUS_PIXELS,
    context_mode: str = CONTEXT_MODE,
) -> tuple[dict, Path]:
    """Run the complete Tanager-only workflow and export all evidence."""

    global SCENE_ID, OUTPUT_NAME, SITE_NAME, FEATURE_LABEL, HERO_SLUG
    global ACQUISITION_DATE, MOUTH_SEED_ROW, MOUTH_SEED_COL
    global MOUTH_SEED_RADIUS_PIXELS, CONTEXT_EDGE_LABEL, CONTEXT_MODE
    SCENE_ID = scene_id
    OUTPUT_NAME = output_name
    SITE_NAME = site_name
    FEATURE_LABEL = feature_label
    HERO_SLUG = hero_slug
    ACQUISITION_DATE = acquisition_date
    MOUTH_SEED_ROW = seed_row
    MOUTH_SEED_COL = seed_col
    MOUTH_SEED_RADIUS_PIXELS = seed_radius
    CONTEXT_MODE = context_mode
    CONTEXT_EDGE_LABEL = (
        "mapped water-side mangrove edge"
        if context_mode == "gmw"
        else "Tanager-mapped vegetated shoreline edge"
    )
    root = project_root or find_project_root(Path(__file__).resolve())
    scene_dir = root / "data" / "Coastal" / "coastal" / SCENE_ID
    output_dir = root / "presentations" / OUTPUT_NAME
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene(scene_dir)
    seed = mouth_seed_mask(scene.marine_mask.shape)
    if not (seed & scene.marine_mask).any():
        raise ValueError("The configured mouth seed does not intersect marine water.")

    results: list[RunResult] = []
    specs = build_run_specs()
    for run_number, spec in enumerate(specs, start=1):
        print(f"Running Tanager ensemble member {run_number}/{len(specs)}: {spec}")
        results.append(run_specification(scene, spec))

    baseline = results[6]
    selection_frequency = np.mean(
        np.stack([result.component for result in results], axis=0),
        axis=0,
    ).astype("float32")
    core = selection_frequency >= CORE_THRESHOLD
    boundary = (
        (selection_frequency >= BOUNDARY_LOW)
        & (selection_frequency < CORE_THRESHOLD)
    )

    run_table = pd.DataFrame(
        [
            {
                "run": index,
                "regime_count": result.spec.regime_count,
                "shore_buffer_pixels": result.spec.shore_buffer_pixels,
                "random_seed": result.spec.random_seed,
                "wavelength_mode": result.spec.wavelength_mode,
                "uncertainty_scale": result.spec.uncertainty_scale,
                "active_pixel_count": result.active_pixel_count,
                "river_connected_area_km2": result.area_km2,
                "maximum_mouth_distance_km": result.reach_km,
                "silhouette": result.silhouette,
                "spatial_neighbor_agreement": result.spatial_agreement,
                "mouth_seed_dominant_class_fraction": result.target_seed_fraction,
            }
            for index, result in enumerate(results, start=1)
        ]
    )
    run_table.to_csv(data_dir / "ensemble_run_metrics.csv", index=False)

    geometry = geometry_table(selection_frequency, scene)
    geometry.to_csv(data_dir / "geometry_thresholds.csv", index=False)
    spectra = baseline_spectra(baseline, scene)
    spectra.to_csv(data_dir / "optical_regime_spectra.csv", index=False)

    stable_row = geometry.loc[
        np.isclose(
            geometry["selection_frequency_threshold"],
            CORE_THRESHOLD,
        )
    ].iloc[0]
    summary = {
        "scene_id": SCENE_ID,
        "acquisition_date": ACQUISITION_DATE,
        "analysis_water_pixels": scene.counts["analysis_water_pixels"],
        "spectral_bands": scene.counts["spectral_bands"],
        "ensemble_runs": len(results),
        "stable_core_threshold": CORE_THRESHOLD,
        "stable_core_area_km2": float(stable_row["area_km2"]),
        "stable_core_maximum_mouth_distance_km": float(
            stable_row["maximum_mouth_distance_km"]
        ),
        "uncertain_boundary_area_km2": float(
            boundary.sum() * scene.pixel_area_km2
        ),
        "reference_silhouette": baseline.silhouette,
        "reference_spatial_neighbor_agreement": baseline.spatial_agreement,
        "ensemble_area_median_km2": float(run_table["river_connected_area_km2"].median()),
        "ensemble_area_p05_km2": float(
            run_table["river_connected_area_km2"].quantile(0.05)
        ),
        "ensemble_area_p95_km2": float(
            run_table["river_connected_area_km2"].quantile(0.95)
        ),
        "ensemble_reach_median_km": float(
            run_table["maximum_mouth_distance_km"].median()
        ),
        "ensemble_reach_p05_km": float(
            run_table["maximum_mouth_distance_km"].quantile(0.05)
        ),
        "ensemble_reach_p95_km": float(
            run_table["maximum_mouth_distance_km"].quantile(0.95)
        ),
        "mouth_seed": {
            "row": MOUTH_SEED_ROW,
            "column": MOUTH_SEED_COL,
            "radius_pixels": MOUTH_SEED_RADIUS_PIXELS,
            "crs": str(scene.crs),
        },
        "claim_boundary": (
            "Constituent-neutral optical triage and mapped proximity only; "
            "no concentration, source attribution, legal determination, "
            "or ecological-damage inference."
        ),
    }

    save_raster(
        data_dir / "selection_frequency.tif",
        selection_frequency,
        scene,
        "float32",
        -9999.0,
    )
    save_raster(
        data_dir / "reference_optical_regimes.tif",
        np.where(baseline.labels_map >= 0, baseline.labels_map + 1, 0),
        scene,
        "uint8",
        0,
    )
    save_raster(
        data_dir / "stable_core.tif",
        core.astype("uint8"),
        scene,
        "uint8",
        0,
    )
    save_raster(
        data_dir / "uncertain_boundary.tif",
        boundary.astype("uint8"),
        scene,
        "uint8",
        0,
    )

    write_mask_geojson(
        data_dir / "stable_core.geojson",
        core,
        scene,
        {"selection_frequency_minimum": CORE_THRESHOLD},
    )
    write_mask_geojson(
        data_dir / "uncertain_boundary.geojson",
        boundary,
        scene,
        {
            "selection_frequency_minimum": BOUNDARY_LOW,
            "selection_frequency_maximum": CORE_THRESHOLD,
        },
    )

    seed_x, seed_y = xy(
        scene.transform,
        MOUTH_SEED_ROW,
        MOUTH_SEED_COL,
        offset="center",
    )
    seed_geometry = Point(seed_x, seed_y).buffer(
        MOUTH_SEED_RADIUS_PIXELS * scene.pixel_size_m
    )
    seed_frame = gpd.GeoDataFrame(
        [
            {
                "scene_id": SCENE_ID,
                "row": MOUTH_SEED_ROW,
                "column": MOUTH_SEED_COL,
                "radius_pixels": MOUTH_SEED_RADIUS_PIXELS,
                "basis": "Visual audit of the georeferenced Tanager RGB image",
            }
        ],
        geometry=[seed_geometry],
        crs=scene.crs,
    ).to_crs("EPSG:4326")
    seed_frame.to_file(data_dir / "mouth_seed.geojson", driver="GeoJSON")

    if CONTEXT_MODE == "gmw":
        layers = query_public_layers(scene, data_dir)
        adjacency, mangrove_edge = mangrove_adjacency(
            layers,
            scene,
            core,
            boundary,
        )
        adjacency_path = data_dir / "mangrove_edge_adjacency.csv"
    else:
        layers = {}
        adjacency, mangrove_edge = local_vegetated_shore_adjacency(
            scene,
            core,
            boundary,
        )
        adjacency_path = data_dir / "shoreline_edge_adjacency.csv"
    adjacency.to_csv(adjacency_path, index=False)
    summary["gmw_feature_count"] = int(len(layers.get("gmw", [])))
    summary["management_context_feature_count"] = int(
        len(layers.get("tn_kutai", []))
    )

    (data_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_hero(
        figure_dir / f"01_tanager_{HERO_SLUG}_hero.png",
        scene,
        core,
        boundary,
    )
    plot_regimes_and_spectra(
        figure_dir / "02_optical_regimes_and_spectra.png",
        baseline,
        spectra,
        scene,
    )
    plot_stability(
        figure_dir / "03_stability_surface.png",
        selection_frequency,
        core,
        boundary,
        scene,
    )
    plot_geometry_sensitivity(
        figure_dir / "04_geometry_threshold_sensitivity.png",
        geometry,
    )
    plot_context(
        figure_dir / "05_shoreline_context.png",
        scene,
        core,
        boundary,
        mangrove_edge,
        layers,
    )

    write_story_brief(output_dir / "story_brief.md", summary, adjacency)
    files = sorted(
        [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.name not in {"README.md", "artifact_manifest.csv"}
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "type": path.suffix.lower().lstrip("."),
            }
            for path in files
        ]
    )
    manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)
    files.append(output_dir / "artifact_manifest.csv")
    write_evidence_readme(output_dir / "README.md", summary, files)
    return summary, output_dir


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root; auto-detected when omitted.",
    )
    parser.add_argument("--scene-id", default=SCENE_ID)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--site-name", default=SITE_NAME)
    parser.add_argument("--feature-label", default=FEATURE_LABEL)
    parser.add_argument("--hero-slug", default=HERO_SLUG)
    parser.add_argument("--acquisition-date", default=ACQUISITION_DATE)
    parser.add_argument("--seed-row", type=int, default=MOUTH_SEED_ROW)
    parser.add_argument("--seed-col", type=int, default=MOUTH_SEED_COL)
    parser.add_argument(
        "--seed-radius",
        type=int,
        default=MOUTH_SEED_RADIUS_PIXELS,
    )
    parser.add_argument(
        "--context-mode",
        choices=("gmw", "local-vegetation"),
        default=CONTEXT_MODE,
    )
    args = parser.parse_args()
    summary, output_dir = run_analysis(
        args.project_root,
        scene_id=args.scene_id,
        output_name=args.output_name,
        site_name=args.site_name,
        feature_label=args.feature_label,
        hero_slug=args.hero_slug,
        acquisition_date=args.acquisition_date,
        seed_row=args.seed_row,
        seed_col=args.seed_col,
        seed_radius=args.seed_radius,
        context_mode=args.context_mode,
    )
    print(json.dumps(summary, indent=2))
    print(f"Evidence pack written to {output_dir}")


if __name__ == "__main__":
    main()
