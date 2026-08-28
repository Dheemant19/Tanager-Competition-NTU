"""
Audit the 385 Southeast Asian Tanager methane plumes in Carbon Mapper.

The existing SEA census saved 385 plume *observations* (not 385 unique sites)
to ``data/sea_satellite_analysis/raw_extracts/tanager_plumes_sea.csv``.  This
script keeps that cohort fixed, enriches every plume from Carbon Mapper's live
public API, and asks two kinds of questions:

1. What is complete or missing in the published detections-  Examples include
   emission rates, uncertainties, wind, quality flags, sector attribution,
   acquisition geometry, imagery, and source grouping.
2. What can the detection-only public record not answer-  Examples include
   plume-free observations, failed tasking attempts, cloud-screened usable
   coverage, scene-specific detection limits, and reasons a rate is hidden.

The distinction is important: a missing value in a published plume record is
measurable, while an observation-design variable absent from the endpoint is a
catalogue limitation.  The report never treats either one as proof that the
satellite failed to observe or quantify a particular source.

Run from the project root in PowerShell:

    python scripts\analyze_carbon_mapper_tanager_sea.py

Use ``--refresh`` to replace today's cached API snapshot with a fresh response:

    python scripts\analyze_carbon_mapper_tanager_sea.py --refresh

Outputs are written under ``data/sea_satellite_analysis`` and the narrative
report is written to ``data/sea_satellite_analysis/analysis/carbon_mapper_tanager_audit/gap_audit.md``.
No plume rasters or other large imagery assets are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Project paths and analysis settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEA_DIR = PROJECT_ROOT / "data" / "sea_satellite_analysis"
RAW_DIR = SEA_DIR / "raw_extracts"
ANALYSIS_DIR = SEA_DIR / "analysis" / "carbon_mapper_tanager_audit"
FIGURE_DIR = PROJECT_ROOT / "figures"
REFERENCE_DIR = RAW_DIR / "reference"
REPORT_PATH = ANALYSIS_DIR / "gap_audit.md"

COHORT_PATH = RAW_DIR / "tanager_plumes_sea.csv"
PLUME_API_URL = "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated"
SOURCE_API_URL = "https://api.carbonmapper.org/api/v1/catalog/sources.geojson"
API_DOCS_URL = "https://api.carbonmapper.org/api/v1/docs"
PRODUCT_GUIDE_URL = "https://carbonmapper.org/articles/product-guide"

# This is the same rectangular frame used by the three existing SEA notebooks.
# A rectangle includes pieces of non-ASEAN countries and ocean, so the script
# derives country from the actual point coordinates instead of calling every
# point inside the rectangle "ASEAN".
SEA_BBOX = (92.0, -11.0, 141.0, 23.5)

# Carbon Mapper's recent geospatial tutorial uses 500 m for source clustering.
# We also calculate 1 km and 5 km counts to show how sensitive "number of sites"
# is to this user-selected definition.
SOURCE_CLUSTER_DISTANCES_M = (500, 1_000, 5_000)
PRIMARY_SOURCE_DISTANCE_M = 500

BATCH_SIZE = 50
PAUSE_SECONDS = 0.20
MAX_RETRIES = 4

NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_0_countries.zip"
)
NATURAL_EARTH_ZIP = REFERENCE_DIR / "ne_10m_admin_0_countries.zip"
NATURAL_EARTH_DIR = REFERENCE_DIR / "ne_10m_admin_0_countries"

ASEAN_COUNTRIES = [
    "Brunei",
    "Cambodia",
    "Indonesia",
    "Laos",
    "Malaysia",
    "Myanmar",
    "Philippines",
    "Singapore",
    "Thailand",
    "Timor-Leste",
    "Vietnam",
]

COUNTRY_NAME_REPLACEMENTS = {
    "Brunei Darussalam": "Brunei",
    "Lao PDR": "Laos",
    "Lao People's Democratic Republic": "Laos",
    "East Timor": "Timor-Leste",
    "Viet Nam": "Vietnam",
}

SECTOR_NAMES = {
    "1B2": "Oil and gas",
    "1B1": "Coal mining",
    "1B1a": "Coal mining",
    "6A": "Solid waste / landfill",
    "6B": "Wastewater",
    "4B": "Agriculture / manure",
    "1A1": "Energy industries",
    "other": "Other / unclassified",
    "NA": "Unclassified",
}

HEADERS = {
    "User-Agent": (
        "Tanager-SEA-gap-audit/1.0 "
        "(research project; contact akshatcrypts2022@gmail.com)"
    )
}


# ---------------------------------------------------------------------------
# Small, reusable helpers
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Read the optional command-line flags."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download today's Carbon Mapper API snapshots instead of using cache.",
    )
    return parser.parse_args()


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    """Format a UTC timestamp without a platform-dependent representation."""

    return value.isoformat().replace("+00:00", "Z")


def make_session() -> requests.Session:
    """Create one polite, reusable web session for the small API requests."""

    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Download and parse one JSON response, with gentle retry delays."""

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=120)
            response.raise_for_status()
            time.sleep(PAUSE_SECONDS)
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            wait_seconds = attempt * 2
            print(
                f"  Request attempt {attempt}/{MAX_RETRIES} failed: {error}. "
                f"Retrying in {wait_seconds} seconds."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Could not retrieve {url}") from last_error


def write_json(path: Path, payload: Any) -> None:
    """Save readable UTF-8 JSON, creating its parent folder if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def chunks(values: list[str], size: int) -> list[list[str]]:
    """Split a list into stable, easy-to-review request batches."""

    return [values[start : start + size] for start in range(0, len(values), size)]


def is_blank(value: Any) -> bool:
    """Treat None, NaN, and empty text as missing."""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def percent(numerator: int | float, denominator: int | float) -> float:
    """Calculate a percentage safely."""

    if denominator == 0:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def safe_median(values: pd.Series) -> float:
    """Return a numeric median or NaN when no numeric values exist."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.median())


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Make a compact Markdown table without requiring the tabulate package."""

    display = frame.copy()
    if columns is not None:
        display = display[columns]
    display = display.fillna("")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        clean = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(clean) + " |")
    return "\n".join(lines)


def list_text(values: list[str]) -> str:
    """Render a readable English list for the report."""

    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def clean_source_name(value: Any) -> str | None:
    """Remove an API query-string artifact from an otherwise valid source name."""

    if is_blank(value):
        return None
    return str(value).split("-", 1)[0]


# ---------------------------------------------------------------------------
# Carbon Mapper API retrieval
# ---------------------------------------------------------------------------

def validate_cohort(cohort: pd.DataFrame) -> None:
    """Stop early if the input is not the intended fixed 385-plume cohort."""

    required = {
        "plume_id",
        "gas",
        "datetime",
        "lat",
        "lon",
        "emission_kg_hr",
        "uncertainty_kg_hr",
        "hidden_emission",
    }
    missing_columns = sorted(required.difference(cohort.columns))
    if missing_columns:
        raise ValueError(f"Cohort CSV is missing required columns: {missing_columns}")
    if cohort["plume_id"].duplicated().any():
        duplicates = cohort.loc[cohort["plume_id"].duplicated(), "plume_id"].tolist()
        raise ValueError(f"Cohort contains duplicate plume IDs: {duplicates[:5]}")
    if len(cohort) != 385:
        raise ValueError(
            f"Expected the fixed 385-plume cohort, but {COHORT_PATH} has {len(cohort)} rows. "
            "Review the changed input before rerunning this audit."
        )


def fetch_plume_snapshot(
    session: requests.Session,
    cohort_ids: list[str],
    snapshot_path: Path,
    refresh: bool,
    fetched_at: datetime,
) -> dict[str, Any]:
    """Fetch the exact cohort by plume ID, or reuse today's cached snapshot."""

    if snapshot_path.exists() and not refresh:
        print(f"Using cached plume snapshot: {snapshot_path.name}")
        return read_json(snapshot_path)

    print(f"Fetching {len(cohort_ids)} exact plume records in polite batches...")
    items: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(chunks(cohort_ids, BATCH_SIZE), start=1):
        params: list[tuple[str, Any]] = [("plume_names", plume_id) for plume_id in batch]
        params.extend(
            [
                ("status", "all"),
                ("sort", "desc"),
                ("limit", len(batch)),
                ("offset", 0),
            ]
        )
        payload = get_json(session, PLUME_API_URL, params=params)
        items.extend(payload.get("items", []))
        print(
            f"  Batch {batch_number}: requested {len(batch)}, "
            f"received {len(payload.get('items', []))}"
        )

    snapshot = {
        "metadata": {
            "fetched_at_utc": iso_utc(fetched_at),
            "endpoint": PLUME_API_URL,
            "status_filter": "all",
            "cohort_source": str(COHORT_PATH.relative_to(PROJECT_ROOT)),
            "requested_count": len(cohort_ids),
            "returned_count": len(items),
        },
        "items": items,
    }
    write_json(snapshot_path, snapshot)
    return snapshot


def source_params(eps_metres: int) -> list[tuple[str, Any]]:
    """Build the repeated query parameters used by Carbon Mapper's source API."""

    west, south, east, north = SEA_BBOX
    return [
        ("bbox", west),
        ("bbox", south),
        ("bbox", east),
        ("bbox", north),
        ("plume_gas", "CH4"),
        ("instrument", "tan"),
        ("status", "published"),
        ("minpoints", 1),
        ("eps", eps_metres),
        ("cloud_cover_pct_max", 25),
    ]


def fetch_source_snapshots(
    session: requests.Session,
    snapshot_path: Path,
    refresh: bool,
    fetched_at: datetime,
) -> dict[str, Any]:
    """Fetch source clusters at three distances for sensitivity analysis."""

    if snapshot_path.exists() and not refresh:
        print(f"Using cached source snapshot: {snapshot_path.name}")
        return read_json(snapshot_path)

    collections: dict[str, Any] = {}
    for eps_metres in SOURCE_CLUSTER_DISTANCES_M:
        print(f"Fetching Carbon Mapper sources with eps={eps_metres} m...")
        collections[str(eps_metres)] = get_json(
            session,
            SOURCE_API_URL,
            params=source_params(eps_metres),
        )

    snapshot = {
        "metadata": {
            "fetched_at_utc": iso_utc(fetched_at),
            "endpoint": SOURCE_API_URL,
            "bbox": list(SEA_BBOX),
            "gas": "CH4",
            "instrument": "tan",
            "status": "published",
            "minpoints": 1,
            "cloud_cover_pct_max": 25,
        },
        "collections_by_eps_metres": collections,
    }
    write_json(snapshot_path, snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# Geographic context
# ---------------------------------------------------------------------------

def ensure_country_boundaries(session: requests.Session) -> Path:
    """Download and unpack the small public-domain Natural Earth country layer."""

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    NATURAL_EARTH_DIR.mkdir(parents=True, exist_ok=True)
    shapefiles = sorted(NATURAL_EARTH_DIR.glob("*.shp"))
    if shapefiles:
        return shapefiles[0]

    if not NATURAL_EARTH_ZIP.exists():
        print("Downloading Natural Earth country boundaries (small reference file)...")
        response = session.get(NATURAL_EARTH_URL, timeout=120)
        response.raise_for_status()
        NATURAL_EARTH_ZIP.write_bytes(response.content)
        time.sleep(PAUSE_SECONDS)

    with zipfile.ZipFile(NATURAL_EARTH_ZIP) as archive:
        archive.extractall(NATURAL_EARTH_DIR)

    shapefiles = sorted(NATURAL_EARTH_DIR.glob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError("Natural Earth archive did not contain a shapefile.")
    return shapefiles[0]


def normalize_country_name(value: Any) -> str:
    """Normalize the few Natural Earth country names used in the ASEAN list."""

    if is_blank(value):
        return "Offshore / boundary-unassigned"
    name = str(value)
    return COUNTRY_NAME_REPLACEMENTS.get(name, name)


def assign_countries(
    frame: pd.DataFrame,
    countries: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Attach land-country names to point coordinates; preserve offshore gaps."""

    output = frame.copy()
    points = gpd.GeoDataFrame(
        output[["plume_id"]].copy(),
        geometry=gpd.points_from_xy(output["longitude"], output["latitude"]),
        crs="EPSG:4326",
    )

    country_name_field = "NAME_EN" if "NAME_EN" in countries.columns else "ADMIN"
    iso_field = "ISO_A3" if "ISO_A3" in countries.columns else "ADM0_A3"
    country_fields = [country_name_field, iso_field, "geometry"]
    joined = gpd.sjoin(
        points,
        countries[country_fields],
        how="left",
        predicate="within",
    )
    # Natural Earth can contain overlapping disputed representations.  Keeping
    # one match prevents a single plume from silently becoming multiple rows.
    joined = joined.loc[~joined.index.duplicated(keep="first")]

    output["country"] = joined[country_name_field].reindex(output.index)
    output["country_iso3"] = joined[iso_field].reindex(output.index)
    output["country"] = output["country"].map(normalize_country_name)
    output["is_asean_land_point"] = output["country"].isin(ASEAN_COUNTRIES)
    return output


# ---------------------------------------------------------------------------
# Flatten and harmonize API records
# ---------------------------------------------------------------------------

def flatten_plumes(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn nested API plume records into one analysis row per plume."""

    rows: list[dict[str, Any]] = []
    asset_fields = [
        "plume_png",
        "plume_rgb_png",
        "plume_tif",
        "con_tif",
        "rgb_png",
    ]
    simple_fields = [
        "id",
        "plume_id",
        "gas",
        "scene_id",
        "scene_timestamp",
        "instrument",
        "mission_phase",
        "platform",
        "emission_auto",
        "emission_uncertainty_auto",
        "emission_cmf_type",
        "gsd",
        "sensitivity_mode",
        "off_nadir",
        "plume_quality",
        "wind_speed_avg_auto",
        "wind_direction_avg_auto",
        "emission_version",
        "processing_software",
        "is_offshore",
        "collection",
        "cmf_type",
        "sector",
        "status",
        "hide_emission",
        "published_at",
        "modified",
    ]

    for item in items:
        row = {field: item.get(field) for field in simple_fields}
        geometry = item.get("geometry_json") or {}
        coordinates = geometry.get("coordinates") or [None, None]
        row["longitude"] = coordinates[0] if len(coordinates) > 0 else None
        row["latitude"] = coordinates[1] if len(coordinates) > 1 else None

        bounds = item.get("plume_bounds") or [None, None, None, None]
        padded_bounds = list(bounds) + [None] * (4 - len(bounds))
        row["plume_west"] = padded_bounds[0]
        row["plume_south"] = padded_bounds[1]
        row["plume_east"] = padded_bounds[2]
        row["plume_north"] = padded_bounds[3]

        publications = item.get("publication_sources") or []
        row["publication_sources_count"] = len(publications)
        row["publication_sources_json"] = json.dumps(publications, ensure_ascii=False)

        for asset in asset_fields:
            row[f"has_{asset}"] = bool(item.get(asset))
            # Signed query strings expire.  The query-free location is useful
            # for provenance without implying that a saved URL is permanent.
            asset_url = item.get(asset)
            row[f"{asset}_base_url"] = asset_url.split("-", 1)[0] if asset_url else None
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    for field in ["scene_timestamp", "published_at", "modified"]:
        frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
    for field in [
        "emission_auto",
        "emission_uncertainty_auto",
        "gsd",
        "off_nadir",
        "wind_speed_avg_auto",
        "wind_direction_avg_auto",
        "longitude",
        "latitude",
    ]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    frame["publication_lag_days"] = (
        frame["published_at"] - frame["scene_timestamp"]
    ).dt.total_seconds() / 86_400.0
    frame["sector_label"] = (
        frame["sector"].fillna("NA").map(lambda code: SECTOR_NAMES.get(code, str(code)))
    )
    frame["sector_is_specific"] = ~frame["sector"].fillna("NA").isin(
        ["NA", "other", "NULL", ""]
    )

    rate_present = frame["emission_auto"].notna()
    uncertainty_present = frame["emission_uncertainty_auto"].notna()
    hidden = frame["hide_emission"].fillna(False).astype(bool)
    frame["quantification_status"] = np.select(
        [
            rate_present & uncertainty_present & ~hidden,
            hidden & ~rate_present,
            rate_present & ~uncertainty_present,
        ],
        [
            "Public rate + uncertainty",
            "Hidden / unavailable rate",
            "Rate present, uncertainty missing",
        ],
        default="Other incomplete combination",
    )
    frame["relative_uncertainty_percent"] = np.where(
        frame["emission_auto"] > 0,
        100.0 * frame["emission_uncertainty_auto"] / frame["emission_auto"],
        np.nan,
    )
    return frame


def compare_with_original(
    plumes: pd.DataFrame,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Record whether live API values have changed since the saved census."""

    original = cohort.copy()
    original["original_emission_kg_hr"] = pd.to_numeric(
        original["emission_kg_hr"], errors="coerce"
    )
    original["original_uncertainty_kg_hr"] = pd.to_numeric(
        original["uncertainty_kg_hr"], errors="coerce"
    )
    original["original_hidden_emission"] = (
        original["hidden_emission"].astype(str).str.lower().map({"true": True, "false": False})
    )
    original["original_scene_timestamp"] = pd.to_datetime(
        original["datetime"], utc=True, errors="coerce", format="mixed"
    )
    original = original[
        [
            "plume_id",
            "original_emission_kg_hr",
            "original_uncertainty_kg_hr",
            "original_hidden_emission",
            "original_scene_timestamp",
        ]
    ]
    merged = plumes.merge(original, on="plume_id", how="left", validate="one_to_one")
    merged["analysis_timestamp"] = merged["scene_timestamp"].fillna(
        merged["original_scene_timestamp"]
    )
    merged["analysis_timestamp_source"] = np.where(
        merged["scene_timestamp"].notna(),
        "Live API scene_timestamp",
        "Saved cohort datetime fallback",
    )
    merged["publication_lag_days_analysis_timestamp"] = (
        merged["published_at"] - merged["analysis_timestamp"]
    ).dt.total_seconds() / 86_400.0

    old_rate_present = merged["original_emission_kg_hr"].notna()
    new_rate_present = merged["emission_auto"].notna()
    both_rates = old_rate_present & new_rate_present
    changed_rate = both_rates & ~np.isclose(
        merged["original_emission_kg_hr"],
        merged["emission_auto"],
        rtol=1e-9,
        atol=1e-9,
    )
    changed_hidden = (
        merged["original_hidden_emission"].notna()
        & merged["hide_emission"].notna()
        & (merged["original_hidden_emission"] != merged["hide_emission"].astype(bool))
    )

    drift = {
        "rates_became_available": int((~old_rate_present & new_rate_present).sum()),
        "rates_became_unavailable": int((old_rate_present & ~new_rate_present).sum()),
        "published_rates_changed": int(changed_rate.sum()),
        "hidden_flags_changed": int(changed_hidden.sum()),
        "live_scene_timestamps_missing": int(merged["scene_timestamp"].isna().sum()),
    }
    return merged, drift


# ---------------------------------------------------------------------------
# Carbon Mapper source mapping
# ---------------------------------------------------------------------------

def source_identifier(feature: dict[str, Any], eps_metres: int) -> str:
    """Create a stable local identifier from the source name and coordinates."""

    properties = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]
    key = "|".join(
        [
            str(eps_metres),
            str(properties.get("gas")),
            str(properties.get("sector")),
            str(clean_source_name(properties.get("source_name"))),
            str(coordinates[0]),
            str(coordinates[1]),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"CM{eps_metres}-{digest}"


def source_cluster_sensitivity(
    source_snapshot: dict[str, Any],
    cohort_ids: set[str],
) -> pd.DataFrame:
    """Count only source features that contain at least one fixed-cohort plume."""

    rows: list[dict[str, Any]] = []
    collections = source_snapshot["collections_by_eps_metres"]
    for eps_text, collection in collections.items():
        matched_features = 0
        mapped_ids: set[str] = set()
        api_ids: set[str] = set()
        for feature in collection.get("features", []):
            plume_ids = set((feature.get("properties") or {}).get("plume_ids") or [])
            overlap = plume_ids.intersection(cohort_ids)
            if overlap:
                matched_features += 1
                mapped_ids.update(overlap)
                api_ids.update(plume_ids)
        rows.append(
            {
                "cluster_distance_m": int(eps_text),
                "source_count": matched_features,
                "cohort_plumes_mapped": len(mapped_ids),
                "cohort_plumes_unmapped": len(cohort_ids.difference(mapped_ids)),
                "noncohort_plume_ids_in_matching_sources": len(api_ids.difference(cohort_ids)),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_distance_m").reset_index(drop=True)


def map_primary_sources(
    plumes: pd.DataFrame,
    source_snapshot: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Map each plume to Carbon Mapper's 500 m source feature and summarize it."""

    collection = source_snapshot["collections_by_eps_metres"][
        str(PRIMARY_SOURCE_DISTANCE_M)
    ]
    cohort_ids = set(plumes["plume_id"])
    plume_to_sources: dict[str, list[str]] = {plume_id: [] for plume_id in cohort_ids}
    source_rows: list[dict[str, Any]] = []

    for feature in collection.get("features", []):
        properties = feature.get("properties") or {}
        all_plume_ids = list(properties.get("plume_ids") or [])
        matching_ids = sorted(cohort_ids.intersection(all_plume_ids))
        if not matching_ids:
            continue

        local_source_id = source_identifier(feature, PRIMARY_SOURCE_DISTANCE_M)
        for plume_id in matching_ids:
            plume_to_sources[plume_id].append(local_source_id)

        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [None, None]
        source_rows.append(
            {
                "cm_source_id": local_source_id,
                "source_name": clean_source_name(properties.get("source_name")),
                "source_longitude": coordinates[0] if len(coordinates) > 0 else None,
                "source_latitude": coordinates[1] if len(coordinates) > 1 else None,
                "api_gas": properties.get("gas"),
                "api_sector": properties.get("sector"),
                "api_plume_count": properties.get("plume_count"),
                "api_emission_auto": properties.get("emission_auto"),
                "api_emission_uncertainty_auto": properties.get(
                    "emission_uncertainty_auto"
                ),
                "api_detection_date_count": properties.get("detection_date_count"),
                "api_observation_date_count": properties.get("observation_date_count"),
                "api_persistence": properties.get("persistence"),
                "api_date_count": properties.get("date_count"),
                "api_timestamp_min": properties.get("timestamp_min"),
                "api_timestamp_max": properties.get("timestamp_max"),
                "api_published_at_min": properties.get("published_at_min"),
                "api_published_at_max": properties.get("published_at_max"),
                "api_observation_scene_count": len(
                    properties.get("observation_scenes_names") or []
                ),
                "api_observation_scenes_json": json.dumps(
                    properties.get("observation_scenes_names") or []
                ),
                "api_plume_ids_json": json.dumps(all_plume_ids),
                "cohort_plume_ids_json": json.dumps(matching_ids),
                "cohort_plume_count": len(matching_ids),
                "noncohort_plume_count": len(set(all_plume_ids).difference(cohort_ids)),
            }
        )

    duplicate_mappings = sum(len(source_ids) > 1 for source_ids in plume_to_sources.values())
    unmatched = sum(len(source_ids) == 0 for source_ids in plume_to_sources.values())
    mapping = {
        plume_id: source_ids[0] if source_ids else None
        for plume_id, source_ids in plume_to_sources.items()
    }

    plume_output = plumes.copy()
    plume_output["cm_source_id"] = plume_output["plume_id"].map(mapping)
    raw_sources = pd.DataFrame(source_rows)
    if raw_sources.empty:
        raise RuntimeError("No Carbon Mapper sources overlapped the fixed plume cohort.")

    source_metrics: list[dict[str, Any]] = []
    for source_id, group in plume_output.groupby("cm_source_id", dropna=False):
        if is_blank(source_id):
            continue
        api_row = raw_sources.loc[raw_sources["cm_source_id"] == source_id].iloc[0]
        country_counts = group["country"].value_counts(dropna=False)
        country = country_counts.index[0] if not country_counts.empty else None
        quantified = group["emission_auto"].notna()
        detection_dates = group["analysis_timestamp"].dt.date.nunique()
        source_metrics.append(
            {
                **api_row.to_dict(),
                "country": country,
                "country_iso3": group["country_iso3"].dropna().mode().iloc[0]
                if group["country_iso3"].notna().any()
                else None,
                "sector": group["sector"].dropna().mode().iloc[0]
                if group["sector"].notna().any()
                else None,
                "sector_label": group["sector_label"].mode().iloc[0],
                "cohort_detection_count": len(group),
                "cohort_detection_date_count": detection_dates,
                "cohort_scene_count": group["scene_id"].nunique(),
                "cohort_quantified_count": int(quantified.sum()),
                "cohort_unquantified_count": int((~quantified).sum()),
                "cohort_quantified_percent": round(percent(quantified.sum(), len(group)), 1),
                "cohort_median_emission_kg_hr": safe_median(group["emission_auto"]),
                "cohort_first_detection": group["analysis_timestamp"].min(),
                "cohort_last_detection": group["analysis_timestamp"].max(),
                "source_name_present": not is_blank(api_row["source_name"]),
            }
        )

    sources = pd.DataFrame(source_metrics)
    sources["quantification_coverage"] = np.select(
        [
            sources["cohort_quantified_count"] == 0,
            sources["cohort_unquantified_count"] == 0,
        ],
        ["No public rates", "All detections quantified"],
        default="Mixed public and hidden rates",
    )
    sources["revisit_priority_tier"] = np.select(
        [
            (sources["cohort_quantified_count"] == 0)
            & (sources["cohort_detection_date_count"] >= 2),
            sources["cohort_quantified_count"] == 0,
            sources["cohort_unquantified_count"] > 0,
        ],
        [
            "Tier 1: repeat-detected, never publicly quantified",
            "Tier 2: single-date, never publicly quantified",
            "Tier 3: partly quantified",
        ],
        default="Tier 4: quantified benchmark / control",
    )
    observation_dates = pd.to_numeric(
        sources["api_observation_date_count"], errors="coerce"
    )
    detection_dates = pd.to_numeric(
        sources["api_detection_date_count"], errors="coerce"
    )
    sources["api_null_observation_date_count"] = (
        observation_dates - detection_dates
    ).clip(lower=0)
    sources = sources.sort_values(
        [
            "revisit_priority_tier",
            "cohort_detection_date_count",
            "cohort_detection_count",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    checks = {
        "unmapped_cohort_plumes": unmatched,
        "plumes_mapped_to_multiple_sources": duplicate_mappings,
    }
    return plume_output, sources, checks


# ---------------------------------------------------------------------------
# Audit tables
# ---------------------------------------------------------------------------

def missingness_table(plumes: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    """Measure whether the most decision-relevant API fields contain values."""

    plume_checks = [
        ("Plume", "Live API scene timestamp", plumes["scene_timestamp"].notna()),
        ("Plume", "Analysis timestamp after saved-cohort fallback", plumes["analysis_timestamp"].notna()),
        ("Plume", "Public emission rate", plumes["emission_auto"].notna()),
        (
            "Plume",
            "Public emission uncertainty",
            plumes["emission_uncertainty_auto"].notna(),
        ),
        ("Plume", "Plume quality value", plumes["plume_quality"].notna()),
        ("Plume", "Wind speed", plumes["wind_speed_avg_auto"].notna()),
        ("Plume", "Wind direction", plumes["wind_direction_avg_auto"].notna()),
        ("Plume", "Specific sector code", plumes["sector_is_specific"]),
        ("Plume", "Mission phase", plumes["mission_phase"].notna()),
        ("Plume", "Sensitivity mode", plumes["sensitivity_mode"].notna()),
        ("Plume", "Ground sample distance", plumes["gsd"].notna()),
        ("Plume", "Off-nadir angle", plumes["off_nadir"].notna()),
        ("Plume", "Processing software version", plumes["processing_software"].notna()),
        ("Plume", "Emission product version", plumes["emission_version"].notna()),
        ("Plume", "Plume image", plumes["has_plume_png"]),
        ("Plume", "Concentration raster", plumes["has_con_tif"]),
        ("Plume", "Publication timestamp", plumes["published_at"].notna()),
        (
            "Plume",
            "Non-empty publication-source list",
            plumes["publication_sources_count"] > 0,
        ),
        ("Plume", "Carbon Mapper 500 m source match", plumes["cm_source_id"].notna()),
    ]
    source_checks = [
        (
            "Source",
            "Carbon Mapper spatial-cluster identifier",
            sources["source_name_present"],
        ),
        (
            "Source",
            "Observation-date denominator",
            pd.to_numeric(sources["api_observation_date_count"], errors="coerce").notna(),
        ),
        (
            "Source",
            "Detection-date count",
            pd.to_numeric(sources["api_detection_date_count"], errors="coerce").notna(),
        ),
        (
            "Source",
            "Persistence metric",
            pd.to_numeric(sources["api_persistence"], errors="coerce").notna(),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for level, field, available in plume_checks + source_checks:
        available_series = pd.Series(available).fillna(False).astype(bool)
        total = len(available_series)
        available_count = int(available_series.sum())
        rows.append(
            {
                "level": level,
                "field_or_capability": field,
                "available_count": available_count,
                "missing_count": total - available_count,
                "available_percent": round(percent(available_count, total), 1),
                "missing_percent": round(percent(total - available_count, total), 1),
            }
        )
    return pd.DataFrame(rows)


def endpoint_gap_table() -> pd.DataFrame:
    """List observation-design variables absent from the plume/source endpoints."""

    rows = [
        (
            "Observation inventory",
            "Complete public region-wide scene inventory and per-scene source coverage detail",
            "Public source summaries provide aggregate qualifying observation dates, but the detailed scene-coverage query required authentication in this audit.",
        ),
        (
            "Tasking history",
            "Requested, acquired, rejected, and unpublished attempts",
            "Needed to measure geographic and seasonal selection bias.",
        ),
        (
            "Scene usability",
            "Per-source cloud, haze, valid-pixel fraction, and usable area",
            "The source query uses a documented cloud threshold, but does not expose enough public detail to calculate tropical valid-observation yield.",
        ),
        (
            "Sensitivity",
            "Scene- and source-specific minimum detection limit",
            "Needed before a non-detection can be interpreted as a valid null.",
        ),
        (
            "Retrieval diagnostics",
            "Local background noise, artifact flags, and dual-window agreement",
            "Needed to compare reliability across vegetation, dark ponds, cities, coasts, and glint.",
        ),
        (
            "Quantification provenance",
            "Reason code when emission_auto is hidden or unavailable",
            "Needed to separate wind failure, plume-shape failure, QC suppression, and policy suppression.",
        ),
        (
            "Illumination",
            "Sun elevation and azimuth in the public plume record",
            "Needed to diagnose radiance and surface-dependent retrieval performance.",
        ),
        (
            "Wind provenance",
            "Wind model, analysis time, spatial resolution, and alternative-wind spread",
            "Needed to reproduce and interpret emission-rate uncertainty.",
        ),
        (
            "Surface context",
            "Land cover and retrieval-background class",
            "Needed to quantify performance by tropical source environment.",
        ),
        (
            "Action readiness",
            "Verified facility/operator, stakeholder, intervention, response, and mitigation status",
            "Needed to turn detection evidence into a defensible action pathway.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["gap_category", "not_available_in_public_endpoints", "why_it_matters"],
    )


def country_summary(plumes: pd.DataFrame) -> pd.DataFrame:
    """Summarize detections and quantification coverage by derived country."""

    grouped = (
        plumes.groupby("country", dropna=False)
        .agg(
            detections=("plume_id", "count"),
            sources_500m=("cm_source_id", "nunique"),
            unique_scenes=("scene_id", "nunique"),
            quantified=("emission_auto", lambda values: int(values.notna().sum())),
            hidden_or_unquantified=("emission_auto", lambda values: int(values.isna().sum())),
            median_public_rate_kg_hr=("emission_auto", safe_median),
        )
        .reset_index()
    )
    grouped["quantified_percent"] = (
        100.0 * grouped["quantified"] / grouped["detections"]
    ).round(1)
    grouped["asean_country"] = grouped["country"].isin(ASEAN_COUNTRIES)

    missing_asean = sorted(set(ASEAN_COUNTRIES).difference(grouped["country"]))
    if missing_asean:
        zeros = pd.DataFrame(
            {
                "country": missing_asean,
                "detections": 0,
                "sources_500m": 0,
                "unique_scenes": 0,
                "quantified": 0,
                "hidden_or_unquantified": 0,
                "median_public_rate_kg_hr": np.nan,
                "quantified_percent": np.nan,
                "asean_country": True,
            }
        )
        grouped = pd.concat([grouped, zeros], ignore_index=True)
    return grouped.sort_values(
        ["asean_country", "detections", "country"], ascending=[False, False, True]
    ).reset_index(drop=True)


def sector_summary(plumes: pd.DataFrame) -> pd.DataFrame:
    """Summarize sector concentration and rate availability."""

    grouped = (
        plumes.groupby(["sector", "sector_label"], dropna=False)
        .agg(
            detections=("plume_id", "count"),
            sources_500m=("cm_source_id", "nunique"),
            quantified=("emission_auto", lambda values: int(values.notna().sum())),
            hidden_or_unquantified=("emission_auto", lambda values: int(values.isna().sum())),
            median_public_rate_kg_hr=("emission_auto", safe_median),
        )
        .reset_index()
    )
    grouped["detection_share_percent"] = (
        100.0 * grouped["detections"] / len(plumes)
    ).round(1)
    grouped["quantified_percent"] = (
        100.0 * grouped["quantified"] / grouped["detections"]
    ).round(1)
    return grouped.sort_values("detections", ascending=False).reset_index(drop=True)


def monthly_summary(plumes: pd.DataFrame) -> pd.DataFrame:
    """Summarize the fixed cohort by acquisition month."""

    monthly = plumes.copy()
    monthly["month"] = monthly["analysis_timestamp"].dt.strftime("%Y-%m")
    output = (
        monthly.groupby("month")
        .agg(
            detections=("plume_id", "count"),
            sources_500m=("cm_source_id", "nunique"),
            unique_scenes=("scene_id", "nunique"),
            quantified=("emission_auto", lambda values: int(values.notna().sum())),
            hidden_or_unquantified=("emission_auto", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )
    output["quantified_percent"] = (
        100.0 * output["quantified"] / output["detections"]
    ).round(1)
    return output


def add_quantification_context(plumes: pd.DataFrame) -> pd.DataFrame:
    """Add transparent bins used only for descriptive completeness comparisons."""

    output = plumes.copy()
    output["absolute_off_nadir_degrees"] = output["off_nadir"].abs()
    output["off_nadir_bin"] = pd.cut(
        output["absolute_off_nadir_degrees"],
        bins=[-np.inf, 5, 15, 30, np.inf],
        labels=["0-5°", ">5-15°", ">15-30°", ">30°"],
    )
    output["wind_speed_bin"] = pd.cut(
        output["wind_speed_avg_auto"],
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["0-2 m/s", ">2-4 m/s", ">4-6 m/s", ">6 m/s"],
    )
    output["plumes_in_same_scene"] = output.groupby("scene_id")["plume_id"].transform(
        "count"
    )
    output["scene_plume_count_bin"] = pd.cut(
        output["plumes_in_same_scene"],
        bins=[0, 1, 3, np.inf],
        labels=["1 plume", "2-3 plumes", "4+ plumes"],
        include_lowest=True,
    )
    output["offshore_label"] = np.where(
        output["is_offshore"].fillna(False).astype(bool), "Offshore", "Not flagged offshore"
    )
    output["analysis_year"] = output["analysis_timestamp"].dt.year.astype("Int64").astype(str)
    return output


def quantification_context_summary(plumes: pd.DataFrame) -> pd.DataFrame:
    """Compare rate availability across acquisition contexts without causal claims."""

    dimensions = [
        ("Sensitivity mode", "sensitivity_mode"),
        ("Mission phase", "mission_phase"),
        ("Absolute off-nadir angle", "off_nadir_bin"),
        ("Wind speed used", "wind_speed_bin"),
        ("Plumes in same scene", "scene_plume_count_bin"),
        ("Offshore flag", "offshore_label"),
        ("Acquisition year", "analysis_year"),
    ]
    rows: list[dict[str, Any]] = []
    for dimension_name, column in dimensions:
        for category, group in plumes.groupby(column, observed=True, dropna=False):
            quantified = int(group["emission_auto"].notna().sum())
            rows.append(
                {
                    "dimension": dimension_name,
                    "category": "Missing" if pd.isna(category) else str(category),
                    "plume_count": len(group),
                    "quantified_count": quantified,
                    "hidden_or_unquantified_count": len(group) - quantified,
                    "quantified_percent": round(percent(quantified, len(group)), 1),
                    "median_public_rate_kg_hr": safe_median(group["emission_auto"]),
                }
            )
    return pd.DataFrame(rows)


def source_observation_summary(sources: pd.DataFrame) -> pd.DataFrame:
    """Summarize Carbon Mapper's public source-level observation opportunity fields."""

    observation_dates = pd.to_numeric(
        sources["api_observation_date_count"], errors="coerce"
    )
    detection_dates = pd.to_numeric(
        sources["api_detection_date_count"], errors="coerce"
    )
    persistence = pd.to_numeric(sources["api_persistence"], errors="coerce")
    null_dates = pd.to_numeric(
        sources["api_null_observation_date_count"], errors="coerce"
    )
    metrics = [
        ("source_count", len(sources), "sources"),
        ("sources_with_observation_denominator", int(observation_dates.notna().sum()), "sources"),
        ("sources_with_two_or_more_qualifying_observation_dates", int((observation_dates >= 2).sum()), "sources"),
        ("sources_with_at_least_one_qualifying_null_date", int((null_dates >= 1).sum()), "sources"),
        ("median_qualifying_observation_dates", float(observation_dates.median()), "dates per source"),
        ("maximum_qualifying_observation_dates", float(observation_dates.max()), "dates"),
        ("median_detection_dates", float(detection_dates.median()), "dates per source"),
        ("median_persistence", float(persistence.median()), "fraction"),
        ("minimum_persistence", float(persistence.min()), "fraction"),
        ("maximum_persistence", float(persistence.max()), "fraction"),
        ("sources_with_persistence_equal_to_one", int(np.isclose(persistence, 1.0).sum()), "sources"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "unit"])


def provisional_portfolio_design() -> pd.DataFrame:
    """Translate the observed gaps into a provisional, mutually exclusive allocation."""

    rows = [
        {
            "portfolio_component": "Missing-country first looks",
            "image_count": 8,
            "evidence_from_audit": "Brunei, Laos, Singapore, and Timor-Leste have zero ASEAN-land detections in the fixed cohort.",
            "selection_rule": "Two facility candidates per missing country, chosen from an external actionable-source registry.",
            "intended_learning": "Separate geographic tasking gaps from source absence and test transfer to new tropical settings.",
        },
        {
            "portfolio_component": "Quantification-recovery revisits",
            "image_count": 8,
            "evidence_from_audit": "12 source clusters have no public rate and each appears on only one cohort date.",
            "selection_rule": "Revisit eight Tier-2 sources balanced across countries, sectors, wind regimes, and surfaces.",
            "intended_learning": "Test whether a second acquisition converts a location-only detection into a quantified result.",
        },
        {
            "portfolio_component": "Underrepresented sector and surface stress tests",
            "image_count": 6,
            "evidence_from_audit": "88.1% of detections are landfill-coded; wastewater, coal, oil/gas, POME-like dark ponds, coasts, and offshore settings are sparse.",
            "selection_rule": "Choose distinct actionable source classes and retrieval backgrounds; do not add more landfill scenes by default.",
            "intended_learning": "Measure transferability and artifact behavior outside the dominant landfill sample.",
        },
        {
            "portfolio_component": "Wet/dry seasonal pairs",
            "image_count": 4,
            "evidence_from_audit": "The public positive catalogue cannot isolate monsoon effects or valid-observation yield.",
            "selection_rule": "Select two high-priority sites and acquire one wet-season and one dry-season observation at each.",
            "intended_learning": "Test cloud, moisture, surface, and source-seasonality effects using paired sites.",
        },
        {
            "portfolio_component": "Quantified controls and cloud contingency",
            "image_count": 4,
            "evidence_from_audit": "39 source clusters are quantified on every cohort detection, while tropical acquisitions remain weather-risky.",
            "selection_rule": "Reserve benchmark revisits and allow failed cloudy acquisitions to be retried without breaking the design.",
            "intended_learning": "Anchor retrieval comparisons and protect the experiment from predictable tropical data loss.",
        },
    ]
    design = pd.DataFrame(rows)
    if int(design["image_count"].sum()) != 30:
        raise AssertionError("The provisional acquisition portfolio must total 30 images.")
    return design


def build_summary_metrics(
    plumes: pd.DataFrame,
    sources: pd.DataFrame,
    sensitivity: pd.DataFrame,
    country: pd.DataFrame,
    drift: dict[str, int],
    mapping_checks: dict[str, int],
    fetched_at: datetime,
) -> pd.DataFrame:
    """Create one compact key-value table for downstream use."""

    quantified = int(plumes["emission_auto"].notna().sum())
    unquantified = len(plumes) - quantified
    all_unquantified_sources = int(
        (sources["quantification_coverage"] == "No public rates").sum()
    )
    partial_sources = int(
        (sources["quantification_coverage"] == "Mixed public and hidden rates").sum()
    )
    singleton_sources = int((sources["cohort_detection_date_count"] == 1).sum())
    repeated_sources = int((sources["cohort_detection_date_count"] >= 2).sum())
    sources_without_cohort_dates = int(
        (sources["cohort_detection_date_count"] == 0).sum()
    )
    asean_with_detections = int(
        ((country["asean_country"]) & (country["detections"] > 0)).sum()
    )
    asean_without_detections = int(
        ((country["asean_country"]) & (country["detections"] == 0)).sum()
    )
    primary_source_count = int(
        sensitivity.loc[
            sensitivity["cluster_distance_m"] == PRIMARY_SOURCE_DISTANCE_M,
            "source_count",
        ].iloc[0]
    )

    metrics = [
        ("audit_fetched_at_utc", iso_utc(fetched_at), "timestamp", "Live API audit date"),
        ("fixed_cohort_plumes", len(plumes), "plume observations", "Existing SEA CSV"),
        ("unique_tanager_scenes", plumes["scene_id"].nunique(), "scenes", "Plume-bearing scenes only"),
        ("carbon_mapper_sources_500m", primary_source_count, "sources", "Native API grouping"),
        ("publicly_quantified_plumes", quantified, "plumes", "Rate and uncertainty available"),
        ("unquantified_or_hidden_plumes", unquantified, "plumes", "No public emission_auto"),
        ("unquantified_or_hidden_percent", round(percent(unquantified, len(plumes)), 1), "percent", "Fixed cohort"),
        ("sources_with_no_public_rates", all_unquantified_sources, "sources", "Every cohort detection lacks rate"),
        ("sources_with_mixed_rate_access", partial_sources, "sources", "Some public, some hidden"),
        ("single_detection_date_sources", singleton_sources, "sources", "Cohort only"),
        ("repeat_detection_date_sources", repeated_sources, "sources", "At least two dates in cohort"),
        ("sources_without_cohort_detection_dates", sources_without_cohort_dates, "sources", "After documented timestamp fallback"),
        ("asean_countries_with_land_detections", asean_with_detections, "countries", "Point-in-polygon"),
        ("asean_countries_without_land_detections", asean_without_detections, "countries", "Not evidence of no emissions"),
        ("plume_quality_values_present", int(plumes["plume_quality"].notna().sum()), "plumes", "Field may exist but be null"),
        ("live_scene_timestamps_missing", drift["live_scene_timestamps_missing"], "plumes", "Saved cohort datetime used for analysis"),
        ("rates_became_available_since_saved_csv", drift["rates_became_available"], "plumes", "Live API drift"),
        ("rates_became_unavailable_since_saved_csv", drift["rates_became_unavailable"], "plumes", "Live API drift"),
        ("published_rates_changed_since_saved_csv", drift["published_rates_changed"], "plumes", "Exact numeric comparison"),
        ("hidden_flags_changed_since_saved_csv", drift["hidden_flags_changed"], "plumes", "Live API drift"),
        ("unmapped_cohort_plumes", mapping_checks["unmapped_cohort_plumes"], "plumes", "500 m source mapping"),
        ("plumes_mapped_to_multiple_sources", mapping_checks["plumes_mapped_to_multiple_sources"], "plumes", "Should be zero"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "unit", "notes"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_summary_figure(
    plumes: pd.DataFrame,
    sources: pd.DataFrame,
    countries: pd.DataFrame,
    monthly: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a four-panel summary of the observed completeness gaps."""

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    quantified = int(plumes["emission_auto"].notna().sum())
    unquantified = len(plumes) - quantified
    axes[0, 0].bar(
        ["Public rate +\nuncertainty", "Hidden /\nunavailable"],
        [quantified, unquantified],
        color=["#2a9d8f", "#e76f51"],
    )
    for index, count in enumerate([quantified, unquantified]):
        axes[0, 0].text(index, count + 5, f"{count}\n({percent(count, len(plumes)):.1f}%)", ha="center")
    axes[0, 0].set_ylim(0, max(quantified, unquantified) * 1.20)
    axes[0, 0].set_ylabel("Tanager plume observations")
    axes[0, 0].set_title("A. Quantification completeness (fixed 385-plume cohort)")

    country_plot = countries[(countries["detections"] > 0)].copy()
    country_plot = country_plot.sort_values("detections")
    axes[0, 1].barh(
        country_plot["country"],
        country_plot["quantified"],
        color="#2a9d8f",
        label="Publicly quantified",
    )
    axes[0, 1].barh(
        country_plot["country"],
        country_plot["hidden_or_unquantified"],
        left=country_plot["quantified"],
        color="#e76f51",
        label="Hidden / unavailable",
    )
    axes[0, 1].set_xlabel("Plume observations")
    axes[0, 1].set_title("B. Detection record by land country")
    axes[0, 1].legend(fontsize=8)

    month_x = np.arange(len(monthly))
    axes[1, 0].bar(
        month_x,
        monthly["quantified"],
        color="#2a9d8f",
        label="Publicly quantified",
    )
    axes[1, 0].bar(
        month_x,
        monthly["hidden_or_unquantified"],
        bottom=monthly["quantified"],
        color="#e76f51",
        label="Hidden / unavailable",
    )
    axes[1, 0].set_xticks(month_x)
    axes[1, 0].set_xticklabels(monthly["month"], rotation=45, ha="right")
    axes[1, 0].set_ylabel("Plume observations")
    axes[1, 0].set_title("C. When the fixed cohort was observed")
    axes[1, 0].legend(fontsize=8)

    source_status = sources["quantification_coverage"].value_counts()
    status_order = [
        "No public rates",
        "Mixed public and hidden rates",
        "All detections quantified",
    ]
    status_colors = ["#e76f51", "#f4a261", "#2a9d8f"]
    counts = [int(source_status.get(label, 0)) for label in status_order]
    axes[1, 1].barh(status_order, counts, color=status_colors)
    for index, count in enumerate(counts):
        axes[1, 1].text(count + 0.5, index, str(count), va="center")
    axes[1, 1].set_xlabel("Carbon Mapper sources (500 m clustering)")
    axes[1, 1].set_title("D. Is each source ever publicly quantified-")

    figure.suptitle(
        "Carbon Mapper Tanager methane record over Southeast Asia: published evidence gaps",
        fontsize=15,
        y=1.01,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_source_map(
    sources: pd.DataFrame,
    country_boundaries: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Map source recurrence and public-rate coverage without implying validation."""

    figure, axis = plt.subplots(figsize=(12, 7.5))
    regional = country_boundaries.cx[88:145, -15:28]
    regional.plot(ax=axis, color="#f2efe9", edgecolor="#777777", linewidth=0.5)

    source_points = gpd.GeoDataFrame(
        sources.copy(),
        geometry=[
            Point(longitude, latitude)
            for longitude, latitude in zip(
                sources["source_longitude"], sources["source_latitude"]
            )
        ],
        crs="EPSG:4326",
    )
    color_map = {
        "No public rates": "#d73027",
        "Mixed public and hidden rates": "#fdae61",
        "All detections quantified": "#1a9850",
    }
    for status, group in source_points.groupby("quantification_coverage"):
        marker_size = 30 + 12 * np.sqrt(group["cohort_detection_count"])
        group.plot(
            ax=axis,
            color=color_map.get(status, "#777777"),
            markersize=marker_size,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            label=f"{status} (n={len(group)})",
        )

    axis.set_xlim(SEA_BBOX[0], SEA_BBOX[2])
    axis.set_ylim(SEA_BBOX[1], SEA_BBOX[3])
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(
        "Tanager CH4 sources in the fixed SEA cohort\n"
        "Carbon Mapper 500 m clusters; marker size increases with plume detections"
    )
    axis.legend(loc="lower left", fontsize=8, framealpha=0.95)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Narrative report
# ---------------------------------------------------------------------------

def build_report(
    plumes: pd.DataFrame,
    sources: pd.DataFrame,
    missingness: pd.DataFrame,
    endpoint_gaps: pd.DataFrame,
    countries: pd.DataFrame,
    sectors: pd.DataFrame,
    quantification_context: pd.DataFrame,
    source_observations: pd.DataFrame,
    portfolio_design: pd.DataFrame,
    sensitivity: pd.DataFrame,
    drift: dict[str, int],
    mapping_checks: dict[str, int],
    fetched_at: datetime,
    plume_snapshot_path: Path,
    source_snapshot_path: Path,
) -> str:
    """Write the evidence-led interpretation that accompanies the CSV tables."""

    plume_count = len(plumes)
    quantified = int(plumes["emission_auto"].notna().sum())
    unquantified = plume_count - quantified
    source_count = len(sources)
    source_no_rates = int(
        (sources["quantification_coverage"] == "No public rates").sum()
    )
    source_mixed = int(
        (sources["quantification_coverage"] == "Mixed public and hidden rates").sum()
    )
    source_all_rates = int(
        (sources["quantification_coverage"] == "All detections quantified").sum()
    )
    repeated_sources = int((sources["cohort_detection_date_count"] >= 2).sum())
    repeat_never_quantified = int(
        (
            (sources["cohort_detection_date_count"] >= 2)
            & (sources["cohort_quantified_count"] == 0)
        ).sum()
    )

    asean_rows = countries[countries["asean_country"]].copy()
    asean_detected = asean_rows.loc[asean_rows["detections"] > 0, "country"].tolist()
    asean_not_detected = asean_rows.loc[asean_rows["detections"] == 0, "country"].tolist()
    non_asean_rows = countries[
        (~countries["asean_country"]) & (countries["detections"] > 0)
    ]
    non_asean_names = non_asean_rows["country"].tolist()

    quality_present = int(plumes["plume_quality"].notna().sum())
    source_names_present = int(sources["source_name_present"].sum())
    observation_denominators = int(sources["api_observation_date_count"].notna().sum())
    publication_source_present = int((plumes["publication_sources_count"] > 0).sum())

    lag = plumes["publication_lag_days_analysis_timestamp"].dropna()
    median_lag = float(lag.median()) if not lag.empty else float("nan")
    p95_lag = float(lag.quantile(0.95)) if not lag.empty else float("nan")

    min_date = plumes["analysis_timestamp"].min().date().isoformat()
    max_date = plumes["analysis_timestamp"].max().date().isoformat()
    live_timestamp_missing = int(plumes["scene_timestamp"].isna().sum())
    asean_land_plumes = int(plumes["is_asean_land_point"].sum())
    asean_source_count = int(sources["country"].isin(ASEAN_COUNTRIES).sum())
    non_asean_or_unassigned_plumes = plume_count - asean_land_plumes

    if repeat_never_quantified:
        repeat_gap_text = (
            f"There are {repeated_sources} sources detected on at least two dates; "
            f"{repeat_never_quantified} of those repeated sources are never publicly quantified. "
            "Those repeat-detected-but-never-quantified locations are the clearest "
            "evidence-led revisit candidates."
        )
    else:
        repeat_gap_text = (
            f"There are {repeated_sources} sources detected on at least two dates, and every one "
            "has at least one public rate. The 12 never-quantified sources each appear on only one "
            "cohort date, so a deliberate revisit is needed to test whether their missing "
            "quantification was scene-specific or persistent."
        )

    country_report = asean_rows[
        [
            "country",
            "detections",
            "sources_500m",
            "quantified",
            "hidden_or_unquantified",
            "quantified_percent",
        ]
    ].copy()
    sector_report = sectors[
        [
            "sector",
            "sector_label",
            "detections",
            "sources_500m",
            "detection_share_percent",
            "quantified_percent",
        ]
    ].copy()
    missing_report = missingness[
        [
            "level",
            "field_or_capability",
            "available_count",
            "missing_count",
            "missing_percent",
        ]
    ].copy()

    # Format only for display; numeric values remain numeric in the CSV outputs.
    country_report["quantified_percent"] = country_report["quantified_percent"].map(
        lambda value: "" if pd.isna(value) else f"{value:.1f}%"
    )
    sector_report["detection_share_percent"] = sector_report[
        "detection_share_percent"
    ].map(lambda value: f"{value:.1f}%")
    sector_report["quantified_percent"] = sector_report["quantified_percent"].map(
        lambda value: f"{value:.1f}%"
    )
    missing_report["missing_percent"] = missing_report["missing_percent"].map(
        lambda value: f"{value:.1f}%"
    )
    context_report = quantification_context.copy()
    context_report["quantified_percent"] = context_report["quantified_percent"].map(
        lambda value: f"{value:.1f}%"
    )
    context_report = context_report[
        [
            "dimension",
            "category",
            "plume_count",
            "quantified_count",
            "hidden_or_unquantified_count",
            "quantified_percent",
        ]
    ]
    observation_values = source_observations.set_index("metric")["value"]
    source_obs_two_plus = int(
        observation_values["sources_with_two_or_more_qualifying_observation_dates"]
    )
    source_obs_null = int(
        observation_values["sources_with_at_least_one_qualifying_null_date"]
    )
    source_obs_median = float(observation_values["median_qualifying_observation_dates"])
    source_persistence_median = float(observation_values["median_persistence"])

    source_sensitivity_report = sensitivity.copy()
    source_sensitivity_report["interpretation"] = source_sensitivity_report[
        "cluster_distance_m"
    ].map(
        {
            500: "Primary Carbon Mapper-style facility grouping",
            1_000: "Moderate grouping sensitivity",
            5_000: "Comparable radius to the earlier cross-sensor notebook",
        }
    )

    lines = [
        "# Carbon Mapper–Tanager Southeast Asia Gap Audit",
        "",
        f"**Audit date (UTC):** {iso_utc(fetched_at)}  ",
        f"**Fixed cohort:** {plume_count} Carbon Mapper Tanager CH₄ plume observations, {min_date} to {max_date}  ",
        f"**Primary source definition:** Carbon Mapper API clustering with `eps={PRIMARY_SOURCE_DISTANCE_M}` m  ",
        "**Purpose:** identify what the public Tanager record establishes, what is missing, and which gaps justify a designed 30-acquisition Southeast Asian campaign.",
        "",
        "## Executive finding",
        "",
        f"The headline number is **{plume_count} plume observations, not {plume_count} sites**. "
        f"Carbon Mapper's 500 m source grouping maps them to **{source_count} source clusters**. "
        f"Only **{quantified}/{plume_count} ({percent(quantified, plume_count):.1f}%)** have a public emission rate and uncertainty; "
        f"**{unquantified}/{plume_count} ({percent(unquantified, plume_count):.1f}%)** are hidden or unquantified. "
        "The API does not expose a reason code for those missing rates, so this audit cannot say whether each case failed because of wind, retrieval quality, plume morphology, policy, or another rule.",
        "",
        f"At source level, {source_no_rates}/{source_count} clusters have no public rate for any cohort detection, "
        f"{source_mixed}/{source_count} have a mixture of public and hidden rates, and {source_all_rates}/{source_count} are quantified on every cohort detection. "
        + repeat_gap_text,
        "",
        "The larger strategic gap is a denominator-detail problem. Carbon Mapper's public source summaries provide qualifying observation-date counts and persistence, using a documented default maximum cloud cover of 25%. However, the unauthenticated API does not expose a complete region-wide record of tasking attempts, rejected or unpublished acquisitions, source-level usable pixels, or scene-specific minimum detection limits; the detailed scene-coverage query returned `401 Unauthorized` during this audit. Therefore, **absence from these 385 records cannot be called a Tanager non-detection, and a country with zero records cannot be called emission-free**. This directly supports the rationale's proposed tropical-observability experiment rather than another plume portal.",
        "",
        "## 1. Cohort and source definitions",
        "",
        "The cohort is held fixed to the IDs saved by `01_sea_data_census.ipynb`; live global counts are not substituted. Every ID was looked up again through the Carbon Mapper annotated-plume endpoint. Source counts are reported at three clustering distances because the word *site* is not a sensor measurement-it depends on the spatial grouping rule. Carbon Mapper's `source_name` is a machine-generated identifier for an ephemeral spatial cluster, not a verified facility or operator name.",
        "",
        markdown_table(
            source_sensitivity_report,
            [
                "cluster_distance_m",
                "source_count",
                "cohort_plumes_mapped",
                "cohort_plumes_unmapped",
                "interpretation",
            ],
        ),
        "",
        f"At 500 m, {mapping_checks['unmapped_cohort_plumes']} cohort plumes are unmapped and {mapping_checks['plumes_mapped_to_multiple_sources']} map to more than one source. The earlier notebook's one-pass 5 km result should not be used as the authoritative site count because single-linkage grouping can depend on row order; the API grouping above is explicit and reproducible.",
        "",
        "## 2. What is missing from the 385 published plume records-",
        "",
        markdown_table(missing_report),
        "",
        f"The live API omits `scene_timestamp` for {live_timestamp_missing}/{plume_count} records even though the saved cohort contains those dates. The audit preserves this as a missingness result and uses the saved values only as an explicit analysis fallback. The `plume_quality` field contains a value for {quality_present}/{plume_count} records. A null field is not interpreted as a bad plume; it means the public response does not supply that label. A non-empty `publication_sources` list appears on {publication_source_present}/{plume_count} records. Carbon Mapper spatial-cluster identifiers are present for {source_names_present}/{source_count} source clusters, and an observation-date denominator is present for {observation_denominators}/{source_count} clusters.",
        "",
        f"Using the live timestamp where present and the documented saved-cohort fallback otherwise, the median delay from acquisition to publication is {median_lag:.1f} days and the 95th percentile is {p95_lag:.1f} days. This describes this fixed public cohort; it is not a guaranteed service-level commitment.",
        "",
        "### Quantification context (descriptive, not causal)",
        "",
        markdown_table(context_report),
        "",
        "The percentage differences above show where hidden/unavailable rates are concentrated in this cohort. They do **not** identify the reason a rate is missing: sensitivity mode, geometry, wind, mission phase, source type, scene complexity, and publication rules can be confounded. Carbon Mapper does not expose a per-record suppression reason, so causal language would be unjustified.",
        "",
        "### What the public source-level denominator does provide",
        "",
        f"At the 500 m source definition, {source_obs_two_plus}/{source_count} sources have at least two qualifying observation dates and {source_obs_null}/{source_count} have at least one qualifying observation date without a detection in the provider summary. The median source has {source_obs_median:.1f} qualifying observation dates, and the median provider persistence is {source_persistence_median:.2f}. These aggregates are valuable, but they do not replace scene-level tropical usability and detection-limit metadata.",
        "",
        markdown_table(source_observations),
        "",
        "## 3. Geographic coverage-and what it does not prove",
        "",
        f"Land-country assignment uses each API point and Natural Earth boundaries. Of the {plume_count} rectangle-filtered records, {asean_land_plumes} are on ASEAN land and {non_asean_or_unassigned_plumes} fall outside ASEAN land or cannot be assigned to a land polygon. The {asean_land_plumes} ASEAN-land detections map to {asean_source_count} primary source clusters. The ASEAN countries with at least one land detection are {list_text(asean_detected)}. The ASEAN countries with no land detection in the fixed cohort are {list_text(asean_not_detected)}. "
        + (
            f"The rectangular notebook frame also contains detections assigned to {list_text(non_asean_names)}; these must not be described as ASEAN detections."
            if non_asean_names
            else "No fixed-cohort land points were assigned to non-ASEAN countries."
        ),
        "",
        markdown_table(country_report),
        "",
        "A zero in this table combines several unknowns: the satellite may not have been tasked, an acquisition may have been unusable, a plume may have been below the scene threshold, an emitter may have been inactive, or no source may have been present. The 30-image case should be framed as an experiment that separates these possibilities, not as a map of where methane does or does not exist.",
        "",
        "## 4. Sector concentration",
        "",
        markdown_table(sector_report),
        "",
        "The record is highly concentrated in whatever sectors Carbon Mapper has already targeted and published. Raw counts therefore describe the public acquisition-and-detection record, not the true regional sector mix. This is precisely why the new acquisition portfolio should reserve images for underrepresented POME/wastewater, coal, oil-and-gas, urban, coastal, and offshore environments instead of simply revisiting the sector with the most existing detections.",
        "",
        "## 5. Gaps that cannot be counted from the current endpoints",
        "",
        markdown_table(endpoint_gaps),
        "",
        "These are not criticisms of the quality of Carbon Mapper's published plume products. They are variables required for the narrower research question in `SEA_PROJECT_RATIONALE_AND_NEXT_STEPS.md`: how tropical observing conditions affect valid-observation yield and how scarce Tanager tasking should be allocated.",
        "",
        "## 6. Evidence-led case for additional Tanager acquisitions",
        "",
        "The audit supports a strong but bounded argument:",
        "",
        f"1. **Tanager is already uniquely productive in the SEA study frame.** The fixed rectangle-filtered cohort contains {plume_count} facility-scale CH₄ detections across {source_count} Carbon Mapper source clusters; {asean_land_plumes} detections and {asean_source_count} source clusters are on ASEAN land. This demonstrates that the sensor can find actionable-scale emitters in a difficult region while correcting the earlier geographic shorthand.",
        f"2. **Detection is not yet equivalent to quantification.** {unquantified} plume records lack a public rate and uncertainty, and {source_no_rates} source clusters are never publicly quantified in this cohort. Repeat acquisition under different wind, cloud, illumination, and surface conditions can test which gaps are recoverable.",
        "3. **Country and sector coverage is opportunity-driven.** The public positives cannot identify true absence because the full observation denominator is unavailable. New acquisitions should deliberately fill geographic and sector cells, with failed and usable observations recorded as outcomes.",
        "4. **Tropical reliability is the novel contribution.** Pair operational products with native-radiance diagnostics, explicit valid-null criteria, and scene-specific limits across landfills, dark POME ponds, vegetation, cities, coasts, and offshore glint.",
        "5. **The 30 images should be a portfolio, not a top-30 emitter list.** Use some for first looks in missing country/sector cells, some for repeat-detected but never-quantified sources, some for wet/dry seasonal pairs, and some as quantified benchmark controls. The exact split should follow target feasibility and stakeholder readiness.",
        "",
        "### Provisional 30-image design envelope",
        "",
        "This allocation is a testable starting constraint derived from the audit, not the final target list. Facility coordinates still require the separate actionable-source registry described in the project rationale. Final selections should combine objectives where possible, but each image must have one primary purpose so the portfolio can be evaluated honestly.",
        "",
        markdown_table(portfolio_design),
        "",
        "The generated `revisit_candidates.csv` is therefore a triage list, not a final allegation or tasking plan. Tier 1 contains sources detected on multiple dates but never publicly quantified; Tier 4 contains quantified sources that can serve as controls. New-location candidates still require a separate facility registry because a detection-only catalogue cannot list facilities Tanager never observed.",
        "",
        "## 7. Data-version and interpretation notes",
        "",
        f"Compared with the saved census CSV, {drift['rates_became_available']} rates became available, {drift['rates_became_unavailable']} became unavailable, {drift['published_rates_changed']} published numeric rates changed, and {drift['hidden_flags_changed']} hidden-rate flags changed in the live API snapshot. However, {drift['live_scene_timestamps_missing']} currently published records no longer expose `scene_timestamp` even though the saved extract contains it. This confirms that the snapshot date and fallback rule belong beside every statistic.",
        "",
        "Guardrails:",
        "",
        "- `hide_emission=true` or a null rate is reported as *hidden/unavailable*, not as a failed retrieval unless Carbon Mapper provides a reason.",
        "- `plume_quality=null` is reported as a missing public value, not as bad quality.",
        "- Carbon Mapper persistence and observation counts are provider source-level summaries; the cohort-only detection counts are kept separate.",
        "- Country assignment is a derived land-polygon join. Offshore points remain `Offshore / boundary-unassigned` and are not forced into the nearest jurisdiction.",
        "- Instantaneous plume rates are not annualized, and source recurrence is not claimed to prove continuous emissions.",
        "- The API cluster identifier is not a facility/operator name. Any later operator attribution requires separate evidence and is not a legal or compliance finding.",
        "",
        "## 8. Reproducibility and sources",
        "",
        f"- Carbon Mapper Data API documentation: {API_DOCS_URL}",
        f"- Carbon Mapper product guide: {PRODUCT_GUIDE_URL}",
        f"- Fixed cohort: `{COHORT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Exact plume API snapshot: `{plume_snapshot_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Source API snapshot: `{source_snapshot_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Script: `scripts/analyze_carbon_mapper_tanager_sea.py`",
        "- Country polygons: Natural Earth 1:10m Admin 0 Countries (public domain)",
        "",
        "All Carbon Mapper statistics in this report are computed from the public API snapshot identified above. Carbon Mapper should be cited as the data provider and its current Terms of Use should be checked before redistribution.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete fixed-cohort audit and save all outputs."""

    arguments = parse_arguments()
    fetched_at = utc_now()
    date_stamp = fetched_at.date().isoformat()

    for folder in [RAW_DIR, ANALYSIS_DIR, FIGURE_DIR, REFERENCE_DIR, REPORT_PATH.parent]:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Fixed cohort: {COHORT_PATH}")
    cohort = pd.read_csv(COHORT_PATH)
    validate_cohort(cohort)
    cohort_ids = cohort["plume_id"].astype(str).tolist()
    cohort_id_set = set(cohort_ids)
    print(f"Validated fixed cohort: {len(cohort)} unique plume IDs")

    plume_snapshot_path = RAW_DIR / f"carbon_mapper_tanager_sea_plumes_api_{date_stamp}.json"
    source_snapshot_path = RAW_DIR / f"carbon_mapper_tanager_sea_sources_api_{date_stamp}.json"
    if not arguments.refresh:
        plume_caches = sorted(RAW_DIR.glob("carbon_mapper_tanager_sea_plumes_api_*.json"))
        source_caches = sorted(RAW_DIR.glob("carbon_mapper_tanager_sea_sources_api_*.json"))
        if plume_caches:
            plume_snapshot_path = plume_caches[-1]
        if source_caches:
            source_snapshot_path = source_caches[-1]

    session = make_session()
    plume_snapshot = fetch_plume_snapshot(
        session,
        cohort_ids,
        plume_snapshot_path,
        arguments.refresh,
        fetched_at,
    )
    source_snapshot = fetch_source_snapshots(
        session,
        source_snapshot_path,
        arguments.refresh,
        fetched_at,
    )
    # When a cached response is reused, its retrieval time-not today's script
    # execution time-is the correct timestamp for every reported API statistic.
    fetched_at = pd.Timestamp(
        plume_snapshot["metadata"]["fetched_at_utc"]
    ).to_pydatetime()

    returned_ids = [str(item.get("plume_id")) for item in plume_snapshot.get("items", [])]
    returned_id_set = set(returned_ids)
    duplicate_api_ids = len(returned_ids) - len(returned_id_set)
    missing_api_ids = cohort_id_set.difference(returned_id_set)
    unexpected_api_ids = returned_id_set.difference(cohort_id_set)
    if duplicate_api_ids or missing_api_ids or unexpected_api_ids:
        raise RuntimeError(
            "Exact plume lookup did not return a one-to-one cohort: "
            f"duplicates={duplicate_api_ids}, missing={len(missing_api_ids)}, "
            f"unexpected={len(unexpected_api_ids)}"
        )

    plumes = flatten_plumes(plume_snapshot["items"])
    if not (plumes["instrument"] == "tan").all():
        raise ValueError("A non-Tanager record appeared in the fixed cohort lookup.")
    if not (plumes["gas"] == "CH4").all():
        raise ValueError("A non-CH4 record appeared in the fixed cohort lookup.")

    country_shape_path = ensure_country_boundaries(session)
    country_boundaries = gpd.read_file(country_shape_path).to_crs("EPSG:4326")
    plumes = assign_countries(plumes, country_boundaries)
    plumes, drift = compare_with_original(plumes, cohort)
    plumes = add_quantification_context(plumes)

    sensitivity = source_cluster_sensitivity(source_snapshot, cohort_id_set)
    plumes, sources, mapping_checks = map_primary_sources(plumes, source_snapshot)

    missingness = missingness_table(plumes, sources)
    endpoint_gaps = endpoint_gap_table()
    countries = country_summary(plumes)
    sectors = sector_summary(plumes)
    monthly = monthly_summary(plumes)
    quantification_context = quantification_context_summary(plumes)
    source_observations = source_observation_summary(sources)
    portfolio_design = provisional_portfolio_design()
    summary = build_summary_metrics(
        plumes,
        sources,
        sensitivity,
        countries,
        drift,
        mapping_checks,
        fetched_at,
    )

    # Save machine-readable outputs.  Timestamp columns are deliberately kept
    # in ISO-like text by pandas so Excel and notebooks can both parse them.
    plume_csv = ANALYSIS_DIR / "plume_detail_audit.csv"
    source_csv = ANALYSIS_DIR / "source_detail_audit_500m.csv"
    revisit_csv = ANALYSIS_DIR / "revisit_candidates.csv"
    missing_csv = ANALYSIS_DIR / "field_completeness.csv"
    endpoint_gap_csv = ANALYSIS_DIR / "public_endpoint_gaps.csv"
    country_csv = ANALYSIS_DIR / "country_coverage.csv"
    sector_csv = ANALYSIS_DIR / "sector_coverage.csv"
    monthly_csv = ANALYSIS_DIR / "monthly_coverage.csv"
    quantification_context_csv = ANALYSIS_DIR / "quantification_context.csv"
    source_observations_csv = ANALYSIS_DIR / "source_observation_summary.csv"
    portfolio_design_csv = ANALYSIS_DIR / "provisional_30_image_design.csv"
    sensitivity_csv = ANALYSIS_DIR / "source_cluster_sensitivity.csv"
    summary_csv = ANALYSIS_DIR / "audit_summary.csv"

    plumes.sort_values("scene_timestamp").to_csv(plume_csv, index=False)
    sources.to_csv(source_csv, index=False)
    sources[
        [
            "revisit_priority_tier",
            "cm_source_id",
            "source_name",
            "country",
            "sector",
            "sector_label",
            "source_latitude",
            "source_longitude",
            "cohort_detection_count",
            "cohort_detection_date_count",
            "cohort_quantified_count",
            "cohort_unquantified_count",
            "cohort_quantified_percent",
            "cohort_median_emission_kg_hr",
            "api_observation_date_count",
            "api_detection_date_count",
            "api_persistence",
            "cohort_first_detection",
            "cohort_last_detection",
        ]
    ].to_csv(revisit_csv, index=False)
    missingness.to_csv(missing_csv, index=False)
    endpoint_gaps.to_csv(endpoint_gap_csv, index=False)
    countries.to_csv(country_csv, index=False)
    sectors.to_csv(sector_csv, index=False)
    monthly.to_csv(monthly_csv, index=False)
    quantification_context.to_csv(quantification_context_csv, index=False)
    source_observations.to_csv(source_observations_csv, index=False)
    portfolio_design.to_csv(portfolio_design_csv, index=False)
    sensitivity.to_csv(sensitivity_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    summary_figure = FIGURE_DIR / "01_tanager_catalog_gap_audit.png"
    source_map = FIGURE_DIR / "01_tanager_source_gaps.png"
    save_summary_figure(plumes, sources, countries, monthly, summary_figure)
    save_source_map(sources, country_boundaries, source_map)

    report = build_report(
        plumes,
        sources,
        missingness,
        endpoint_gaps,
        countries,
        sectors,
        quantification_context,
        source_observations,
        portfolio_design,
        sensitivity,
        drift,
        mapping_checks,
        fetched_at,
        plume_snapshot_path,
        source_snapshot_path,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nAudit complete.")
    print(summary.to_string(index=False))
    print("\nMain outputs:")
    for path in [
        REPORT_PATH,
        summary_csv,
        plume_csv,
        source_csv,
        revisit_csv,
        summary_figure,
        source_map,
    ]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
