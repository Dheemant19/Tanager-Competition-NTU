"""Refresh only the public SEA sources approved for this analysis version.

Examples from the project root:

    python scripts\\refresh_sea_public_sources.py --sron
    python scripts\\refresh_sea_public_sources.py --emit-nasa

``--sron`` caches newly published weekly CSVs without touching Tanager or
Carbon Mapper data.  ``--emit-nasa`` refreshes the eight already accepted
NASA CH4 plume metadata records, correcting the exact official emission-rate
field name.  It requires the existing Earthdata credentials in ``.env``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "sea_satellite_analysis" / "raw_extracts"
BBOX = (92.0, -11.0, 141.0, 23.5)
SRON_PAGE = "https://www.sron.nl/en/pillars/science/earth/methane/methane-plume-maps/"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
EMIT_PLM = "C3242707413-LPCLOUD"  # EMITL2BCH4PLM V002


def safe_float(value) -> float:
    """Convert numeric metadata while preserving NASA's NA values as missing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def exact_or_containing(properties: dict, *names: str):
    """Prefer exact official fields, with a documented safe fallback."""
    for name in names:
        if name in properties:
            return properties[name]
    lower_names = [name.lower() for name in names]
    for key, value in properties.items():
        key_lower = key.lower()
        if any(name in key_lower for name in lower_names):
            return value
    return None


def polygon_centroid(granule: dict) -> tuple[float, float]:
    """Return the legacy granule centroid used by the current CSV schema."""
    coordinates = [float(value) for value in granule["polygons"][0][0].split()]
    latitudes = coordinates[0::2]
    longitudes = coordinates[1::2]
    return float(np.mean(latitudes)), float(np.mean(longitudes))


def refresh_sron_weekly_cache() -> list[Path]:
    """Download only new official weekly SRON CSVs into the existing raw cache."""
    session = requests.Session()
    session.headers["User-Agent"] = "SEA-methane-analysis/1.1 (research cache refresh)"
    response = session.get(SRON_PAGE, timeout=60)
    response.raise_for_status()
    urls = sorted(
        set(re.findall(r'https://www\\.sron\\.nl/wp-content/uploads/[^"\\s]+-\\.csv', response.text))
    )
    weekly_dir = RAW_DIR / "sron_weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    new_files: list[Path] = []
    for url in urls:
        path = weekly_dir / url.rsplit("/", 1)[1]
        if path.exists():
            continue
        file_response = session.get(url, timeout=60)
        file_response.raise_for_status()
        path.write_bytes(file_response.content)
        new_files.append(path)
        time.sleep(0.2)
    print(f"SRON archive files listed: {len(urls)}")
    print(f"New weekly CSVs cached: {len(new_files)}")
    for path in new_files:
        print(path.name)
    return new_files


def refresh_emit_nasa_metadata() -> pd.DataFrame:
    """Refresh the approved NASA EMIT SEA extract with exact rate-field names.

    This does not add a new processing pipeline or expand the spatial query;
    it re-downloads the eight official V002 plume-complex metadata JSON files.
    """
    from dotenv import load_dotenv
    import earthaccess

    load_dotenv(PROJECT_ROOT / ".env")
    earthaccess.login(strategy="environment")
    session = earthaccess.get_requests_https_session()
    bbox = ",".join(str(value) for value in BBOX)
    search = requests.get(
        CMR,
        params={"collection_concept_id": EMIT_PLM, "bounding_box": bbox, "page_size": 50},
        timeout=60,
    )
    search.raise_for_status()
    granules = search.json()["feed"]["entry"]
    rows: list[dict] = []
    for granule in granules:
        metadata_url = next(
            link["href"]
            for link in granule["links"]
            if link.get("href", "").startswith("https")
            and link.get("href", "").endswith(".json")
            and "PLMMETA" in link.get("href", "")
        )
        metadata = session.get(metadata_url, timeout=60)
        metadata.raise_for_status()
        payload = metadata.json()
        properties = payload["features"][0]["properties"] if "features" in payload else payload
        latitude, longitude = polygon_centroid(granule)
        rows.append(
            {
                "sensor": "EMIT (NASA)",
                "plume_id": granule["title"],
                "datetime": granule["time_start"],
                # The 11-column legacy schema has one location only.  Preserve
                # its legacy granule centroid and do not call it a source point.
                "lat": latitude,
                "lon": longitude,
                "source": "NASA LP DAAC EMITL2BCH4PLM.002 (CH4PLMMETA JSON)",
                # The exact plural key fixes the prior all-missing-rate bug.
                "emission_kg_hr": safe_float(
                    exact_or_containing(
                        properties,
                        "Emissions Rate Estimate (kg/hr)",
                        "emission rate estimate",
                    )
                ),
                "uncertainty_kg_hr": safe_float(
                    exact_or_containing(
                        properties,
                        "Emissions Rate Estimate Uncertainty (kg/hr)",
                        "Emissions Rate Uncertainty (kg/hr)",
                        "emissions rate estimate uncertainty",
                        "emission rate uncertainty",
                    )
                ),
                "max_conc_ppm_m": safe_float(
                    exact_or_containing(
                        properties,
                        "Max Plume Concentration (ppm-m)",
                        "max plume concentration",
                    )
                ),
                "sector": exact_or_containing(properties, "Sector") or "unknown",
            }
        )
        time.sleep(0.2)

    output = pd.DataFrame(rows)
    output["datetime"] = pd.to_datetime(output["datetime"], utc=True)
    output.to_csv(RAW_DIR / "emit_plumes_nasa_sea.csv", index=False)
    print(f"Official NASA EMIT V002 plume complexes written: {len(output)}")
    print(f"Rows with a public rate: {output.emission_kg_hr.notna().sum()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sron", action="store_true", help="cache newly posted SRON weekly CSV files")
    parser.add_argument("--emit-nasa", action="store_true", help="refresh the accepted NASA EMIT V002 metadata rows")
    arguments = parser.parse_args()
    if not arguments.sron and not arguments.emit_nasa:
        parser.error("choose --sron and/or --emit-nasa")
    if arguments.sron:
        refresh_sron_weekly_cache()
    if arguments.emit_nasa:
        refresh_emit_nasa_metadata()


if __name__ == "__main__":
    main()
