"""
scripts/inventory.py

Build a CSV and Parquet inventory of the Planet Tanager STAC catalog.

The crawl follows catalog, collection, and item links, then records each
scene's metadata and asset URLs.

Usage:
    python scripts/inventory.py
"""

import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import pandas as pd
from tqdm import tqdm


# Settings

# Tanager catalog root.
ROOT_URL = "https://www.planet.com/data/stac/tanager-core-imagery/catalog.json"

# Identify the client to the catalog service.
HEADERS = {
    "User-Agent": (
        "tanager-comp-inventory/0.1 "
        "(research project; contact akshatcrypts2022@gmail.com)"
    )
}

# Pause between requests to avoid overloading the service.
SLEEP_SECONDS = 0.15

# Retries per request.
MAX_RETRIES = 3

# Record every listed asset as an `asset_<name>` column.

# Resolve paths relative to the repository.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "inventory"


# Helpers

def fetch_json(session, url):
    """Fetch JSON with retries and a pause after successful requests."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()      # turn HTTP errors (404, 403...) into exceptions
            time.sleep(SLEEP_SECONDS)        # be polite to Planet's server
            return response.json()
        except requests.RequestException as error:
            # Increase the delay after each failed request.
            wait = 0.5 * attempt
            print(f"    ! request failed ({error}) - retry {attempt}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
    print(f"    ! GIVING UP on {url}")
    return None


def links_with_rel(stac_doc, rel, base_url):
    """Return absolute STAC link URLs matching `rel`."""
    urls = []
    for link in stac_doc.get("links", []):
        if link.get("rel") == rel and link.get("href"):
            urls.append(urljoin(base_url, link["href"]))
    return urls


def parse_item(item, collection_id):
    """Flatten one STAC item into an inventory row."""
    props = item.get("properties", {})
    bbox = item.get("bbox")  # [west, south, east, north] in degrees, if present

    # Bounding-box centre for mapping.
    centroid_lon = centroid_lat = None
    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        centroid_lon = (west + east) / 2
        centroid_lat = (south + north) / 2

    row = {
        "collection": collection_id,
        "item_id": item.get("id"),
        "datetime": props.get("datetime"),
        # Planet-specific quality fields.
        "cloud_percent": props.get("cloud_percent"),
        "light_haze_percent": props.get("light_haze_percent"),
        "quality_category": props.get("quality_category"),
        "collection_mode": props.get("collection_mode"),
        "location_description": props.get("location_description"),
        # Standard STAC view-extension fields.
        "sun_elevation": props.get("view:sun_elevation"),
        "sun_azimuth": props.get("view:sun_azimuth"),
        "view_azimuth": props.get("view:azimuth"),
        "off_nadir": props.get("view:off_nadir"),
        "gsd": props.get("gsd"),
        "bbox": str(bbox),
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
    }

    # Store each asset URL in its own column.
    assets = item.get("assets", {})
    for key, asset in assets.items():
        row[f"asset_{key}"] = asset.get("href")

    return row


# Catalog crawl

def main():
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Catalog root : {ROOT_URL}\n")

    # Reuse one session with the catalog header.
    session = requests.Session()
    session.headers.update(HEADERS)

    print("Fetching root catalog...")
    root = fetch_json(session, ROOT_URL)
    if root is None:
        print("Could not fetch the root catalog. Check your internet connection.")
        sys.exit(1)

    collection_urls = links_with_rel(root, "child", ROOT_URL)
    print(f"Found {len(collection_urls)} collections.\n")

    all_rows = []

    # Visit each collection and item.
    for collection_url in collection_urls:
        collection = fetch_json(session, collection_url)
        if collection is None:
            continue

        collection_id = collection.get("id", "unknown")
        item_urls = links_with_rel(collection, "item", collection_url)
        print(f"[{collection_id}] {len(item_urls)} scenes")

        for item_url in tqdm(item_urls, desc=f"  {collection_id}", unit="scene"):
            item = fetch_json(session, item_url)
            if item is None:
                continue
            all_rows.append(parse_item(item, collection_id))

    if not all_rows:
        print("\nNo scenes found - nothing to save. Something went wrong upstream.")
        sys.exit(1)

    # Save the inventory.
    df = pd.DataFrame(all_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "tanager_inventory.csv"
    parquet_path = OUTPUT_DIR / "tanager_inventory.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    # Print a collection summary.
    print("\n" + "=" * 50)
    print(f"DONE. {len(df)} scenes saved.")
    print(f"  CSV    : {csv_path}")
    print(f"  Parquet: {parquet_path}")
    print("=" * 50)
    print("\nScenes per collection:")
    counts = df["collection"].value_counts().sort_index()
    for name, count in counts.items():
        print(f"  {name:<24} {count}")
    print(f"\n  {'TOTAL':<24} {len(df)}")


if __name__ == "__main__":
    main()
