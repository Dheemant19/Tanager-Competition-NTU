"""Rebuild the SEA methane comparison from the accepted local raw extracts.

This module intentionally does not refresh the Carbon Mapper Tanager or EMIT
Carbon Mapper extracts.  Those inputs are frozen for this study version.

It does refresh the *derived* TROPOMI SEA extract from the locally cached SRON
weekly CSV files.  The cache is updated separately by
``refresh_sea_public_sources.py`` so notebook execution remains reproducible
and does not depend on live network services.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sea_satellite_analysis"
RAW_DIR = DATA_DIR / "raw_extracts"
ANALYSIS_DIR = DATA_DIR / "analysis"
FIG_DIR = PROJECT_ROOT / "figures"

# This is deliberately named the SEA STUDY FRAME, not ASEAN.  It contains
# ASEAN land plus neighbouring/offshore areas that lie in this rectangle.
BBOX = (92.0, -11.0, 141.0, 23.5)
WINDOW_START = pd.Timestamp("2024-09-01", tz="UTC")

SENSOR_ORDER = [
    "TROPOMI (SRON)",
    "EMIT (NASA)",
    "EMIT (CarbonMapper)",
    "GHGSat (open sample)",
    "Tanager",
]
COLORS = {
    "Tanager": "#d62728",
    "EMIT (NASA)": "#1f77b4",
    "EMIT (CarbonMapper)": "#17becf",
    "TROPOMI (SRON)": "#ff7f0e",
    "GHGSat (open sample)": "#2ca02c",
    "EMIT": "#1f77b4",
    "TROPOMI": "#ff7f0e",
    "GHGSat": "#2ca02c",
}


def ensure_directories() -> None:
    """Create output directories if the project is being used on a new clone."""
    for directory in (RAW_DIR, ANALYSIS_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def read_csv_datetime(path: Path) -> pd.DataFrame:
    """Read a raw/derived CSV and parse its common datetime field when present."""
    frame = pd.read_csv(path)
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, format="mixed")
    return frame


def build_tropomi_extract_from_cache() -> pd.DataFrame:
    """Build the standard SEA subset from every locally cached SRON weekly CSV.

    The output keeps the existing ``tropomi_plumes_sron_sea.csv`` column layout.
    It is a selected weekly plume catalogue, not a complete L2 XCH4 record.
    """
    weekly_dir = RAW_DIR / "sron_weekly"
    files = sorted(weekly_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No SRON weekly files found in {weekly_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, index_col=0)
        frame["_weekly_file"] = path.name
        frames.append(frame)

    sron = pd.concat(frames, ignore_index=True)
    sron["datetime"] = pd.to_datetime(
        sron["date"].astype(str) + " " + sron["time_UTC"].astype(str),
        utc=True,
        format="mixed",
    )
    # A record may be reissued in the live archive.  Keep a single copy using
    # its physical/time/rate fingerprint rather than its row position.
    duplicate_key = ["datetime", "lat", "lon", "source_rate_t/h", "uncertainty_t/h"]
    sron = sron.drop_duplicates(subset=duplicate_key).copy()

    in_frame = (
        sron["lon"].between(BBOX[0], BBOX[2])
        & sron["lat"].between(BBOX[1], BBOX[3])
    )
    tropomi = sron.loc[in_frame].copy()
    tropomi["sensor"] = "TROPOMI (SRON)"
    tropomi["emission_kg_hr"] = tropomi["source_rate_t/h"] * 1000.0
    tropomi["uncertainty_kg_hr"] = tropomi["uncertainty_t/h"] * 1000.0
    tropomi["source"] = "SRON Weekly Methane Plumes CSV (Schuit et al. 2023 method)"
    tropomi = tropomi.sort_values(["datetime", "lat", "lon"]).reset_index(drop=True)
    tropomi.drop(columns=["_weekly_file"], errors="ignore").to_csv(
        RAW_DIR / "tropomi_plumes_sron_sea.csv", index=False
    )
    return tropomi


def conform(frame: pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
    """Return one sensor extract in the original 11-column harmonised schema."""
    keep = [
        "sensor",
        "plume_id",
        "datetime",
        "lat",
        "lon",
        "emission_kg_hr",
        "uncertainty_kg_hr",
        "max_conc_ppm_m",
        "sector",
        "source",
    ]
    output = frame.copy()
    if id_column and "plume_id" not in output:
        output["plume_id"] = output[id_column].astype(str)
    for column in keep:
        if column not in output.columns:
            output[column] = np.nan
    return output[keep]


def build_census() -> pd.DataFrame:
    """Rebuild the standard raw SEA subset and harmonised plume table.

    Tanager, both EMIT extracts, and GHGSat are read exactly as accepted local
    inputs.  The TROPOMI subset is rebuilt from the local weekly cache so newly
    downloaded weeks are incorporated automatically.
    """
    ensure_directories()
    tanager = read_csv_datetime(RAW_DIR / "tanager_plumes_sea.csv")
    emit_nasa = read_csv_datetime(RAW_DIR / "emit_plumes_nasa_sea.csv")
    emit_cm = read_csv_datetime(RAW_DIR / "emit_plumes_carbonmapper_sea.csv")
    tropomi = build_tropomi_extract_from_cache()
    ghgsat = read_csv_datetime(RAW_DIR / "ghgsat_plumes_sea.csv")

    # Preserve the historical method for stable TROPOMI IDs.  The number is a
    # deterministic row ordinal after date/location sorting, not a source ID.
    tropomi_for_table = tropomi.copy()
    tropomi_for_table["plume_id"] = (
        "SRON_"
        + tropomi_for_table["datetime"].dt.strftime("%Y%m%dT%H%M%S")
        + "_"
        + tropomi_for_table.index.astype(str)
    )
    ghgsat_for_table = ghgsat.copy()
    if "plume_id" not in ghgsat_for_table.columns:
        ghgsat_for_table["plume_id"] = (
            "GHGSat_site"
            + ghgsat_for_table["site_ID"].astype(str)
            + "_"
            + ghgsat_for_table["datetime"].dt.strftime("%Y%m%d")
        )

    table = pd.concat(
        [
            conform(tanager),
            conform(emit_nasa),
            conform(emit_cm),
            conform(tropomi_for_table),
            conform(ghgsat_for_table),
        ],
        ignore_index=True,
    )
    table["datetime"] = pd.to_datetime(table["datetime"], utc=True, format="mixed")
    table["in_common_window"] = table["datetime"] >= WINDOW_START
    table.to_csv(ANALYSIS_DIR / "sea_plume_table.csv", index=False)
    return table


def pct_ladder(values: pd.Series) -> pd.Series:
    """Return the pre-existing rate-distribution CSV columns for one sensor."""
    values = values.dropna()
    if values.empty:
        return pd.Series({"n_quantified": 0})
    quantiles = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return pd.Series(
        {
            "n_quantified": len(values),
            "min": quantiles[0],
            "p5": quantiles[1],
            "p25": quantiles[2],
            "median": quantiles[3],
            "p75": quantiles[4],
            "p95": quantiles[5],
            "max": quantiles[6],
        }
    )


def sensor_family(sensor: str) -> str:
    """Map processing catalogues to their physical instrument family."""
    if sensor.startswith("EMIT"):
        return "EMIT"
    if sensor.startswith("TROPOMI"):
        return "TROPOMI"
    if sensor.startswith("GHGSat"):
        return "GHGSat"
    return sensor


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres; accepts floats or NumPy arrays."""
    radius_km = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    delta_p = np.radians(np.asarray(lat2) - np.asarray(lat1))
    delta_l = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(delta_p / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(delta_l / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def time_bin(hours: float) -> str:
    """Use the original time-gap bins, with labels appropriate for candidate events."""
    absolute_hours = abs(hours)
    if absolute_hours < 3:
        return "<3h (same overpass)"
    if absolute_hours < 24:
        return "<24h (same day)"
    if absolute_hours < 24 * 7:
        return "<7d"
    if absolute_hours < 24 * 30:
        return "<30d"
    return ">=30d"


def _event_priority(row: pd.Series) -> tuple[int, int]:
    """Prefer a public rate, then NASA confirmation, when EMIT records duplicate."""
    has_rate = int(pd.notna(row["emission_kg_hr"]))
    is_nasa = int(row["sensor"] == "EMIT (NASA)")
    return has_rate, is_nasa


def _canonical_emit_events(table: pd.DataFrame) -> pd.DataFrame:
    """Merge same-overpass NASA/Carbon Mapper EMIT records for comparison only.

    The original harmonised CSV intentionally keeps both processing catalogues.
    This temporary view prevents them being treated as independent instruments
    or doubling a Tanager/EMIT comparison.
    """
    emit = table[table["sensor"].str.startswith("EMIT")].copy().reset_index(drop=True)
    if emit.empty:
        return emit

    parent = list(range(len(emit)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # 10 km and two hours follows the audit's conservative rough-matching
    # tolerance for the same physical EMIT observation.
    for left in range(len(emit)):
        for right in range(left + 1, len(emit)):
            if emit.loc[left, "sensor"] == emit.loc[right, "sensor"]:
                continue
            distance = haversine_km(
                emit.loc[left, "lat"],
                emit.loc[left, "lon"],
                emit.loc[right, "lat"],
                emit.loc[right, "lon"],
            )
            hours = abs((emit.loc[right, "datetime"] - emit.loc[left, "datetime"]).total_seconds() / 3600)
            if distance <= 10 and hours <= 2:
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(emit)):
        groups[find(index)].append(index)

    canonical: list[pd.Series] = []
    for indices in groups.values():
        candidates = emit.loc[indices].copy()
        candidates["_priority"] = candidates.apply(_event_priority, axis=1)
        chosen = candidates.sort_values("_priority", ascending=False).iloc[0].drop(labels="_priority")
        chosen["sensor"] = "EMIT"
        canonical.append(chosen)
    return pd.DataFrame(canonical).reset_index(drop=True)


def _comparison_events(table: pd.DataFrame) -> pd.DataFrame:
    """Return a physical-sensor view used only for overlap/rate comparisons."""
    non_emit = table[~table["sensor"].str.startswith("EMIT")].copy()
    non_emit["sensor"] = non_emit["sensor"].map(sensor_family)
    emit = _canonical_emit_events(table)
    return pd.concat([non_emit, emit], ignore_index=True).dropna(subset=["lat", "lon"])


def build_one_to_one_overlaps(table: pd.DataFrame) -> pd.DataFrame:
    """Match unique candidate events without the old all-pairs inflation.

    Within each physical-sensor pair, feasible candidates are sorted by smallest
    absolute time gap then distance.  A greedy one-to-one assignment ensures an
    individual detection is used at most once for that sensor pair.
    """
    events = _comparison_events(table).reset_index(drop=True)
    sensors = sorted(events["sensor"].unique())
    records: list[dict] = []

    for left_pos, left_sensor in enumerate(sensors):
        for right_sensor in sensors[left_pos + 1 :]:
            left = events[events["sensor"] == left_sensor].reset_index(drop=True)
            right = events[events["sensor"] == right_sensor].reset_index(drop=True)
            radius = 10.0 if "TROPOMI" in {left_sensor, right_sensor} else 5.0
            candidates: list[tuple[float, float, int, int, float]] = []
            for left_index, left_row in left.iterrows():
                distances = haversine_km(
                    left_row["lat"], left_row["lon"], right["lat"].to_numpy(), right["lon"].to_numpy()
                )
                for right_index, distance in enumerate(distances):
                    if distance > radius:
                        continue
                    hours = (right.loc[right_index, "datetime"] - left_row["datetime"]).total_seconds() / 3600
                    candidates.append((abs(hours), float(distance), left_index, right_index, hours))

            used_left: set[int] = set()
            used_right: set[int] = set()
            for _, distance, left_index, right_index, hours in sorted(candidates):
                if left_index in used_left or right_index in used_right:
                    continue
                used_left.add(left_index)
                used_right.add(right_index)
                left_row, right_row = left.loc[left_index], right.loc[right_index]
                ratio = np.nan
                if pd.notna(left_row["emission_kg_hr"]) and pd.notna(right_row["emission_kg_hr"]):
                    if right_row["emission_kg_hr"] > 0:
                        ratio = left_row["emission_kg_hr"] / right_row["emission_kg_hr"]
                records.append(
                    {
                        "sensor_a": left_sensor,
                        "sensor_b": right_sensor,
                        "plume_a": left_row["plume_id"],
                        "plume_b": right_row["plume_id"],
                        "dist_km": round(distance, 2),
                        "dt_hours": round(hours, 1),
                        "dt_bin": time_bin(hours),
                        "rate_a_kg_hr": left_row["emission_kg_hr"],
                        "rate_b_kg_hr": right_row["emission_kg_hr"],
                        "rate_ratio_a_over_b": ratio,
                        "lat": left_row["lat"],
                        "lon": left_row["lon"],
                    }
                )

    columns = [
        "sensor_a", "sensor_b", "plume_a", "plume_b", "dist_km", "dt_hours", "dt_bin",
        "rate_a_kg_hr", "rate_b_kg_hr", "rate_ratio_a_over_b", "lat", "lon",
    ]
    overlaps = pd.DataFrame(records, columns=columns)
    if not overlaps.empty:
        overlaps = overlaps.sort_values("dt_hours", key=lambda series: series.abs()).reset_index(drop=True)
    overlaps.to_csv(ANALYSIS_DIR / "cross_sensor_overlaps.csv", index=False)

    return overlaps


def cluster_tanager_sources(tanager: pd.DataFrame, radius_km: float = 0.5) -> pd.DataFrame:
    """Return the approved 500 m source groups, with a robust local fallback.

    The Carbon Mapper gap audit already contains the reviewed 385-plume to
    93-source mapping.  Reusing it keeps this visualisation exactly aligned
    with the audit instead of silently creating a different clustering rule.
    """
    tanager = tanager.dropna(subset=["lat", "lon"]).reset_index(drop=True).copy()
    audit_mapping_path = (
        ANALYSIS_DIR
        / "carbon_mapper_tanager_audit"
        / "plume_detail_audit.csv"
    )
    if audit_mapping_path.exists() and "plume_id" in tanager.columns:
        audit_mapping = pd.read_csv(audit_mapping_path, usecols=["plume_id", "cm_source_id"])
        tanager = tanager.merge(audit_mapping, on="plume_id", how="left", validate="many_to_one")
        if tanager["cm_source_id"].notna().all():
            tanager["source_cluster"] = tanager["cm_source_id"]
            return tanager.drop(columns="cm_source_id")
        # The fallback is only for future extracts that include genuinely new
        # plumes absent from the frozen audit mapping.
        tanager = tanager.drop(columns="cm_source_id")

    # Fallback: connected components, avoiding the old order-dependent pass.
    neighbours: list[set[int]] = [set() for _ in range(len(tanager))]
    for left in range(len(tanager)):
        distances = haversine_km(
            tanager.loc[left, "lat"],
            tanager.loc[left, "lon"],
            tanager["lat"].to_numpy(),
            tanager["lon"].to_numpy(),
        )
        for right in np.flatnonzero(distances <= radius_km):
            neighbours[left].add(int(right))

    cluster_id = np.full(len(tanager), -1, dtype=int)
    next_cluster = 0
    for start in range(len(tanager)):
        if cluster_id[start] >= 0:
            continue
        stack = [start]
        cluster_id[start] = next_cluster
        while stack:
            current = stack.pop()
            for neighbour in neighbours[current]:
                if cluster_id[neighbour] < 0:
                    cluster_id[neighbour] = next_cluster
                    stack.append(neighbour)
        next_cluster += 1
    tanager["source_cluster"] = cluster_id
    return tanager


def build_corroboration(table: pd.DataFrame) -> pd.DataFrame:
    """Build the existing corroboration CSV using 500 m source clusters."""
    tanager = cluster_tanager_sources(table[table["sensor"] == "Tanager"], radius_km=0.5)
    others = _comparison_events(table)
    others = others[others["sensor"] != "Tanager"]
    rows: list[dict] = []
    for cluster, group in tanager.groupby("source_cluster"):
        lat, lon = group["lat"].mean(), group["lon"].mean()
        seen_families: set[str] = set()
        for sensor, detections in others.groupby("sensor"):
            radius = 10.0 if sensor == "TROPOMI" else 5.0
            distance = haversine_km(lat, lon, detections["lat"].to_numpy(), detections["lon"].to_numpy())
            if (distance <= radius).any():
                seen_families.add(sensor)
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "revisits": len(group),
                "n_other_sensors": len(seen_families),
                "median_rate": group["emission_kg_hr"].median(),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(ANALYSIS_DIR / "tanager_sources_corroboration.csv", index=False)
    return output


def build_analysis_tables(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regenerate all existing derived CSVs plus the close-time rate-comparison CSV."""
    distribution = table.groupby("sensor")["emission_kg_hr"].apply(pct_ladder).unstack()
    # Preserve the exact pre-existing CSV field order for downstream notebooks
    # and slide-table code that may already read it by position.
    distribution = distribution.reindex(
        columns=["max", "median", "min", "n_quantified", "p25", "p5", "p75", "p95"]
    )
    distribution.to_csv(ANALYSIS_DIR / "emission_rate_distributions.csv")
    overlaps = build_one_to_one_overlaps(table)
    corroboration = build_corroboration(table)
    return distribution, overlaps, corroboration


def _add_study_frame(ax) -> None:
    """Draw the rectangle explicitly so it cannot be mistaken for ASEAN borders."""
    x0, y0, x1, y1 = BBOX
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#4A5568", lw=0.9, ls="--")


def make_figure_01(table: pd.DataFrame) -> None:
    """Figure 01: descriptive catalogue-rate distributions, not sensitivity thresholds."""
    sensors = [sensor for sensor in SENSOR_ORDER if table.loc[table.sensor == sensor, "emission_kg_hr"].notna().any()]
    fig, ax = plt.subplots(figsize=(9.5, 0.9 * len(sensors) + 2.1))
    bins = np.logspace(1.5, 5.5, 40)
    np.random.seed(0)
    for index, sensor in enumerate(sensors):
        rates = table.loc[table.sensor == sensor, "emission_kg_hr"].dropna()
        jitter = np.random.uniform(-0.18, 0.18, len(rates))
        ax.scatter(rates, np.full(len(rates), index) + jitter, s=14, alpha=0.55, color=COLORS[sensor])
        ax.scatter([rates.median()], [index], marker="|", s=600, color="black", zorder=5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(sensors)))
    ax.set_yticklabels(sensors)
    ax.set_xlabel("Public emission-rate estimate (kg CH₄ h⁻¹, log scale) · black tick = median")
    ax.set_title("SEA study frame: public catalogue emission-rate distributions\n"
                 "Descriptive samples, not intrinsic sensor detection thresholds")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_rate_distributions.png", dpi=180)
    plt.close(fig)


def make_figure_02(table: pd.DataFrame) -> None:
    """Figure 02: availability and map; removes incomparable EMIT/TROPOMI counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    starts, ends, labels = [], [], []
    for sensor in SENSOR_ORDER:
        records = table[table.sensor == sensor]
        if records.empty:
            continue
        starts.append(records.datetime.min())
        ends.append(records.datetime.max())
        labels.append(f"{sensor}  (n={len(records)})")
    for index, (start, end, label) in enumerate(zip(starts, ends, labels)):
        ax.hlines(index, start, end, color=COLORS[label.split("  ")[0]], lw=7, alpha=0.8)
        ax.scatter([start, end], [index, index], color=COLORS[label.split("  ")[0]], s=28, zorder=3)
    earliest_catalogue_date = min(starts)
    if earliest_catalogue_date < WINDOW_START:
        ax.axvspan(earliest_catalogue_date, WINDOW_START, color="#718096", alpha=0.10)
    ax.axvline(WINDOW_START, color="#2D3748", lw=1, ls="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Temporal availability of the public plume catalogues\n"
                 "Lines show catalogue date spans - not sensor coverage")

    ax = axes[1]
    for sensor, group in table.dropna(subset=["lat", "lon"]).groupby("sensor"):
        ax.scatter(group.lon, group.lat, s=12, alpha=0.6, label=f"{sensor} ({len(group)})", color=COLORS[sensor])
    _add_study_frame(ax)
    ax.set_xlim(90, 143)
    ax.set_ylim(-12, 25)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Detected plumes in the SEA study frame\n(rectangle includes non-ASEAN land and offshore areas)")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_coverage_map.png", dpi=180)
    plt.close(fig)


def make_figure_03(table: pd.DataFrame) -> None:
    """Figure 03: the public-catalogue timeline."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for index, sensor in enumerate(SENSOR_ORDER):
        group = table[table.sensor == sensor]
        rated = group[group.emission_kg_hr.notna()]
        unrated = group[group.emission_kg_hr.isna()]
        ax.scatter(
            rated.datetime,
            [index] * len(rated),
            s=8 + 25 * (np.log10(rated.emission_kg_hr.clip(lower=50)) - 1.5),
            color=COLORS[sensor],
            alpha=0.6,
            zorder=3,
        )
        ax.scatter(unrated.datetime, [index] * len(unrated), s=18, facecolors="none", edgecolors=COLORS[sensor], marker="s", alpha=0.7, zorder=2)
    ax.axvspan(table.datetime.min(), WINDOW_START, color="gray", alpha=0.12)
    ax.axvline(WINDOW_START, color="black", lw=1, ls="--")
    ax.text(WINDOW_START, len(SENSOR_ORDER) - 0.35, "  Tanager-era window to", fontsize=9)
    ax.set_yticks(range(len(SENSOR_ORDER)))
    ax.set_yticklabels(SENSOR_ORDER)
    ax.set_title("Public methane-plume catalogues over time in the SEA study frame\n"
                 "dot size = public rate; open square = no public rate")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_catalog_timeline.png", dpi=180)
    plt.close(fig)


def make_figure_04(overlaps: pd.DataFrame) -> None:
    """Figure 04: unique candidate matches and valid close-time rate comparisons."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    bins = ["<3h (same overpass)", "<24h (same day)", "<7d", "<30d", ">=30d"]
    if overlaps.empty:
        matrix = pd.DataFrame(columns=bins)
    else:
        pair_labels = overlaps.sensor_a + " × " + overlaps.sensor_b
        matrix = overlaps.assign(pair=pair_labels).groupby(["pair", "dt_bin"]).size().unstack(fill_value=0)
        matrix = matrix.reindex(columns=bins, fill_value=0)
    if matrix.empty:
        ax.text(0.5, 0.5, "No cross-sensor candidate events", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.imshow(np.log10(matrix.values + 1), cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.values[row, column]
                ax.text(column, row, str(value) if value else "–", ha="center", va="center", fontsize=9)
        ax.set_title("One-to-one candidate event matches\n"
                     "physical sensors; NASA/CM EMIT not treated as separate")

    ax = axes[1]
    close_rates = overlaps[
        (overlaps["dt_hours"].abs() < 24)
        & overlaps["rate_a_kg_hr"].notna()
        & overlaps["rate_b_kg_hr"].notna()
        & (overlaps["rate_a_kg_hr"] > 0)
        & (overlaps["rate_b_kg_hr"] > 0)
    ].copy()
    if close_rates.empty:
        ax.text(
            0.5,
            0.58,
            "No rate-comparable public pairs",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#2D3748",
        )
        ax.text(
            0.5,
            0.40,
            "Criterion: different physical sensors, ≤24 h apart,\n"
            "co-located and both rates public.",
            ha="center",
            va="center",
            fontsize=9,
            color="#4A5568",
        )
        ax.set_axis_off()
    else:
        for (sensor_a, sensor_b), group in close_rates.groupby(["sensor_a", "sensor_b"]):
            ax.scatter(group.rate_a_kg_hr, group.rate_b_kg_hr, label=f"{sensor_a} × {sensor_b}", alpha=0.8)
        min_rate = min(close_rates.rate_a_kg_hr.min(), close_rates.rate_b_kg_hr.min())
        max_rate = max(close_rates.rate_a_kg_hr.max(), close_rates.rate_b_kg_hr.max())
        ax.plot([min_rate, max_rate], [min_rate, max_rate], "k--", lw=1, label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sensor A rate (kg CH₄ h⁻¹)")
        ax.set_ylabel("sensor B rate (kg CH₄ h⁻¹)")
        ax.legend(fontsize=8)
        ax.set_title("Rate comparison for valid close-time pairs")
    figure.suptitle("Cross-sensor comparison in the SEA study frame", y=1.01, fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "01_overlap_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_figure_05(corroboration: pd.DataFrame) -> None:
    """Figure 05: 500 m Tanager source clusters, clearly labelled as catalogue corroboration."""
    data = corroboration.copy()
    data["corroborated"] = data["n_other_sensors"] > 0
    only_tanager = (~data["corroborated"]).sum()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for flag, color, label in [
        (False, "#d62728", f"Only Tanager in public catalogue (n={only_tanager})"),
        (True, "#1f77b4", f"Also recorded by ≥1 other physical sensor (n={data.corroborated.sum()})"),
    ]:
        group = data[data["corroborated"] == flag]
        ax.scatter(group.lon, group.lat, s=25 + 14 * group.revisits, color=color, label=label, edgecolors="black", linewidths=0.4, alpha=0.8)
    for revisits in (1, 10, 40):
        ax.scatter([], [], s=25 + 14 * revisits, color="gray", edgecolors="black", linewidths=0.4, alpha=0.6, label=f"dot size: {revisits} Tanager detections")
    _add_study_frame(ax)
    ax.set_xlim(90, 143)
    ax.set_ylim(-12, 25)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Tanager source clusters in the SEA study frame (500 m grouping)\n"
        f"{only_tanager}/{len(data)} have no other public catalogue record - not a same-day validation"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_corroboration_map.png", dpi=180)
    plt.close(fig)


def make_figure_06(table: pd.DataFrame) -> None:
    """Figure 06: relabelled descriptive rate-gap view, not a sensor-performance claim."""
    window = table[table.in_common_window & table.emission_kg_hr.notna()].copy()
    tropomi_rates = window.loc[window.sensor == "TROPOMI (SRON)", "emission_kg_hr"]
    if window.empty or tropomi_rates.empty:
        return
    tropomi_floor = tropomi_rates.min()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    np.random.seed(1)
    for sensor in ["Tanager", "EMIT (CarbonMapper)", "TROPOMI (SRON)"]:
        rates = window.loc[window.sensor == sensor, "emission_kg_hr"]
        if rates.empty:
            continue
        ax.scatter(rates, np.random.uniform(-0.35, 0.35, len(rates)), s=18, alpha=0.6, color=COLORS[sensor], label=f"{sensor} (n={len(rates)})")
    tanager_rates = window.loc[window.sensor == "Tanager", "emission_kg_hr"]
    lower = window.emission_kg_hr.min()
    ax.axvspan(lower, tropomi_floor, color="gold", alpha=0.15)
    ax.axvline(tropomi_floor, color="black", ls="--", lw=1.2)
    fraction_below = (tanager_rates < tropomi_floor).mean() if not tanager_rates.empty else np.nan
    ax.text(tropomi_floor * 1.15, -0.30, f"Smallest public SRON weekly-catalogue rate\n({tropomi_floor:,.0f} kg h⁻¹)", fontsize=9)
    ax.text(
        np.sqrt(lower * tropomi_floor),
        0.44,
        f"{fraction_below:.0%} of quantified Tanager records fall below this catalogue minimum",
        fontsize=10,
        ha="center",
        style="italic",
    )
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel("public emission-rate estimate (kg CH₄ h⁻¹, log scale) · Tanager-era window")
    ax.set_title("Catalogue rate ranges in the SEA study frame\n"
                 "Descriptive public-record gap, not a sensor detection-limit comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_public_rate_floor.png", dpi=180)
    plt.close(fig)


def make_figure_07(table: pd.DataFrame) -> None:
    """Figure 07: original sector summary with clearer geographical wording."""
    tanager = table[table.sensor == "Tanager"].copy()
    sector_names = {
        "1B2": "oil & gas",
        "1B1": "coal mining",
        "1B1a": "coal mining",
        "6A": "landfill",
        "6B": "wastewater",
        "4B": "manure/agri",
        "1A1": "energy industry",
        "other": "other/unclassified",
        "NA": "unclassified",
    }
    tanager["sector_name"] = tanager.sector.fillna("NA").map(lambda code: sector_names.get(code, code))
    summary = tanager.groupby("sector_name").agg(n=("plume_id", "count"), median_rate=("emission_kg_hr", "median")).sort_values("n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(summary.index, summary.n, color="#d62728", alpha=0.8)
    axes[0].set_title("Tanager detections by sector")
    axes[0].set_xlabel("catalogue detections")
    axes[1].barh(summary.index, summary.median_rate, color="#7f7f7f", alpha=0.8)
    axes[1].set_title("Median public emission-rate estimate by sector")
    axes[1].set_xlabel("kg CH₄ h⁻¹")
    for axis in axes:
        axis.grid(axis="x", alpha=0.3)
    fig.suptitle("Tanager methane records in the SEA study frame", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_tanager_sectors.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_interactive_map(table: pd.DataFrame) -> None:
    """Rebuild the HTML map with an explicit study-frame caption and rectangle."""
    map_object = folium.Map(location=[7, 113], zoom_start=5, tiles="cartodbpositron")
    caption = (
        '<div style="position: fixed; top: 12px; left: 58px; z-index: 1000; '
        'background: rgba(255,255,255,0.93); padding: 7px 10px; border: 1px solid #718096; '
        'font-size: 13px; font-weight: 600;">SEA study frame - rectangle includes non-ASEAN areas</div>'
    )
    map_object.get_root().html.add_child(folium.Element(caption))
    folium.Rectangle([[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]], color="#4A5568", weight=1, dash_array="5,5", fill=False, tooltip="SEA study frame (not ASEAN-only)").add_to(map_object)
    for sensor, group in table.dropna(subset=["lat", "lon"]).groupby("sensor"):
        feature_group = folium.FeatureGroup(name=f"{sensor} ({len(group)})")
        for _, row in group.iterrows():
            rate_text = f"{row.emission_kg_hr:,.0f} kg/h" if pd.notna(row.emission_kg_hr) else "not public / not quantified"
            radius = 4 + 3 * np.log10(max(row.emission_kg_hr, 100)) - 6 if pd.notna(row.emission_kg_hr) else 4
            folium.CircleMarker(
                [row.lat, row.lon],
                radius=max(radius, 3),
                color=COLORS[sensor],
                fill=True,
                fill_opacity=0.6,
                weight=1,
                popup=folium.Popup(
                    f"<b>{sensor}</b><br>{row.plume_id}<br>{row.datetime.date()}<br>"
                    f"rate: {rate_text}<br>sector: {row.sector}",
                    max_width=260,
                ),
            ).add_to(feature_group)
        feature_group.add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(FIG_DIR / "01_sea_methane_map.html")


def rebuild_everything() -> dict[str, int]:
    """Run the whole local rebuild and return a compact result summary."""
    table = build_census()
    distribution, overlaps, corroboration = build_analysis_tables(table)
    make_figure_01(table)
    make_figure_02(table)
    make_figure_03(table)
    make_figure_04(overlaps)
    make_figure_05(corroboration)
    make_figure_06(table)
    make_figure_07(table)
    make_interactive_map(table)
    close_rate_count = len(overlaps[
        (overlaps["dt_hours"].abs() < 24)
        & overlaps["rate_a_kg_hr"].notna()
        & overlaps["rate_b_kg_hr"].notna()
        & (overlaps["rate_a_kg_hr"] > 0)
        & (overlaps["rate_b_kg_hr"] > 0)
    ])
    return {
        "plume_rows": len(table),
        "sensors": table.sensor.nunique(),
        "overlap_events": len(overlaps),
        "rate_comparable_pairs": close_rate_count,
        "tanager_source_clusters": len(corroboration),
        "distribution_rows": len(distribution),
    }


if __name__ == "__main__":
    print(rebuild_everything())
