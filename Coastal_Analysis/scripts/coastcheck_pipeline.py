"""Run the CoastCheck Sangatta processing and triage workflow.

The command reads Tanager surface reflectance, cross-checks turbidity against
same-day Sentinel-2 ACOLITE data, evaluates shoreline monitoring zones, and
exports figures and spatial outputs.

Usage:
    python scripts/coastcheck_pipeline.py [--output-dir ./outputs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import rankdata, spearmanr


# Plotting palette
INK = "#17212B"
BLUE = "#1769AA"
BLUE_DARK = "#0B3C5D"
BLUE_LIGHT = "#A9D6E5"
ORANGE = "#E67E22"
ORANGE_LIGHT = "#F5B35C"
GOLD = "#C59B27"
GREEN = "#2E7D32"
GREY = "#5A6B7C"
SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "signal", [BLUE_DARK, BLUE, "#D6EAF8", "#FAD7A0", ORANGE, "#900C3F"], N=256
)

# Dogliotti et al. (2015) coefficients
A_RED, C_RED = 228.1, 0.1641
A_NIR, C_NIR = 3078.9, 0.2112
FILL_VALUE = -9999.0


def true_colour_rgba(visual: np.ndarray) -> np.ndarray:
    """Normalize a uint8 RGB image to float RGBA."""
    norm = np.clip(visual.astype("float32") / 255.0, 0.0, 1.0)
    alpha = np.ones((norm.shape[0], norm.shape[1], 1), dtype="float32")
    return np.concatenate([norm, alpha], axis=-1)


def extent_for(transform: rasterio.Affine, shape: tuple[int, int]) -> list[float]:
    """Return [left, right, bottom, top] extent for imshow."""
    left, bottom, right, top = array_bounds(shape[0], shape[1], transform)
    return [left, right, bottom, top]


def clean_axis(axis: plt.Axes) -> None:
    """Apply minimal clean styling to a matplotlib axis."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#C9D4DC")
    axis.spines[["left", "bottom"]].set_linewidth(1.1)
    axis.tick_params(colors=INK, labelsize=11)
    axis.grid(True, linestyle=(0, (2, 3)), linewidth=0.9, color="#E1E8ED", alpha=0.9)


def add_map_furniture(
    axis: plt.Axes,
    length_m: float = 2000.0,
    color: str = "white",
    side: str = "right",
) -> None:
    """Draw a minimalist scale bar and north arrow."""
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    dx = x_max - x_min
    dy = y_max - y_min

    bar_x = (x_max - 0.06 * dx - length_m) if side == "right" else (x_min + 0.06 * dx)
    bar_y = y_min + 0.06 * dy

    axis.plot(
        [bar_x, bar_x + length_m],
        [bar_y, bar_y],
        color=color,
        linewidth=3.2,
        solid_capstyle="butt",
        zorder=20,
    )
    for cap_x in [bar_x, bar_x + length_m]:
        axis.plot(
            [cap_x, cap_x],
            [bar_y - 45, bar_y + 45],
            color=color,
            linewidth=2.0,
            solid_capstyle="butt",
            zorder=20,
        )
    axis.text(
        bar_x + length_m / 2,
        bar_y + 130,
        f"{length_m / 1000:g} km",
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=color,
        zorder=20,
    )


def turbidity_dogliotti(reflectance: np.ndarray, a_val: float, c_val: float) -> np.ndarray:
    """Compute single-band Nechad/Dogliotti turbidity in FNU."""
    bounded = np.clip(reflectance, 0, 0.9 * c_val)
    return a_val * bounded / (1.0 - bounded / c_val)


def run_pipeline(
    sr_path: Path,
    visual_path: Path,
    s2_tur_path: Path,
    gmw_path: Path,
    mouth_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> dict:
    """Execute the complete CoastCheck Sangatta workflow."""

    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Processing Tanager-1 data: {sr_path.name}")
    with rasterio.open(visual_path) as src:
        target_transform = src.transform
        target_crs = src.crs
        target_shape = src.shape
        visual = np.moveaxis(src.read([1, 2, 3]), 0, -1)

    df_path = "HDFEOS/GRIDS/HYP/Data Fields"
    with h5py.File(sr_path, "r") as h5:
        sr_ds = h5[f"{df_path}/surface_reflectance"]
        wavelengths = np.asarray(sr_ds.attrs["wavelengths"], float)
        good = np.asarray(sr_ds.attrs["good_wavelengths"]).astype(bool)

        def band_window(low: float, high: float) -> np.ndarray:
            return np.where((wavelengths >= low) & (wavelengths <= high) & good)[0]

        def band_mean(indices: np.ndarray) -> np.ndarray:
            vals = sr_ds[indices].astype("float32")
            vals[vals == FILL_VALUE] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmean(vals, axis=0)

        red = band_mean(band_window(620, 670))
        nir = band_mean(band_window(841, 876))
        swir = band_mean(band_window(1600, 1620))
        green = band_mean(band_window(558, 562))
        nodata = h5[f"{df_path}/nodata_pixels"][:]
        cloud = h5[f"{df_path}/beta_cloud_mask"][:]
        cirrus = h5[f"{df_path}/beta_cirrus_mask"][:]

    # Water mask
    ndwi = (green - nir) / (green + nir)
    water_mask = (
        (nodata == 0) & (cloud == 0) & (cirrus == 0)
        & np.isfinite(red) & (ndwi > 0)
    )

    # SWIR correction and Dogliotti switching
    red_corr = np.clip(red - swir, 0, None)
    nir_corr = np.clip(nir - swir, 0, None)
    nir_weight = np.clip((red_corr - 0.05) / 0.02, 0, 1)

    saturated = water_mask & (
        (red_corr > 0.9 * C_RED) | (nir_corr > 0.9 * C_NIR)
    )
    tanager_tur = np.where(
        water_mask & ~saturated,
        (1.0 - nir_weight) * turbidity_dogliotti(red_corr, A_RED, C_RED)
        + nir_weight * turbidity_dogliotti(nir_corr, A_NIR, C_NIR),
        np.nan,
    )
    valid_tanager = int(np.isfinite(tanager_tur).sum())
    print(f"    - Valid Tanager water pixels: {valid_tanager:,}")

    # Reproject Sentinel-2 ACOLITE turbidity.
    print(f"[*] Reprojecting Sentinel-2 ACOLITE product: {s2_tur_path.name}")
    with rasterio.open(s2_tur_path) as s2_src:
        s2_native = s2_src.read(1).astype("float32")
        s2_native[s2_native > 1e6] = np.nan
        sentinel_tur = np.full(target_shape, np.nan, dtype="float32")
        reproject(
            s2_native,
            sentinel_tur,
            src_transform=s2_src.transform,
            src_crs=s2_src.crs,
            src_nodata=np.nan,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    common_water = np.isfinite(tanager_tur) & np.isfinite(sentinel_tur)
    tan_vals = tanager_tur[common_water]
    sen_vals = sentinel_tur[common_water]
    rank_rho = float(spearmanr(tan_vals, sen_vals).statistic)
    ratio = np.divide(sen_vals, tan_vals, out=np.full_like(sen_vals, np.nan), where=tan_vals > 0)
    within_2x = float(100 * np.nanmean((ratio >= 0.5) & (ratio <= 2.0)))
    tan_p95 = float(np.percentile(tan_vals, 95))
    sen_p95 = float(np.percentile(sen_vals, 95))

    print(f"    - Cross-sensor matched pixels: {int(common_water.sum()):,}")
    print(f"    - Spearman rho agreement: {rank_rho:.3f}")
    print(f"    - Agreement within factor of 2: {within_2x:.1f}%")
    print(f"    - Tanager p95 FNU: {tan_p95:.1f} | Sentinel-2 p95 FNU: {sen_p95:.1f}")

    same_day_df = pd.DataFrame([{
        "date": "2025-03-02",
        "minutes_between_sensors": 22,
        "matched_pixels": int(common_water.sum()),
        "spearman_rho": rank_rho,
        "within_factor_two_percent": within_2x,
        "tanager_p95_fnu_estimate": tan_p95,
        "sentinel_p95_fnu_estimate": sen_p95,
    }])
    same_day_df.to_csv(data_dir / "same_day_agreement.csv", index=False)

    # Segment the shoreline into 1 km sections.
    print(f"[*] Extracting shoreline monitoring sections using GMW: {gmw_path.name}")
    gmw = gpd.read_file(gmw_path).to_crs(target_crs)
    mouth = gpd.read_file(mouth_path).to_crs(target_crs)
    mouth_centre = mouth.geometry.union_all().centroid
    study_zone = mouth_centre.buffer(5000)
    mangroves = gmw[gmw.intersects(study_zone)].copy()

    mangrove_mask = rasterize(
        [(geom, 1) for geom in mangroves.geometry],
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    zone_mask = rasterize(
        [(study_zone, 1)],
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    edge_mask = mangrove_mask & zone_mask & ~binary_erosion(
        mangrove_mask, structure=np.ones((3, 3)), border_value=0
    )

    dist_px, nearest = distance_transform_edt(~common_water, return_indices=True)
    dist_m = dist_px * abs(target_transform.a)
    nearest_rows, nearest_cols = nearest
    edge_rows, edge_cols = np.where(edge_mask & (dist_m <= 300))
    edge_x, edge_y = rasterio.transform.xy(target_transform, edge_rows, edge_cols, offset="center")

    edge_df = pd.DataFrame({
        "x_utm": edge_x,
        "y_utm": edge_y,
        "distance_to_water_m": dist_m[edge_rows, edge_cols],
        "tanager_fnu_estimate": tanager_tur[nearest_rows[edge_rows, edge_cols], nearest_cols[edge_rows, edge_cols]],
        "sentinel_fnu_estimate": sentinel_tur[nearest_rows[edge_rows, edge_cols], nearest_cols[edge_rows, edge_cols]],
    })
    edge_df["tanager_relative_signal"] = 100 * edge_df["tanager_fnu_estimate"].rank(pct=True)
    edge_df["sentinel_relative_signal"] = 100 * edge_df["sentinel_fnu_estimate"].rank(pct=True)
    edge_df["combined_relative_signal"] = edge_df[["tanager_relative_signal", "sentinel_relative_signal"]].mean(axis=1)

    edge_df["along_coast_offset_km"] = (edge_df["y_utm"] - mouth_centre.y) / 1000.0
    edge_df["section"] = pd.cut(
        edge_df["along_coast_offset_km"],
        bins=[-5, -4, -3, -2, -1, 0, 1.2],
        labels=["S5", "S4", "S3", "S2", "S1", "N1"],
        include_lowest=True,
        right=False,
    )
    edge_df = edge_df.dropna(subset=["section"]).copy()

    sections = (
        edge_df.groupby("section", observed=True)
        .agg(
            edge_pixels=("section", "size"),
            x_label=("x_utm", "median"),
            y_label=("y_utm", "median"),
            median_distance_m=("distance_to_water_m", "median"),
            tanager_median_relative_signal=("tanager_relative_signal", "median"),
            sentinel_median_relative_signal=("sentinel_relative_signal", "median"),
            relative_signal_score=("combined_relative_signal", "median"),
        )
        .reset_index()
    )
    sections["monitoring_rank"] = sections["relative_signal_score"].rank(method="first", ascending=False).astype(int)
    sections["tanager_rank"] = sections["tanager_median_relative_signal"].rank(method="first", ascending=False).astype(int)
    sections["sentinel_rank"] = sections["sentinel_median_relative_signal"].rank(method="first", ascending=False).astype(int)
    sections = sections.sort_values("monitoring_rank").reset_index(drop=True)

    sec_pts = gpd.GeoSeries(gpd.points_from_xy(sections["x_label"], sections["y_label"]), crs=target_crs)
    sec_pts_wgs84 = sec_pts.to_crs("EPSG:4326")
    sections["longitude"] = sec_pts_wgs84.x
    sections["latitude"] = sec_pts_wgs84.y

    edge_df.to_csv(data_dir / "shoreline_edge_pixels.csv", index=False)
    sections.to_csv(data_dir / "shoreline_monitoring_sections.csv", index=False)
    gpd.GeoDataFrame(sections.copy(), geometry=sec_pts, crs=target_crs).to_crs("EPSG:4326").to_file(
        data_dir / "shoreline_monitoring_sections.geojson", driver="GeoJSON"
    )

    top2 = sections.nsmallest(2, "monitoring_rank")["section"].tolist()
    print(f"    - Shoreline ranking: {', '.join(sections['section'].tolist())}")
    print(f"    - Top priority triage zones: {', '.join(top2)}")

    # -------------------------------------------------------------------------
    # Render & Export Figures
    # -------------------------------------------------------------------------
    print(f"[*] Generating figures under {fig_dir} (DPI={dpi})...")
    extent = extent_for(target_transform, target_shape)
    visual_rgba = true_colour_rgba(visual)

    # 01: River Segments
    storet = pd.DataFrame({
        "segment": ["Upstream 1", "Upstream 2", "Downstream 1", "Downstream 2", "Downstream 3"],
        "short": ["U1", "U2", "D1", "D2", "D3"],
        "score": [-10, -40, -48, -52, -50],
        "status": ["Moderately polluted", "Heavily polluted", "Heavily polluted", "Heavily polluted", "Heavily polluted"],
    })
    storet.to_csv(data_dir / "published_storet_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-72, 4)
    ax.set_ylim(-1.4, 4.8)
    river_x = [-69.2, -67.2, -65.2, -63.2, -61.2]
    ax.plot([-71.0, -59.5], [-0.75, -0.75], color=BLUE_LIGHT, linewidth=8, solid_capstyle="round", zorder=1)
    ax.scatter([-67.2], [-0.75], s=210, facecolor="white", edgecolor=ORANGE, linewidth=3, zorder=5)
    for idx, row in storet.iterrows():
        y_val = 4 - idx
        col = ORANGE if row["status"] == "Heavily polluted" else GOLD
        x_node = river_x[idx]
        ax.scatter(x_node, y_val, s=520, facecolor=BLUE_DARK, edgecolor="white", linewidth=2.2, zorder=5)
        ax.text(x_node, y_val, row["short"], ha="center", va="center", color="white", fontsize=13, fontweight="bold", zorder=6)
        ax.hlines(y_val, x_node + 1.2, -60.5, color="#C9D4DC", linewidth=1.3, linestyle=(0, (2, 3)), zorder=1)
        ax.hlines(y_val, row["score"], 0, color=col, linewidth=7, alpha=0.88, zorder=3)
        ax.scatter(row["score"], y_val, s=250, facecolor=col, edgecolor="white", linewidth=2.0, zorder=4)
        ax.text(row["score"] - 1.5, y_val, str(row["score"]), ha="right", va="center", fontsize=14, fontweight="bold")
    ax.set_xticks([-60, -50, -40, -31, -20, -10, 0])
    ax.set_xlabel("STORET score", labelpad=12)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(fig_dir / "01_river_segments_and_scores.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 02: Coastal Context Map
    story_crop = [mouth_centre.x - 5700, mouth_centre.x + 5700, mouth_centre.y - 5200, mouth_centre.y + 1200]
    fig = plt.figure(figsize=(16, 9), facecolor=BLUE_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BLUE_DARK)
    ax.imshow(visual_rgba, extent=extent, interpolation="bicubic", resample=True)
    mangroves.plot(ax=ax, facecolor=GREEN, edgecolor="white", linewidth=0.65, alpha=0.52, zorder=5)
    ax.scatter(mouth_centre.x, mouth_centre.y, s=190, facecolor=ORANGE, edgecolor="white", linewidth=2.5, zorder=12)
    ax.annotate(
        "Sangatta River mouth",
        xy=(mouth_centre.x, mouth_centre.y),
        xytext=(mouth_centre.x + 850, mouth_centre.y + 620),
        fontsize=15, fontweight="bold", color="white",
        arrowprops=dict(arrowstyle="-", color="white", linewidth=2.0),
        bbox=dict(facecolor=INK, alpha=0.88, edgecolor="none", pad=5),
        zorder=13,
    )
    ax.set_xlim(story_crop[0], story_crop[1])
    ax.set_ylim(story_crop[2], story_crop[3])
    ax.set_axis_off()
    add_map_furniture(ax)
    fig.savefig(fig_dir / "02_coastal_context_map.png", dpi=dpi)
    plt.close(fig)

    # 03: Tanager Relative Signal
    tanager_signal = np.full(target_shape, np.nan, dtype="float32")
    tanager_valid = np.isfinite(tanager_tur)
    tanager_signal[tanager_valid] = (100 * rankdata(tanager_tur[tanager_valid], method="average") / tanager_valid.sum())
    signal_crop = [mouth_centre.x - 4980, mouth_centre.x + 4980, mouth_centre.y - 5200, mouth_centre.y + 1200]
    fig = plt.figure(figsize=(16, 9), facecolor=INK)
    map_ax = fig.add_axes([0, 0, 0.875, 1])
    legend_ax = fig.add_axes([0.875, 0, 0.125, 1])
    map_ax.set_facecolor(INK)
    legend_ax.set_facecolor("#F4EFE6")
    map_ax.imshow(visual_rgba, extent=extent, interpolation="bicubic", resample=True, alpha=0.42)
    sig_img = map_ax.imshow(tanager_signal, cmap=SIGNAL_CMAP, vmin=0, vmax=100, extent=extent, interpolation="bicubic", resample=True, alpha=0.95)
    map_ax.scatter(mouth_centre.x, mouth_centre.y, s=190, facecolor=BLUE_DARK, edgecolor="white", linewidth=2.5, zorder=12)
    map_ax.set_xlim(signal_crop[0], signal_crop[1])
    map_ax.set_ylim(signal_crop[2], signal_crop[3])
    map_ax.set_axis_off()
    add_map_furniture(map_ax)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_axis_off()
    legend_ax.text(0.50, 0.82, "Relative turbidity\nsignal", ha="center", va="center", fontsize=13.5, fontweight="bold", color=INK)
    col_ax = fig.add_axes([0.925, 0.285, 0.024, 0.445])
    cb = fig.colorbar(sig_img, cax=col_ax, orientation="vertical")
    cb.set_ticks([0, 50, 100])
    fig.savefig(fig_dir / "03_tanager_relative_signal.png", dpi=dpi)
    plt.close(fig)

    # 04: Same-day check
    tan_rank_map = np.full(target_shape, np.nan, dtype="float32")
    sen_rank_map = np.full(target_shape, np.nan, dtype="float32")
    tan_rank_map[common_water] = 100 * rankdata(tan_vals, method="average") / common_water.sum()
    sen_rank_map[common_water] = 100 * rankdata(sen_vals, method="average") / common_water.sum()
    comp_crop = [mouth_centre.x - 3600, mouth_centre.x + 2750, mouth_centre.y - 5600, mouth_centre.y + 1600]
    fig = plt.figure(figsize=(16, 9), facecolor=INK)
    left = fig.add_axes([0.000, 0, 0.497, 1])
    right = fig.add_axes([0.503, 0, 0.497, 1])
    for ax, img, lbl in [(left, tan_rank_map, "TANAGER"), (right, sen_rank_map, "SENTINEL-2 + ACOLITE")]:
        ax.set_facecolor(INK)
        ax.imshow(visual_rgba, extent=extent, interpolation="bicubic", resample=True, alpha=0.36)
        ax.imshow(img, cmap=SIGNAL_CMAP, vmin=0, vmax=100, extent=extent, interpolation="bicubic", resample=True, alpha=0.96)
        ax.scatter(mouth_centre.x, mouth_centre.y, s=110, facecolor=BLUE_DARK, edgecolor="white", linewidth=2.0, zorder=10)
        ax.set_xlim(comp_crop[0], comp_crop[1])
        ax.set_ylim(comp_crop[2], comp_crop[3])
        ax.set_axis_off()
        ax.text(0.035, 0.965, lbl, transform=ax.transAxes, va="top", fontsize=15.5, fontweight="bold", color="white", bbox=dict(facecolor=INK, alpha=0.90, edgecolor="none", pad=6))
    fig.savefig(fig_dir / "04_same_day_independent_check.png", dpi=dpi)
    plt.close(fig)

    # 05a & 05b: Priority Zones & Ranking
    color_by_section = {"N1": ORANGE, "S1": ORANGE_LIGHT, "S2": BLUE, "S3": "#4F8FC0", "S4": "#78ACCF", "S5": BLUE_LIGHT}
    zone_crop = [mouth_centre.x - 3200, mouth_centre.x + 3200, mouth_centre.y - 5400, mouth_centre.y + 1000]
    map_fig = plt.figure(figsize=(8, 8), facecolor=BLUE_DARK)
    map_ax = map_fig.add_axes([0, 0, 1, 1])
    map_ax.set_facecolor(BLUE_DARK)
    map_ax.imshow(visual_rgba, extent=extent, interpolation="bicubic", resample=True, alpha=0.92)
    for sec, grp in edge_df.groupby("section", observed=True):
        focal = sec in ["N1", "S1"]
        map_ax.scatter(grp["x_utm"], grp["y_utm"], s=22 if focal else 13, color=color_by_section[sec], linewidths=0, alpha=0.98, zorder=8)
        smry = sections.loc[sections["section"] == sec].iloc[0]
        map_ax.text(smry["x_label"], smry["y_label"], sec, ha="center", va="center", fontsize=15.5, fontweight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.28", facecolor=color_by_section[sec], edgecolor="white", linewidth=1.8), zorder=12)
    map_ax.scatter(mouth_centre.x, mouth_centre.y, marker="D", s=105, facecolor=INK, edgecolor="white", linewidth=1.8, zorder=13)
    map_ax.set_xlim(zone_crop[0], zone_crop[1])
    map_ax.set_ylim(zone_crop[2], zone_crop[3])
    map_ax.set_axis_off()
    add_map_furniture(map_ax, length_m=1000)
    map_fig.savefig(fig_dir / "05a_priority_zone_map.png", dpi=dpi)
    plt.close(map_fig)

    rank_fig, rank_ax = plt.subplots(figsize=(8, 8))
    rank_fig.patch.set_facecolor("white")
    rank_fig.subplots_adjust(left=0.17, right=0.94, top=0.94, bottom=0.15)
    ranked = sections.sort_values("relative_signal_score", ascending=True)
    bars = rank_ax.barh(ranked["section"], ranked["relative_signal_score"], color=[color_by_section[v] for v in ranked["section"]], height=0.60)
    clean_axis(rank_ax)
    rank_ax.set_xlim(0, 100)
    rank_ax.set_xlabel("Relative turbidity signal - 0-100 within this scene (unitless)", labelpad=10)
    for bar, val in zip(bars, ranked["relative_signal_score"]):
        rank_ax.text(min(val + 2.0, 94), bar.get_y() + bar.get_height() / 2, f"{val:.0f}", va="center", fontsize=12.5, fontweight="bold")
    rank_fig.savefig(fig_dir / "05b_priority_zone_ranking.png", dpi=dpi, bbox_inches="tight")
    plt.close(rank_fig)

    print(f"[+] CoastCheck Sangatta pipeline completed successfully!")
    return {
        "status": "success",
        "matched_water_pixels": int(common_water.sum()),
        "spearman_rho": rank_rho,
        "agreement_within_2x": within_2x,
        "priority_order": sections["section"].tolist(),
        "top_priority": top2,
    }


def main():
    parser = argparse.ArgumentParser(description="CoastCheck Sangatta Processing & Triage Pipeline")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--tanager-sr", type=Path, default=None, help="Tanager SR HDF5 path")
    parser.add_argument("--tanager-visual", type=Path, default=None, help="Tanager Visual GeoTIFF path")
    parser.add_argument("--sentinel-tur", type=Path, default=None, help="Sentinel-2 ACOLITE turbidity GeoTIFF path")
    parser.add_argument("--gmw-path", type=Path, default=None, help="Global Mangrove Watch GeoJSON path")
    parser.add_argument("--mouth-path", type=Path, default=None, help="River mouth seed GeoJSON path")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for exported figures")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    scene_dir = root / "data" / "coastal" / "20250302_030003_92_4001"

    sr_cand = [
        args.tanager_sr,
        scene_dir / "ortho_sr_hdf5.h5",
    ]
    sr_path = next((p for p in sr_cand if p and p.exists()), None)

    vis_cand = [
        args.tanager_visual,
        scene_dir / "ortho_visual.tif",
    ]
    vis_path = next((p for p in vis_cand if p and p.exists()), None)

    s2_cand = [
        args.sentinel_tur,
        *list((root / "data" / "acolite" / "sangatta_s2c_20250302").glob("*TUR_Dogliotti2015.tif")),
    ]
    s2_path = next((p for p in s2_cand if p and p.exists()), None)

    gmw_cand = [
        args.gmw_path,
        root / "data" / "gmw_clipped.geojson",
    ]
    gmw_path = next((p for p in gmw_cand if p and p.exists()), None)

    mouth_cand = [
        args.mouth_path,
        root / "data" / "mouth_seed.geojson",
    ]
    mouth_path = next((p for p in mouth_cand if p and p.exists()), None)

    missing = []
    if not sr_path: missing.append("Tanager SR HDF5")
    if not vis_path: missing.append("Tanager Visual GeoTIFF")
    if not s2_path: missing.append("Sentinel-2 Dogliotti GeoTIFF")
    if not gmw_path: missing.append("GMW Mangrove GeoJSON")
    if not mouth_path: missing.append("Mouth Seed GeoJSON")

    if missing:
        raise FileNotFoundError(f"Missing required input files: {', '.join(missing)}")

    result = run_pipeline(
        sr_path=sr_path,
        visual_path=vis_path,
        s2_tur_path=s2_path,
        gmw_path=gmw_path,
        mouth_path=mouth_path,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
