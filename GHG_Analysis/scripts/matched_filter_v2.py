"""Run the v2 column-wise matched filter for the two Brazil scenes.

The first pass marks likely plume pixels. The second pass leaves those pixels
out when it estimates the background for each detector column. A brightness
term reduces false signals from unusually bright or dark pixels. The script
saves the three 2.3 micrometre figures explained in Notebook 4.

Run from the package root with:
    python scripts\matched_filter_v2.py
"""

import os, io, sys, time, warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter, median_filter, label

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    import hapi
    HAVE_HITRAN = True
except Exception:
    HAVE_HITRAN = False
TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ---------------------------------------------------------------- constants
PROJ = Path(__file__).resolve().parent.parent
FIG_DIR = PROJ / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
ROOT = PROJ / "data" / "ghg"
IDS = {"north": "20250423_134021_00_4001", "south": "20250423_134026_31_4001"}
TIF = {s: ROOT / IDS[s] / "ortho_ql_ch4.tif" for s in IDS}

RAD = "HDFEOS/SWATHS/HYP/Data Fields/toa_radiance"
LATD = "HDFEOS/SWATHS/HYP/Geolocation Fields/Latitude"
LOND = "HDFEOS/SWATHS/HYP/Geolocation Fields/Longitude"
CLOUD = "HDFEOS/SWATHS/HYP/Data Fields/beta_cloud_mask"
CIRRUS = "HDFEOS/SWATHS/HYP/Data Fields/beta_cirrus_mask"
NODATA = "HDFEOS/SWATHS/HYP/Data Fields/nodata_pixels"
FILL = -9999.0

# primary band only (2.3 um) - the notebook's headline configuration
BAND = dict(lo=2170, hi=2450, clo=2280, chi=2380, table="CH4_b23",
            envc=2330, envf=45, label="2.3 µm")

import pandas as pd
SCENE_METADATA = PROJ / "data" / "ghg" / "scene_metadata.csv"

GSD = 47.6
APIX_V2 = GSD ** 2
OMEGA = 6.7e-7             # kg/m2 per ppm-m (Varon 2018)

# ---------------------------------------------------------------- shared engine
def load_plumes(geojson, strip):
    gdf = gpd.read_file(geojson); out = []
    for _, r in gdf.iterrows():
        g = r.geometry
        lon, lat = (g.x, g.y) if g.geom_type == "Point" else (g.centroid.x, g.centroid.y)
        pid = str(r.get("plume_id", "-"))
        out.append({"label": pid.split("_strip_")[-1] if "_strip_" in pid else pid,
                    "strip": strip, "quality": r.get("plume_quality", "-"),
                    "emission": float(r.get("emission", np.nan)),
                    "ime": float(r.get("ime", np.nan)),
                    "wind": float(r.get("wind_speed_avg", np.nan)),
                    "fetch": float(r.get("fetch", np.nan)),
                    "lon": lon, "lat": lat})
    return out

PLUMES = {s: load_plumes(ROOT / IDS[s] / "ql_ch4_json.geojson", s) for s in IDS}

def load_basic(path, lo, hi):
    with h5py.File(path, "r") as f:
        ds = f[RAD]
        wl = ds.attrs["wavelengths"][:].astype(float)
        fw = ds.attrs["fwhm"][:].astype(float)
        sel = np.where((wl >= lo) & (wl <= hi))[0]
        arr = ds[sel.min():sel.max() + 1].astype(np.float32)
        wlw, fwhm_w = wl[sel.min():sel.max() + 1], fw[sel.min():sel.max() + 1]
        lat = f[LATD][()].astype(float); lon = f[LOND][()].astype(float)
        cl, ci, nd = f[CLOUD][()], f[CIRRUS][()], f[NODATA][()]
    arr[arr <= FILL + 1] = np.nan
    valid = (~((cl == 1) | (ci == 1) | (nd == 1) | (cl == 255))) \
        & np.isfinite(arr[0]) & (arr[0] > 0)
    return arr, wlw, fwhm_w, lat, lon, valid

def methane_envelope(wl, center, fwhm):
    s = fwhm / 2.3548
    A = np.exp(-0.5 * ((wl - center) / s) ** 2)
    return A / A.max()

def hitran_k(wl, fwhm_arr, table):
    if not HAVE_HITRAN or TRAPZ is None:
        return None
    try:
        cache = PROJ / "data" / "hitran_cache"
        os.makedirs(cache, exist_ok=True); hapi.db_begin(str(cache))
        nl, nh = 1e7 / wl.max(), 1e7 / wl.min()
        if table not in hapi.tableList():
            hapi.fetch(table, 6, 1, nl - 30, nh + 30)
        nu, xs = hapi.absorptionCoefficient_Voigt(
            SourceTables=table, HITRAN_units=True,
            Environment={"p": 1.0, "T": 290.0}, WavenumberStep=0.01)
        lam = 1e7 / nu; o = np.argsort(lam); lam, xs = lam[o], xs[o]
        k = np.empty(len(wl))
        for i, (w, fwh) in enumerate(zip(wl, fwhm_arr)):
            s = fwh / 2.3548; m = np.abs(lam - w) < 5 * fwh
            g = np.exp(-0.5 * ((lam[m] - w) / s) ** 2)
            k[i] = TRAPZ(xs[m] * g, lam[m]) / TRAPZ(g, lam[m])
        return k * 2.46e15
    except Exception as e:
        print("  HITRAN failed", repr(e)); return None

def robust_stats(a):
    o = a[np.isfinite(a)]
    m = np.median(o); s = np.median(np.abs(o - m)) * 1.4826 + 1e-12
    return m, s

def nearest(lat, lon, plat, plon):
    return tuple(int(v) for v in np.unravel_index(
        np.nanargmin((lat - plat) ** 2 + (lon - plon) ** 2), lat.shape))

# ---------------------------------------------------------------- two-pass engine
def cwmf_v2(arr, valid, k_vec, SHO, exclude=None, ridge=1e-3):
    """One pass of the v2 engine: brightness-outlier rejection + optional
    exclusion mask for background stats + Foote albedo normalisation r_j."""
    B, Sc, D = arr.shape
    alpha = np.full((Sc, D), np.nan, np.float32)
    for d in range(D):
        good = valid[:, d]
        if exclude is not None:
            good = good & ~exclude[:, d]
        if int(good.sum()) < max(3 * B, 60):
            continue
        Xc = arr[:, good, d].T
        bk = Xc[:, SHO].mean(1)
        bmed = np.median(bk); bmad = np.median(np.abs(bk - bmed)) * 1.4826 + 1e-9
        keep = np.abs(bk - bmed) < 3 * bmad
        Xb = Xc[keep] if int(keep.sum()) > max(3 * B, 60) else Xc
        mu = Xb.mean(0); C = np.cov(Xb, rowvar=False)
        C.flat[::B + 1] += ridge * np.trace(C) / B
        try: Cinv = np.linalg.inv(C)
        except Exception: continue
        t = mu * k_vec; Ct = Cinv @ t; den = float(t @ Ct)
        if den <= 0: continue
        X = arr[:, :, d].T                      # (Sc, B) - score EVERY pixel
        r = (X @ mu) / float(mu @ mu)           # Foote 2020 albedo factor r_j
        r = np.clip(r, 0.5, 3.0)                # guard: don't let dim pixels amplify >2x
        # proper MAG1C form: residual against the BRIGHTNESS-SCALED background r*mu
        alpha[:, d] = (-(((X - r[:, None] * mu) @ Ct) / (r * den))).astype(np.float32)
    return alpha

def run_v2(arr, valid, k_vec, SHO):
    """Two-pass wrapper: pass 1 -> flag >2-sigma pixels -> pass 2 without them."""
    a1 = cwmf_v2(arr, valid, k_vec, SHO)
    a1[~valid] = np.nan
    m0, s0 = robust_stats(a1)
    a1[np.abs((a1 - m0) / s0) > 25] = np.nan       # pathological-pixel gate
    med1, sig1 = robust_stats(a1)
    flag = np.isfinite(a1) & (a1 > med1 + 2 * sig1)  # plume-candidate pixels
    a2 = cwmf_v2(arr, valid, k_vec, SHO, exclude=flag)
    a2[~valid] = np.nan
    m0, s0 = robust_stats(a2)
    a2[np.abs((a2 - m0) / s0) > 25] = np.nan
    return a2

# ---------------------------------------------------------------- rate check
def estimate_rate(alpha, med, q, lat, lon, tif):
    """FOLLOW-TANAGER flux (v3). Integrate OUR ppm·m retrieval over PLANET'S
    published plume footprint (the nonzero pixels of their ortho_ql_ch4
    quicklook, i.e. their own delineation), including sub-threshold pixels -
    over a fixed footprint the retrieval noise is zero-mean, so the diffuse
    tail mass is captured without bias. Then apply Carbon Mapper's own rate
    convention, verified from their published numbers:
        Q = IME * U10 * 3600 / fetch
    (reproduces their Q from their IME/wind/fetch exactly, e.g. strip_C:
     214.83 kg * 1.0185 m/s * 3600 / 990 m = 795.7 kg/h).
    Every input is a published Planet product; only our retrieval amplitude
    is being tested."""
    with rasterio.open(tif) as src:
        ql = src.read(1)
        lab, n = label(ql > 0)
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        # component id at (or nearest to) the reported source point
        x0, y0 = tr.transform(q["lon"], q["lat"])
        r0, c0 = src.index(x0, y0)
        r0 = int(np.clip(r0, 0, ql.shape[0] - 1)); c0 = int(np.clip(c0, 0, ql.shape[1] - 1))
        cid = lab[r0, c0]
        if cid == 0:                                  # source px off-plume: search a window
            win = lab[max(0, r0 - 20):r0 + 21, max(0, c0 - 20):c0 + 21]
            ids, cnt = np.unique(win[win > 0], return_counts=True)
            if ids.size == 0:
                return np.nan, np.nan, 0
            cid = ids[np.argmax(cnt)]
        # sample the footprint at OUR sensor pixels (crop to +/- ~5 km first)
        dlat, dlon = 5.0 / 111.0, 5.0 / (111.0 * np.cos(np.radians(q["lat"])))
        box = (np.abs(lat - q["lat"]) < dlat) & (np.abs(lon - q["lon"]) < dlon)
        rr, cc = np.where(box)
        xs, ys = tr.transform(lon[rr, cc], lat[rr, cc])
        pr, pc = rasterio.transform.rowcol(src.transform, xs, ys)
        pr = np.clip(np.asarray(pr), 0, ql.shape[0] - 1)
        pc = np.clip(np.asarray(pc), 0, ql.shape[1] - 1)
        inside = lab[pr, pc] == cid
    mask = np.zeros_like(alpha, dtype=bool)
    mask[rr[inside], cc[inside]] = True
    vals = alpha[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0, np.nan
    # LOCAL background: median of the surrounding (non-plume) pixels in the
    # same crop box - removes the local glint/albedo baseline that a global
    # median cannot see (standard IME practice: enhancement is defined
    # relative to the plume's immediate surroundings).
    ring = np.zeros_like(alpha, dtype=bool)
    ring[rr, cc] = True
    ring &= ~mask & np.isfinite(alpha)
    local_med = float(np.median(alpha[ring])) if ring.any() else med
    ime = float(np.sum(vals - local_med) * OMEGA * APIX_V2)  # noise zero-mean
    Q = ime * q["wind"] * 3600.0 / q["fetch"]                # CM's own convention
    # BOOTSTRAP uncertainty: translate the same footprint to random plume-free
    # positions and redo the identical sum - measures how much mass a
    # footprint of this size accumulates from correlated background structure.
    rng = np.random.default_rng(42)
    mrows, mcols = np.where(mask)
    mr0, mc0 = mrows.min(), mcols.min()
    sh_r, sh_c = mrows - mr0, mcols - mc0
    H, W = alpha.shape
    sums = []
    tries = 0
    while len(sums) < 25 and tries < 400:
        tries += 1
        orow = rng.integers(0, H - (sh_r.max() + 1))
        ocol = rng.integers(0, W - (sh_c.max() + 1))
        rr2, cc2 = sh_r + orow, sh_c + ocol
        v2_ = alpha[rr2, cc2]
        ok = np.isfinite(v2_)
        if ok.mean() < 0.9 or mask[rr2, cc2].any():
            continue
        sums.append(np.sum(v2_[ok] - np.median(v2_[ok])) * OMEGA * APIX_V2
                    * (len(v2_) / max(ok.sum(), 1)))
    unc = float(np.std(sums)) if len(sums) >= 8 else np.nan
    return ime, Q, int(vals.size), unc

# ---------------------------------------------------------------- figure helpers (notebook styles)
def crop_slices(r, c, shape, half=30):
    return (slice(max(0, r - half), min(shape[0], r + half)),
            slice(max(0, c - half), min(shape[1], c + half)))

def display_map(alpha, med):
    a = np.where(np.isfinite(alpha), alpha, med)
    d = median_filter(a, size=3)
    d[~np.isfinite(alpha)] = np.nan
    return d

HK = 3.0
def _hw(plat, hk=HK): return hk / 111.0, hk / (111.0 * np.cos(np.radians(plat)))
def crop_cube_geo(field, lat, lon, plat, plon, hk=HK):
    dlat, dlon = _hw(plat, hk)
    m = (np.abs(lat - plat) < dlat) & (np.abs(lon - plon) < dlon)
    rr, cc = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return field[rr.min():rr.max() + 1, cc.min():cc.max() + 1] if rr.size and cc.size else field
def crop_planet_geo(tif, plat, plon, hk=HK):
    dlat, dlon = _hw(plat, hk)
    with rasterio.open(tif) as src:
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True); rs, cs = [], []
        for lo, la in [(plon - dlon, plat - dlat), (plon + dlon, plat + dlat),
                       (plon - dlon, plat + dlat), (plon + dlon, plat - dlat)]:
            x, y = tr.transform(lo, la); r, c = src.index(x, y); rs.append(r); cs.append(c)
        band = src.read(1).astype(float)
    return band[max(0, min(rs)):max(rs), max(0, min(cs)):max(cs)]
def fill_disp(z, s=1.2):
    m = np.isfinite(z).astype(np.float32); zf = np.nan_to_num(z)
    out = gaussian_filter(zf * m, s) / np.maximum(gaussian_filter(m, s), 1e-6)
    return gaussian_filter(np.where(np.isfinite(z), z, out), 0.7)

# ---------------------------------------------------------------- main
def main():
    t00 = time.time()
    print("=" * 64)
    print("Two-pass CWMF, Brazil validation (2.3 µm)")
    print("=" * 64)

    # target
    a0, wlw, fwhm_w, _, _, _ = load_basic(ROOT / IDS["north"] / "basic_radiance_hdf5.h5",
                                          BAND["lo"], BAND["hi"]); del a0
    SHO = ~((wlw >= BAND["clo"]) & (wlw <= BAND["chi"]))
    k = hitran_k(wlw, fwhm_w, BAND["table"])
    UNIT = "ppm·m" if k is not None else "relative"
    if k is None:
        k = methane_envelope(wlw, BAND["envc"], BAND["envf"])
    print(f"target built ({UNIT})")

    # AIRMASS FACTOR: sunlight crosses the plume twice (down at SZA, up at
    # VZA), so the retrieved absorption = AMF x the vertical column. All
    # operational retrievals divide this out (Thompson 2015).
    inv = pd.read_csv(SCENE_METADATA).drop_duplicates("item_id").set_index("item_id")
    AMF = {}
    for s, iid in IDS.items():
        sza = np.radians(90.0 - float(inv.loc[iid, "sun_elevation"]))
        vza = np.radians(float(inv.loc[iid, "off_nadir"]) * 1.07)  # ground VZA approx
        AMF[s] = 1.0 / np.cos(sza) + 1.0 / np.cos(vza)
        print(f"  {s}: AMF = {AMF[s]:.2f} (SZA {np.degrees(sza):.1f}, VZA {np.degrees(vza):.1f})")

    S = {}
    for s in IDS:
        arr, _, _, lat, lon, valid = load_basic(ROOT / IDS[s] / "basic_radiance_hdf5.h5",
                                                BAND["lo"], BAND["hi"])
        k_s = k * AMF[s]          # slant-path absorption -> alpha in VERTICAL ppm.m
        t0 = time.time()
        a_v2 = run_v2(arr, valid, k_s, SHO)
        med2, sig2 = robust_stats(a_v2)
        pts = []
        for p in PLUMES[s]:
            r, c = nearest(lat, lon, p["lat"], p["lon"])
            pk2 = float(np.nanmax(a_v2[max(0, r - 5):r + 6, max(0, c - 5):c + 6]))
            pts.append({**p, "row": r, "col": c, "sig2": (pk2 - med2) / sig2})
        fp2 = int(np.nansum(((a_v2 - med2) / sig2) > 5))
        S[s] = dict(a2=a_v2, med2=med2, sig2=sig2,
                    pts=pts, lat=lat, lon=lon, fp2=fp2)
        print(f"  {s}: 1σ={sig2:.0f} {UNIT} | "
              f">5σ pixels={fp2} | {time.time()-t0:.0f}s")
        del arr

    # ---------- printed comparison table ----------
    print("\nDETECTION SIGNIFICANCE (σ)")
    for s in IDS:
        for q in S[s]["pts"]:
            print(f"  strip_{q['label']:1} ({q['quality']:12}): {q['sig2']:5.1f}")

    print("\nFLUX  (Planet reference in brackets)")
    print(f"  {'plume':8} {'IME':>8} {'[Planet]':>9}   "
          f"{'Q':>7} {'[Planet]':>9}  {'pixels':>6}")
    for s in IDS:
        for q in S[s]["pts"]:
            i3, q3, n3, u3 = estimate_rate(S[s]["a2"], S[s]["med2"], q,
                                     S[s]["lat"], S[s]["lon"], TIF[s])
            print(f"  strip_{q['label']:2} {i3:8.1f} {q['ime']:9.1f}   "
                  f"{q3:7.0f} {q['emission']:9.1f}  {n3:6d}")
            print(f"    IME uncertainty (background bootstrap, same footprint size): "
                  f"± {u3:.0f} kg")

    # ---------- fig: detection map (2x2, notebook layout) ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, s, title in [(axes[0, 0], "north", "Northern strip (strip_C)"),
                         (axes[0, 1], "south", "Southern strip (strip_A / strip_B)")]:
        st = S[s]; d = display_map(st["a2"], st["med2"])
        thr = st["med2"] + 2 * st["sig2"]; hi = np.nanpercentile(d, 99.8)
        ax.imshow(np.where(np.isfinite(d), 0., np.nan), cmap="gray", vmin=-1, vmax=1)
        im = ax.imshow(np.where(d > thr, d, np.nan), cmap="inferno", vmin=thr, vmax=hi)
        plt.colorbar(im, ax=ax, fraction=.046, pad=.04, label=f"CH₄ enhancement ({UNIT})")
        for q in st["pts"]:
            ec = "cyan" if q["quality"] == "good" else "yellow"
            ax.plot(q["col"], q["row"], "o", mfc="none", mec=ec, ms=22, mew=1.8)
            ax.annotate(f"strip_{q['label']} {q['sig2']:.1f}σ", (q["col"], q["row"]),
                        color=ec, fontsize=9, xytext=(8, 8), textcoords="offset points")
        ax.set_title(title); ax.axis("off")
    for ax, (s, lbl) in zip([axes[1, 0], axes[1, 1]], [("north", "C"), ("south", "B")]):
        st = S[s]; d = display_map(st["a2"], st["med2"])
        q = next(x for x in st["pts"] if x["label"] == lbl)
        rs, cs = crop_slices(q["row"], q["col"], d.shape, 30); sub = d[rs, cs]
        im = ax.imshow(sub, cmap="inferno", vmin=st["med2"],
                       vmax=max(st["med2"] + 3 * st["sig2"], np.nanpercentile(sub, 99)))
        plt.colorbar(im, ax=ax, fraction=.046, pad=.04, label=f"CH₄ enhancement ({UNIT})")
        ax.plot(q["col"] - cs.start, q["row"] - rs.start, "+", color="cyan", ms=20, mew=2)
        ax.set_title(f"Zoom strip_{lbl} (peak {q['sig2']:.1f}σ)"); ax.axis("off")
    plt.suptitle(f"Two-pass column-wise CWMF with albedo normalisation - CH₄ enhancement (2.3 µm, {UNIT})",
                 fontweight="bold"); plt.tight_layout()
    plt.savefig(FIG_DIR / "04_cwmf_detection_map.png", dpi=150, bbox_inches="tight")
    plt.close(); print("\nSaved 04_cwmf_detection_map.png")

    # ---------- fig: significance (hist + bars, notebook layout) ----------
    PLUME_COL = {"C": "#00e5ff", "A": "#ff3df0", "B": "#39ff14"}
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(15, 4.8))
    for s in IDS:
        st = S[s]
        z2 = (st["a2"] - st["med2"]) / st["sig2"]
        axh.hist(z2[np.isfinite(z2)], bins=200, range=(-5, 25), density=True,
                 color="steelblue", alpha=.45, label="ocean background" if s == "north" else None)
    xx = np.linspace(-5, 25, 400)
    axh.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.3, label="N(0,1) null")
    for s in IDS:
        for q in S[s]["pts"]:
            axh.axvline(q["sig2"], color=PLUME_COL.get(q["label"], "red"), lw=2.4,
                        label=f"strip_{q['label']} {q['sig2']:.1f}σ")
    axh.axvline(3, color="green", ls=":", lw=1.4, label="3σ")
    axh.set_yscale("log"); axh.set_ylim(1e-8, 1.0)   # keep axis readable
    axh.set_xlabel("enhancement (σ)")
    axh.set_ylabel("density (log)")
    axh.set_title("Background distribution and plume signals - 2.3 µm")
    axh.legend(fontsize=7.5)
    labels = [f"strip_{q['label']}" for s in IDS for q in S[s]["pts"]]
    v2s = [q["sig2"] for s in IDS for q in S[s]["pts"]]
    xp = np.arange(len(labels))
    axb.bar(xp, v2s, color="crimson", label="two-pass CWMF")
    axb.axhline(3, color="green", ls=":", label="3σ")
    axb.set_xticks(xp); axb.set_xticklabels(labels)
    axb.set_ylabel("detection significance (σ)")
    axb.set_title("Two-pass CWMF at 2.3 µm"); axb.legend(fontsize=8)
    plt.suptitle("CWMF detection significance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_cwmf_significance.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Saved 04_cwmf_significance.png")

    # ---------- fig: vs Planet (3x2 crops, notebook layout) ----------
    panels = [("north", "C"), ("south", "A"), ("south", "B")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 16.5))
    for i, (s, lbl) in enumerate(panels):
        st = S[s]; q = next(x for x in st["pts"] if x["label"] == lbl)
        ours = fill_disp(crop_cube_geo(st["a2"], st["lat"], st["lon"], q["lat"], q["lon"]))
        planet = crop_planet_geo(TIF[s], q["lat"], q["lon"])
        im0 = axes[i, 0].imshow(ours, cmap="inferno", vmin=st["med2"],
                                vmax=max(st["med2"] + 3 * st["sig2"],
                                         np.nanpercentile(ours, 99.5)))
        # Keep the paired-map labels short, explicit, and consistent with the
        # method and provider product used in the comparison.  These titles are
        # also the source labels for figures cropped into the presentation.
        axes[i, 0].set_title(f"Native CWMF (Basic Radiance) - strip {lbl}\n2.3 µm")
        plt.colorbar(im0, ax=axes[i, 0], fraction=.046, pad=.04, label=UNIT)
        im1 = axes[i, 1].imshow(planet, cmap="inferno")
        axes[i, 1].set_title(f"Tanager Open STAC CH4 Index - strip {lbl}\n2.3 µm")
        plt.colorbar(im1, ax=axes[i, 1], fraction=.046, pad=.04, label="quicklook index")
        for a in (axes[i, 0], axes[i, 1]):
            a.axis("off")
    plt.suptitle("Native CWMF (Basic Radiance) vs Tanager Open STAC CH4 Index - "
                 "all three plumes (2.3 µm; same ~6 km)",
                 fontweight="bold"); plt.tight_layout()
    plt.savefig(FIG_DIR / "04_cwmf_vs_planet.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Saved 04_cwmf_vs_planet.png")

    print(f"\nDone in {time.time()-t00:.0f}s.")

if __name__ == "__main__":
    main()
