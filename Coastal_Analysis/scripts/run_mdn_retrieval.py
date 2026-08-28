"""
scripts/run_mdn_retrieval.py

Run the Pahlevan/Smith Mixture Density Network (MDN) on the Borneo coastal
Tanager scene to retrieve chlorophyll-a, total suspended solids (TSS), and
CDOM.

Run this in the isolated `tanager-mdn` conda environment. It resamples the
surface-reflectance cube to Sentinel-3 OLCI bands, builds remote-sensing
reflectance (Rrs), runs the MDN, and saves product maps for
`notebooks/05_ai_water_quality.ipynb`.

Usage from the repository root:
    conda run -n tanager-mdn python scripts/run_mdn_retrieval.py
"""
from pathlib import Path
import sys
import numpy as np
import h5py
from MDN import image_estimates, get_sensor_bands

SENSOR = "OLCI"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_ID = "20250302_030003_92_4001"
DATA_DIR = PROJECT_ROOT / "data" / "coastal" / SCENE_ID
SR_PATH = DATA_DIR / "ortho_sr_hdf5.h5"
OUT = DATA_DIR / "mdn_olci_products.npz"
DF, FILL = "HDFEOS/GRIDS/HYP/Data Fields", -9999.0

# MDN OLCI band centres (nm), must match get_sensor_bands('OLCI') order
OLCI = [411, 442, 490, 510, 560, 619, 664, 673, 681, 708, 753, 778]


def estimate_and_save(cube, water, out):
    """Run MDN retrieval and save masked products."""
    H, W = water.shape
    rr, cc = np.where(water)
    print("Rrs median by band:", np.round(np.nanmedian(cube[rr, cc, :], axis=0), 5))

    res = image_estimates(cube, sensor=SENSOR, product="chl,tss,cdom")
    est = res[0] if isinstance(res, (list, tuple)) else res
    slices = res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else None
    est = np.array(est)
    print("estimates shape:", est.shape, "| slices:", slices)

    P = est.shape[-1]
    est2 = est.reshape(H * W, P)
    wflat = water.reshape(-1)

    products = {}
    items = slices.items() if isinstance(slices, dict) else [(f"product_{j}", j) for j in range(P)]
    for name, sl in items:
        col = est2[:, sl]
        col = col[:, 0] if getattr(col, "ndim", 1) == 2 else col
        m = np.full(H * W, np.nan, "float32"); m[wflat] = np.asarray(col)[wflat]
        products[str(name)] = m.reshape(H, W)

    print("products saved:", list(products.keys()))
    np.savez_compressed(out, water=water, **products)
    print("saved ->", out)


def main_from_rrs(rrs_npz):
    """Run MDN retrieval from a prepared OLCI-band Rrs cube."""
    rrs_npz = Path(rrs_npz)
    d = np.load(rrs_npz, allow_pickle=True)
    cube = d["rrs"].astype("float32"); water = d["water"].astype(bool)
    out = rrs_npz.with_name(rrs_npz.stem.replace("_prep", "") + "_mdn_olci_products.npz")
    print(f"input {rrs_npz.name}: cube {cube.shape}, water {int(water.sum())} px")
    estimate_and_save(cube, water, out)


def main():
    mdn_bands = list(np.array(get_sensor_bands(SENSOR)).ravel())
    print("MDN", SENSOR, "bands:", mdn_bands)
    print("our resample targets:", OLCI)

    with h5py.File(SR_PATH, "r") as f:
        SR = f[f"{DF}/surface_reflectance"]
        wl = np.asarray(SR.attrs["wavelengths"], float)
        gw = np.asarray(SR.attrs["good_wavelengths"]).astype(bool)

        def winmean(c, half=6.0):
            idx = np.where((wl >= c - half) & (wl <= c + half) & gw)[0]
            a = SR[idx, :, :].astype("float32"); a[a == FILL] = np.nan
            return np.nanmean(a, axis=0)

        def band(c):
            i = int(np.argmin(np.abs(wl - c))); b = SR[i].astype("float32"); b[b == FILL] = np.nan; return b

        g560, nir861, swir1610 = band(560), band(861), band(1610)
        layers = [winmean(c) for c in OLCI]
        nodata = f[f"{DF}/nodata_pixels"][:]; cloud = f[f"{DF}/beta_cloud_mask"][:]; cirrus = f[f"{DF}/beta_cirrus_mask"][:]

    H, W = g560.shape
    ndwi = (g560 - nir861) / (g560 + nir861)
    water = (nodata == 0) & (cloud == 0) & (cirrus == 0) & (ndwi > 0) & np.isfinite(layers[0])
    print("water pixels:", int(water.sum()))

    # Remove SWIR glint, convert rho to Rrs, and stack bands.
    cube = np.stack([np.clip(L - swir1610, 0, None) / np.pi for L in layers], axis=-1).astype("float32")
    estimate_and_save(cube, water, OUT)


if __name__ == "__main__":
    # MDN parses sys.argv internally, so preserve and hide the --rrs arguments.
    _argv = sys.argv[1:]
    sys.argv = [sys.argv[0]]
    if len(_argv) >= 2 and _argv[0] == "--rrs":
        main_from_rrs(_argv[1])
    else:
        main()
