"""Reusable methane-retrieval methods for the Tanager SEA study.

The implementation follows the column-wise matched-filter formulation in
Thompson et al. (2015, AMT, Sect. 2.3-2.4):

* work in the non-orthorectified pushbroom geometry;
* estimate one background mean and covariance per detector column;
* construct the methane target as the background radiance multiplied by a
  HITRAN-derived unit absorption spectrum; and
* report concentration-length separately from standardized significance.

The second background pass excludes strong positive candidates before
re-estimating the background, following the sparse-target logic used by
iterative methane matched-filter workflows. Covariance conditioning is an
explicit diagonal ridge rather than an undocumented matrix inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pyproj import Transformer


RADIANCE_PATH = "HDFEOS/SWATHS/HYP/Data Fields/toa_radiance"
LATITUDE_PATH = "HDFEOS/SWATHS/HYP/Geolocation Fields/Latitude"
LONGITUDE_PATH = "HDFEOS/SWATHS/HYP/Geolocation Fields/Longitude"
DATA_FIELDS = "HDFEOS/SWATHS/HYP/Data Fields"


@dataclass
class BasicRadianceScene:
    """Tanager basic-radiance data retained in native detector geometry."""

    radiance: np.ndarray
    wavelengths_nm: np.ndarray
    fwhm_nm: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    sensor_zenith_deg: np.ndarray
    cloud: np.ndarray
    cirrus: np.ndarray
    nodata: np.ndarray
    valid: np.ndarray
    radiance_units: str


@dataclass
class MatchedFilterResult:
    """Column-wise matched-filter output and the diagnostics needed for QA."""

    enhancement_los_ppm_m: np.ndarray
    enhancement_vertical_ppm_m: np.ndarray
    significance: np.ndarray
    first_pass_ppm_m: np.ndarray
    valid: np.ndarray
    exclusion_mask: np.ndarray
    target_absorption_per_ppm_m: np.ndarray
    background_means: np.ndarray
    valid_spectra_per_detector: np.ndarray
    background_spectra_per_detector: np.ndarray
    noise_ppm_m_per_detector: np.ndarray
    theoretical_noise_ppm_m_per_detector: np.ndarray
    condition_number_per_detector: np.ndarray


@dataclass
class SparseMatchedFilterResult:
    """Foote et al. (2020) MAG1C output adapted to native detector columns."""

    enhancement_los_ppm_m: np.ndarray
    enhancement_vertical_ppm_m: np.ndarray
    albedo_factor: np.ndarray
    valid: np.ndarray
    background_spectra_per_detector: np.ndarray
    condition_number_per_detector: np.ndarray


def _as_text(value: Any) -> str:
    """Decode HDF5 text attributes without exposing byte-string syntax."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_basic_radiance(
    path: Path,
    wavelength_min_nm: float = 2104.0,
    wavelength_max_nm: float = 2459.0,
) -> BasicRadianceScene:
    """Load only the methane-window bands from a Tanager basic HDF5 cube.

    The returned array order is ``(band, along_track, detector)``. This is
    deliberately not orthorectified: each final axis index remains one
    physical pushbroom detector column.
    """

    with h5py.File(path, "r") as handle:
        dataset = handle[RADIANCE_PATH]
        wavelengths = np.asarray(dataset.attrs["wavelengths"], dtype=float).squeeze()
        fwhm = np.asarray(dataset.attrs["fwhm"], dtype=float).squeeze()
        selected = np.flatnonzero(
            (wavelengths >= wavelength_min_nm)
            & (wavelengths <= wavelength_max_nm)
        )
        if selected.size == 0:
            raise ValueError("No Tanager bands fall inside the requested methane window.")
        if not np.all(np.diff(selected) == 1):
            raise ValueError("Selected methane bands are unexpectedly non-contiguous.")

        band_slice = slice(int(selected[0]), int(selected[-1]) + 1)
        radiance = np.asarray(dataset[band_slice], dtype=np.float32)
        fill_value = float(dataset.attrs.get("_FillValue", -9999.0))
        units = _as_text(dataset.attrs.get("Unit", "unknown"))

        latitude = np.asarray(handle[LATITUDE_PATH], dtype=float)
        longitude = np.asarray(handle[LONGITUDE_PATH], dtype=float)
        fields = handle[DATA_FIELDS]
        cloud = np.asarray(fields["beta_cloud_mask"], dtype=np.uint8)
        cirrus = np.asarray(fields["beta_cirrus_mask"], dtype=np.uint8)
        nodata = np.asarray(fields["nodata_pixels"], dtype=np.uint8)
        sensor_zenith = np.asarray(fields["sensor_zenith"], dtype=np.float32)

    invalid_radiance = (
        ~np.all(np.isfinite(radiance), axis=0)
        | np.any(radiance <= fill_value + 1.0, axis=0)
        | (np.nanmedian(radiance, axis=0) <= 0.0)
    )
    invalid_mask = (
        (cloud == 1)
        | (cirrus == 1)
        | (nodata == 1)
        | (cloud == 255)
        | (cirrus == 255)
        | (nodata == 255)
    )
    valid = ~(invalid_radiance | invalid_mask)

    return BasicRadianceScene(
        radiance=radiance,
        wavelengths_nm=wavelengths[band_slice],
        fwhm_nm=fwhm[band_slice],
        latitude=latitude,
        longitude=longitude,
        sensor_zenith_deg=sensor_zenith,
        cloud=cloud,
        cirrus=cirrus,
        nodata=nodata,
        valid=valid,
        radiance_units=units,
    )


def build_hitran_ch4_target(
    wavelengths_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    cache_directory: Path,
    table_name: str = "CH4_A1_2104_2459",
    pressure_atm: float = 1.0,
    temperature_k: float = 290.0,
) -> np.ndarray:
    """Return CH4 optical depth per ppm m at each Tanager band.

    HITRAN/HAPI supplies line-by-line cross-sections in cm2 molecule-1. Each
    cross-section is convolved with a Gaussian approximation to Tanager's
    spectral response. The conversion to one ppm metre uses the ideal-gas
    molecular number density at the documented pressure and temperature.

    This is a first-order Beer-Lambert target, matching Thompson et al. (2015)
    Eqs. 9-11. It is not a full atmospheric radiative-transfer calculation.
    """

    import hapi  # Imported here so loading unrelated utilities stays quiet.

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    fwhm_nm = np.asarray(fwhm_nm, dtype=float)
    if wavelengths_nm.shape != fwhm_nm.shape:
        raise ValueError("Wavelength and FWHM arrays must have the same shape.")

    cache_directory.mkdir(parents=True, exist_ok=True)
    hapi.db_begin(str(cache_directory))
    if table_name not in hapi.tableList():
        raise FileNotFoundError(
            f"HITRAN table {table_name!r} is absent from {cache_directory}. "
            "The analysis intentionally does not download spectroscopy silently."
        )

    wavenumber, cross_section = hapi.absorptionCoefficient_Voigt(
        SourceTables=table_name,
        HITRAN_units=True,
        Environment={"p": pressure_atm, "T": temperature_k},
        WavenumberStep=0.01,
    )
    line_wavelength_nm = 1.0e7 / np.asarray(wavenumber, dtype=float)
    cross_section = np.asarray(cross_section, dtype=float)
    order = np.argsort(line_wavelength_nm)
    line_wavelength_nm = line_wavelength_nm[order]
    cross_section = cross_section[order]

    convolved = np.full(wavelengths_nm.shape, np.nan, dtype=float)
    for index, (center, width) in enumerate(zip(wavelengths_nm, fwhm_nm)):
        sigma = width / 2.354820045
        inside = np.abs(line_wavelength_nm - center) <= 5.0 * width
        if inside.sum() < 3:
            continue
        weights = np.exp(
            -0.5 * ((line_wavelength_nm[inside] - center) / sigma) ** 2
        )
        convolved[index] = np.trapezoid(
            cross_section[inside] * weights,
            line_wavelength_nm[inside],
        ) / np.trapezoid(weights, line_wavelength_nm[inside])

    if not np.all(np.isfinite(convolved)):
        missing = int(np.sum(~np.isfinite(convolved)))
        raise RuntimeError(f"HITRAN convolution failed for {missing} Tanager bands.")

    # molecules cm-2 in a 1 ppm, 1 m column at the stated p and T
    boltzmann_j_k = 1.380649e-23
    pressure_pa = pressure_atm * 101_325.0
    molecules_m3 = pressure_pa / (boltzmann_j_k * temperature_k)
    molecules_cm2_per_ppm_m = molecules_m3 * 1.0e-6 * 1.0 / 1.0e4
    return convolved * molecules_cm2_per_ppm_m


def synthetic_target_test(
    target_absorption_per_ppm_m: np.ndarray,
    injected_ppm_m: float = 1000.0,
) -> dict[str, float]:
    """Confirm that the Beer-Lambert target returns a positive known injection."""

    k = np.asarray(target_absorption_per_ppm_m, dtype=float)
    wavelength_index = np.linspace(0.0, 1.0, k.size)
    background = 2.0 + 0.4 * wavelength_index
    target = -background * k
    covariance = np.diag(np.linspace(0.8, 1.2, k.size))
    solved = np.linalg.solve(covariance, target)
    denominator = float(target @ solved)
    injected_spectrum = background + target * injected_ppm_m
    recovered = float((injected_spectrum - background) @ solved / denominator)
    return {
        "injected_ppm_m": float(injected_ppm_m),
        "recovered_ppm_m": recovered,
        "relative_error": (recovered - injected_ppm_m) / injected_ppm_m,
    }


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and Gaussian-consistent median absolute deviation."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    location = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - location)))
    return location, max(scale, np.finfo(float).eps)


def _column_standardize(
    enhancement: np.ndarray,
    valid: np.ndarray,
    background_exclusion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each detector using robust background-only statistics."""

    rows, detectors = enhancement.shape
    significance = np.full((rows, detectors), np.nan, dtype=np.float32)
    noise = np.full(detectors, np.nan, dtype=float)
    for detector in range(detectors):
        background = valid[:, detector] & np.isfinite(enhancement[:, detector])
        if background_exclusion is not None:
            background &= ~background_exclusion[:, detector]
        location, scale = robust_location_scale(enhancement[background, detector])
        if not np.isfinite(scale):
            continue
        noise[detector] = scale
        scoreable = valid[:, detector] & np.isfinite(enhancement[:, detector])
        significance[scoreable, detector] = (
            (enhancement[scoreable, detector] - location) / scale
        ).astype(np.float32)
    return significance, noise


def _matched_filter_pass(
    radiance: np.ndarray,
    valid: np.ndarray,
    target_absorption_per_ppm_m: np.ndarray,
    excluded_background: np.ndarray | None,
    ridge_fraction: float,
    min_samples_per_band: float,
    brightness_quantiles: tuple[float, float],
    background_half_window: int = 3,
    maximum_half_window: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score each detector using an adaptive local across-track background.

    Short or heavily masked scenes often cannot provide five spectra per band
    inside one detector column. Instead of dropping that detector entirely,
    pool background candidates from a small symmetric detector neighbourhood,
    expanding only as far as needed. The fitted filter is still applied only
    to the centre detector's native spectra.
    """

    bands, rows, detectors = radiance.shape
    enhancement = np.full((rows, detectors), np.nan, dtype=np.float32)
    means = np.full((detectors, bands), np.nan, dtype=np.float32)
    valid_counts = np.sum(valid, axis=0).astype(int)
    background_counts = np.zeros(detectors, dtype=int)
    theoretical_noise = np.full(detectors, np.nan, dtype=float)
    condition_numbers = np.full(detectors, np.nan, dtype=float)
    minimum_samples = max(int(np.ceil(min_samples_per_band * bands)), bands + 2)

    for detector in range(detectors):
        half_window = max(0, int(background_half_window))
        while True:
            left = max(0, detector - half_window)
            right = min(detectors, detector + half_window + 1)
            background_mask = valid[:, left:right].copy()
            if excluded_background is not None:
                background_mask &= ~excluded_background[:, left:right]
            if int(background_mask.sum()) >= minimum_samples or half_window >= maximum_half_window:
                break
            half_window = min(maximum_half_window, max(half_window + 1, half_window * 2))

        if excluded_background is not None:
            background_mask &= ~excluded_background[:, left:right]
        if int(background_mask.sum()) < minimum_samples:
            continue

        column = radiance[:, :, detector].T.astype(float, copy=False)
        neighbourhood = radiance[:, :, left:right].transpose(1, 2, 0).astype(float, copy=False)
        candidate_spectra = neighbourhood[background_mask]
        brightness = np.nanmedian(candidate_spectra, axis=1)
        lower, upper = np.quantile(
            brightness,
            brightness_quantiles,
        )
        retained = (brightness >= lower) & (brightness <= upper)
        if int(retained.sum()) < minimum_samples:
            continue

        background = candidate_spectra[retained]
        mean = np.mean(background, axis=0)
        covariance = np.cov(background, rowvar=False, ddof=1)
        ridge = ridge_fraction * float(np.trace(covariance)) / bands
        conditioned = covariance + np.eye(bands) * ridge

        # Methane absorption lowers radiance. Keeping that minus sign inside the
        # target makes a positive enhancement return a positive alpha.
        target = -mean * target_absorption_per_ppm_m
        try:
            solved_target = np.linalg.solve(conditioned, target)
        except np.linalg.LinAlgError:
            continue
        denominator = float(target @ solved_target)
        if not np.isfinite(denominator) or denominator <= 0.0:
            continue

        scoreable = valid[:, detector] & np.all(np.isfinite(column), axis=1)
        enhancement[scoreable, detector] = (
            (column[scoreable] - mean) @ solved_target / denominator
        ).astype(np.float32)
        means[detector] = mean.astype(np.float32)
        background_counts[detector] = int(retained.sum())
        theoretical_noise[detector] = np.sqrt(1.0 / denominator)
        condition_numbers[detector] = np.linalg.cond(conditioned)

    return (
        enhancement,
        means,
        valid_counts,
        background_counts,
        theoretical_noise,
        condition_numbers,
    )


def run_iterative_columnwise_matched_filter(
    scene: BasicRadianceScene,
    target_absorption_per_ppm_m: np.ndarray,
    ridge_fraction: float = 1.0e-3,
    min_samples_per_band: float = 5.0,
    first_pass_exclusion_sigma: float = 2.0,
    brightness_quantiles: tuple[float, float] = (0.005, 0.995),
    background_half_window: int = 3,
    maximum_half_window: int = 12,
) -> MatchedFilterResult:
    """Run a two-pass detector-column matched filter on native radiance.

    ``min_samples_per_band`` defaults to five so cloudy columns can still be
    processed, while the returned sample counts allow the preferred seven
    spectra per band from Ayasse et al. (2023) to be audited explicitly.
    """

    target = np.asarray(target_absorption_per_ppm_m, dtype=float)
    if target.shape != scene.wavelengths_nm.shape:
        raise ValueError("Target length does not match the loaded radiance bands.")
    if ridge_fraction <= 0.0:
        raise ValueError("ridge_fraction must be positive.")

    first = _matched_filter_pass(
        scene.radiance,
        scene.valid,
        target,
        excluded_background=None,
        ridge_fraction=ridge_fraction,
        min_samples_per_band=min_samples_per_band,
        brightness_quantiles=brightness_quantiles,
        background_half_window=background_half_window,
        maximum_half_window=maximum_half_window,
    )
    first_enhancement = first[0]
    first_significance, _ = _column_standardize(
        first_enhancement,
        scene.valid,
    )
    exclusion = (
        scene.valid
        & np.isfinite(first_significance)
        & (first_significance > first_pass_exclusion_sigma)
    )

    second = _matched_filter_pass(
        scene.radiance,
        scene.valid,
        target,
        excluded_background=exclusion,
        ridge_fraction=ridge_fraction,
        min_samples_per_band=min_samples_per_band,
        brightness_quantiles=brightness_quantiles,
        background_half_window=background_half_window,
        maximum_half_window=maximum_half_window,
    )
    final_enhancement = second[0]
    significance, robust_noise = _column_standardize(
        final_enhancement,
        scene.valid,
        background_exclusion=exclusion,
    )

    # A localized near-surface enhancement is traversed on the sensor-to-ground
    # leg. Convert line-of-sight concentration length to an approximate vertical
    # column using the per-pixel sensor zenith angle.
    vertical = final_enhancement * np.cos(
        np.deg2rad(scene.sensor_zenith_deg)
    ).astype(np.float32)

    return MatchedFilterResult(
        enhancement_los_ppm_m=final_enhancement,
        enhancement_vertical_ppm_m=vertical,
        significance=significance,
        first_pass_ppm_m=first_enhancement,
        valid=scene.valid,
        exclusion_mask=exclusion,
        target_absorption_per_ppm_m=target,
        background_means=second[1],
        valid_spectra_per_detector=second[2],
        background_spectra_per_detector=second[3],
        noise_ppm_m_per_detector=robust_noise,
        theoretical_noise_ppm_m_per_detector=second[4],
        condition_number_per_detector=second[5],
    )


def run_columnwise_mag1c(
    scene: BasicRadianceScene,
    target_absorption_per_ppm_m: np.ndarray,
    iterations: int = 30,
    covariance_update_scaling: float = 1.0,
    covariance_diagonal_fraction: float = 1.0e-3,
    brightness_quantiles: tuple[float, float] = (0.005, 0.995),
    min_samples_per_band: float = 5.0,
    background_half_window: int = 3,
    maximum_half_window: int = 12,
) -> SparseMatchedFilterResult:
    """Run an albedo-corrected, reweighted-L1 sparse matched filter.

    This is a NumPy adaptation of the public MAG1C equations and reference
    implementation from Foote et al. (2020). The published defaults retained
    here are 30 iterations, per-detector grouping, a non-negative solution,
    reweighted-L1 sparsity, and full removal of the previously retrieved target
    from each background-statistics update.

    The only intentional numerical addition is a 0.001 diagonal covariance
    interpolation, needed to condition short satellite columns. The returned
    values are ppm m; MAG1C's internal 1e5 scaling is applied and then removed
    exactly as in the authors' implementation.
    """

    k = np.asarray(target_absorption_per_ppm_m, dtype=float)
    if k.shape != scene.wavelengths_nm.shape:
        raise ValueError("Target length does not match the loaded radiance bands.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if not 0.0 <= covariance_diagonal_fraction <= 1.0:
        raise ValueError("covariance_diagonal_fraction must be in [0, 1].")

    bands, rows, detectors = scene.radiance.shape
    minimum_samples = max(int(np.ceil(min_samples_per_band * bands)), bands + 2)
    enhancement = np.full((rows, detectors), np.nan, dtype=np.float32)
    albedo = np.full((rows, detectors), np.nan, dtype=np.float32)
    background_counts = np.zeros(detectors, dtype=int)
    condition_numbers = np.full(detectors, np.nan, dtype=float)

    # MAG1C's reference target is scaled so one internal concentration unit
    # equals 100,000 ppm m; the final result is multiplied by the same factor.
    concentration_scale = 1.0e5
    template = -k * concentration_scale
    epsilon = 1.0e-9

    for detector in range(detectors):
        scoreable = scene.valid[:, detector]
        if not np.any(scoreable):
            continue
        row_indices = np.flatnonzero(scoreable)
        spectra = scene.radiance[:, scoreable, detector].T.astype(float, copy=False)

        half_window = max(0, int(background_half_window))
        while True:
            left = max(0, detector - half_window)
            right = min(detectors, detector + half_window + 1)
            candidate_mask = scene.valid[:, left:right]
            if int(candidate_mask.sum()) >= minimum_samples or half_window >= maximum_half_window:
                break
            half_window = min(maximum_half_window, max(half_window + 1, half_window * 2))
        if int(candidate_mask.sum()) < minimum_samples:
            continue

        neighbourhood = scene.radiance[:, :, left:right].transpose(1, 2, 0).astype(float, copy=False)
        candidates = neighbourhood[candidate_mask]
        brightness = np.nanmedian(candidates, axis=1)
        lower, upper = np.quantile(brightness, brightness_quantiles)
        stats_mask = (brightness >= lower) & (brightness <= upper)
        if int(stats_mask.sum()) < minimum_samples:
            continue

        background = candidates[stats_mask].copy()
        mean = np.mean(background, axis=0)
        albedo_factor = (spectra @ mean) / float(mean @ mean)
        background_albedo = (background @ mean) / float(mean @ mean)
        if np.any(~np.isfinite(albedo_factor)) or np.any(albedo_factor <= 0.0):
            continue
        if np.any(~np.isfinite(background_albedo)) or np.any(background_albedo <= 0.0):
            continue

        target = mean * template
        centered = background - mean
        covariance = centered.T @ centered / float(background.shape[0])
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            (1.0 - covariance_diagonal_fraction) * covariance
            + covariance_diagonal_fraction * diagonal
        )
        try:
            solved_target = np.linalg.solve(covariance, target)
        except np.linalg.LinAlgError:
            continue
        normalizer = float(target @ solved_target)
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            continue
        matched = ((spectra - mean) @ solved_target) / (
            albedo_factor * normalizer
        )
        matched = np.maximum(matched, 0.0)
        background_matched = np.maximum(
            ((background - mean) @ solved_target) / (background_albedo * normalizer),
            0.0,
        )

        for _ in range(iterations):
            regularizer = 1.0 / (albedo_factor * (matched + epsilon))
            background_regularizer = 1.0 / (
                background_albedo * (background_matched + epsilon)
            )
            modified_background = (
                background
                - covariance_update_scaling
                * background_albedo[:, None]
                * background_matched[:, None]
                * target[None, :]
            )
            mean = np.mean(modified_background, axis=0)
            target = mean * template
            centered = modified_background - mean
            covariance = centered.T @ centered / float(modified_background.shape[0])
            diagonal = np.diag(np.diag(covariance))
            covariance = (
                (1.0 - covariance_diagonal_fraction) * covariance
                + covariance_diagonal_fraction * diagonal
            )
            try:
                solved_target = np.linalg.solve(covariance, target)
            except np.linalg.LinAlgError:
                matched[:] = np.nan
                break
            normalizer = max(float(target @ solved_target), 1.0)
            matched = (
                (spectra - mean) @ solved_target - regularizer
            ) / (albedo_factor * normalizer)
            matched = np.maximum(matched, 0.0)
            background_matched = (
                (background - mean) @ solved_target - background_regularizer
            ) / (background_albedo * normalizer)
            background_matched = np.maximum(background_matched, 0.0)

        if not np.all(np.isfinite(matched)):
            continue
        enhancement[row_indices, detector] = (
            matched * concentration_scale
        ).astype(np.float32)
        albedo[row_indices, detector] = albedo_factor.astype(np.float32)
        background_counts[detector] = int(background.shape[0])
        condition_numbers[detector] = np.linalg.cond(covariance)

    vertical = enhancement * np.cos(
        np.deg2rad(scene.sensor_zenith_deg)
    ).astype(np.float32)
    return SparseMatchedFilterResult(
        enhancement_los_ppm_m=enhancement,
        enhancement_vertical_ppm_m=vertical,
        albedo_factor=albedo,
        valid=scene.valid,
        background_spectra_per_detector=background_counts,
        condition_number_per_detector=condition_numbers,
    )


def projected_coordinates_and_pixel_area(
    latitude: np.ndarray,
    longitude: np.ndarray,
    epsg: int = 32648,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project geolocation and estimate each native pixel's ground area.

    The Jacobian of the native row/column-to-UTM coordinate mapping captures
    the actual local scan geometry instead of assuming a fixed 32 m square.
    """

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    dx_drow, dx_dcol = np.gradient(x)
    dy_drow, dy_dcol = np.gradient(y)
    area = np.abs(dx_dcol * dy_drow - dx_drow * dy_dcol)
    return np.asarray(x), np.asarray(y), np.asarray(area)


def nearest_native_pixel(
    latitude: np.ndarray,
    longitude: np.ndarray,
    target_latitude: float,
    target_longitude: float,
) -> tuple[int, int]:
    """Return the native row and detector nearest a lon/lat point."""

    distance_proxy = (
        (latitude - target_latitude) ** 2
        + ((longitude - target_longitude) * np.cos(np.deg2rad(target_latitude))) ** 2
    )
    return tuple(
        int(value) for value in np.unravel_index(np.nanargmin(distance_proxy), latitude.shape)
    )
