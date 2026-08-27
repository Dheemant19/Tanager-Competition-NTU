"""Focused tests for the compact workbench GHG API."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from api.ghg import (
    _north_up_grid,
    capabilities_response,
    load_scene_manifest,
    reference_response,
    scene_is_methane_eligible,
)
from api.ghg_methods import BasicRadianceScene, run_iterative_columnwise_matched_filter


HCMC_SCENE_ID = "20250407_035509_25_4001"


class GhgApiTests(unittest.TestCase):
    def test_manifest_covers_ghg_scenes_plus_hcmc(self) -> None:
        scenes = load_scene_manifest()["scenes"]
        self.assertIn(HCMC_SCENE_ID, scenes)
        self.assertTrue(scene_is_methane_eligible(HCMC_SCENE_ID))
        self.assertEqual(len(scenes), 9)
        self.assertTrue(all(scene.get("basic_radiance", {}).get("url") for scene in scenes.values()))

    def test_capabilities_match_scene_scope(self) -> None:
        hcmc = capabilities_response(HCMC_SCENE_ID)
        self.assertTrue(hcmc["methane_available"])
        self.assertTrue(hcmc["reference_available"])

    def test_native_geolocation_is_binned_north_up(self) -> None:
        latitude = np.repeat(np.linspace(1.0, 0.0, 6)[:, None], 6, axis=1)
        longitude = np.repeat(np.linspace(100.0, 101.0, 6)[None, :], 6, axis=0)
        values = latitude.copy()
        mask = np.ones_like(values, dtype=bool)

        grid, occupied, bounds = _north_up_grid(values, latitude, longitude, mask, 160)
        self.assertEqual(bounds, [100.0, 0.0, 101.0, 1.0])
        self.assertEqual(int(occupied.sum()), 36)
        self.assertGreater(float(np.nanmean(grid[0])), float(np.nanmean(grid[-1])))

    def test_short_columns_use_local_detector_background(self) -> None:
        rng = np.random.default_rng(7)
        bands, rows, detectors = 5, 12, 5
        radiance = (2.0 + rng.normal(0.0, 0.02, (bands, rows, detectors))).astype(np.float32)
        valid = np.zeros((rows, detectors), dtype=bool)
        valid[:6] = True  # Six per column is below the required ten.
        grid = np.zeros((rows, detectors), dtype=np.float32)
        scene = BasicRadianceScene(
            radiance=radiance,
            wavelengths_nm=np.linspace(2200.0, 2300.0, bands),
            fwhm_nm=np.full(bands, 5.0),
            latitude=grid,
            longitude=grid,
            sensor_zenith_deg=grid,
            cloud=grid,
            cirrus=grid,
            nodata=grid,
            valid=valid,
            radiance_units="test",
        )
        target = np.linspace(1.0e-5, 2.0e-5, bands)
        result = run_iterative_columnwise_matched_filter(
            scene,
            target,
            min_samples_per_band=2.0,
            background_half_window=1,
            maximum_half_window=2,
        )
        self.assertEqual(int(np.isfinite(result.enhancement_vertical_ppm_m).sum()), int(valid.sum()))
        self.assertTrue(np.all(result.background_spectra_per_detector > 0))

    @patch("api.ghg._load_geojson", return_value=[])
    @patch("api.ghg._read_reference_raster")
    def test_empty_provider_quicklook_is_a_valid_comparison(self, read_raster, _load_geojson) -> None:
        read_raster.return_value = (
            np.zeros((20, 20), dtype=np.float32),
            0.0,
            [0.0, 0.0, 1.0, 1.0],
        )
        response = reference_response(
            "empty-test-scene",
            {"published_reference": {"url": "https://example.invalid/reference.tif"}},
            480,
        )
        self.assertEqual(response["product"]["metrics"]["valid_pixel_count"], 0)
        self.assertEqual(response["product"]["range"], [0.0, 1.0])
        self.assertEqual(response["product"]["units"], "display value")


if __name__ == "__main__":
    unittest.main()
