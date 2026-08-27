"""Focused tests for the compact workbench GHG API."""

from __future__ import annotations

import unittest

import numpy as np

from api.ghg import (
    _north_up_grid,
    capabilities_response,
    load_scene_manifest,
    scene_is_methane_eligible,
)


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


if __name__ == "__main__":
    unittest.main()
