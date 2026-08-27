"""Small local tests for ROI statistics and the arbitrary-band composer."""

from __future__ import annotations

import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

from api.composite import composite_from_root
from api.coastal import coastal_from_root, scene_is_coastal, scene_is_eligible
from api.roi import (
    MAX_ROI_SPECTRAL_WINDOW_PIXELS,
    extract_roi_product_from_root,
    parse_polygon,
    polygon_mask,
    spectral_sampling_grid,
)
from api.spectrum import SpectrumError, extract_product_from_root


class ScienceApiTests(unittest.TestCase):
    @contextmanager
    def hdf5_fixture(self):
        path = Path(__file__).with_name(f"_science_test_{uuid.uuid4().hex}.h5")
        try:
            with self.make_test_hdf5(path) as root:
                yield root
        finally:
            path.unlink(missing_ok=True)

    def make_test_hdf5(self, path: Path) -> h5py.File:
        root = h5py.File(path, "w")
        grid = root.create_group("HDFEOS/GRIDS/HYP")
        grid.attrs["epsg_code"] = 32631
        fields = root.create_group("HDFEOS/GRIDS/HYP/Data Fields")
        cube = np.stack(
            [
                np.arange(16, dtype=float).reshape(4, 4) + 10,
                np.arange(16, dtype=float).reshape(4, 4) + 20,
                np.arange(16, dtype=float).reshape(4, 4) + 30,
            ]
        )
        sr = fields.create_dataset("surface_reflectance", data=cube)
        sr.attrs["wavelengths"] = np.asarray([500.0, 600.0, 700.0])
        # Kerchunk exposes these flags as numeric 0/1, not Boolean values.
        sr.attrs["good_wavelengths"] = np.asarray([1, 0, 1], dtype=np.uint8)
        sr.attrs["wavelengths_units"] = "nm"
        sr.attrs["Unit"] = "unitless"
        sr.attrs["_FillValue"] = -9999.0
        fields.create_dataset("beta_cloud_mask", data=np.zeros((4, 4), dtype=np.uint8))
        fields.create_dataset("beta_cirrus_mask", data=np.zeros((4, 4), dtype=np.uint8))
        nodata = np.zeros((4, 4), dtype=np.uint8)
        nodata[0, 0] = 1
        fields.create_dataset("nodata_pixels", data=nodata)
        metadata = (
            "UpperLeftPointMtrs=(500000,1000)\n"
            "LowerRightMtrs=(500400,600)\n"
            "XDim=4\n"
            "YDim=4\n"
            "ZoneCode=31\n"
        )
        root.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.bytes_(metadata))
        return root

    def make_coastal_hdf5(self, path: Path) -> h5py.File:
        root = h5py.File(path, "w")
        grid = root.create_group("HDFEOS/GRIDS/HYP")
        grid.attrs["epsg_code"] = 32631
        fields = root.create_group("HDFEOS/GRIDS/HYP/Data Fields")
        wavelengths = np.asarray([
            443, 560, 620, 630, 640, 650, 660, 665, 670, 708,
            841, 850, 859, 861, 865, 870, 876, 1600, 1610, 1620,
        ], dtype=float)
        yy, xx = np.mgrid[0:12, 0:12]
        gradient = xx / 11
        cube = np.empty((wavelengths.size, 12, 12), dtype=np.float32)
        for index, wavelength in enumerate(wavelengths):
            if wavelength == 443:
                cube[index] = 0.032 + yy * 0.0002
            elif wavelength == 560:
                cube[index] = 0.050 + yy * 0.0003
            elif 620 <= wavelength <= 670:
                cube[index] = 0.018 + gradient * 0.035
            elif wavelength == 708:
                cube[index] = 0.014 + gradient * 0.020
            elif 841 <= wavelength <= 876:
                cube[index] = 0.010 + yy * 0.0001
            else:
                cube[index] = 0.002
        sr = fields.create_dataset("surface_reflectance", data=cube)
        sr.attrs["wavelengths"] = wavelengths
        sr.attrs["good_wavelengths"] = np.ones(wavelengths.size, dtype=np.uint8)
        sr.attrs["_FillValue"] = -9999.0
        fields.create_dataset("beta_cloud_mask", data=np.zeros((12, 12), dtype=np.uint8))
        fields.create_dataset("beta_cirrus_mask", data=np.zeros((12, 12), dtype=np.uint8))
        fields.create_dataset("nodata_pixels", data=np.zeros((12, 12), dtype=np.uint8))
        metadata = (
            "UpperLeftPointMtrs=(500000,1000)\n"
            "LowerRightMtrs=(500360,640)\n"
            "XDim=12\n"
            "YDim=12\n"
            "ZoneCode=31\n"
        )
        root.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.bytes_(metadata))
        return root

    @staticmethod
    def full_grid_ring() -> list[tuple[float, float]]:
        inverse = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
        corners = [
            (500000, 1000),
            (500400, 1000),
            (500400, 600),
            (500000, 600),
            (500000, 1000),
        ]
        return [inverse.transform(x, y) for x, y in corners]

    def test_polygon_validation_and_mask(self) -> None:
        payload = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
            }
        }
        ring = parse_polygon(payload)
        mask = polygon_mask(
            np.asarray([0.5, 1.5, 2.5]),
            np.asarray([0.5, 1.5, 2.5]),
            np.asarray(ring),
        )
        self.assertEqual(int(mask.sum()), 4)

    def test_large_roi_uses_bounded_spectral_grid(self) -> None:
        mask = np.ones((400, 400), dtype=bool)
        stride, _, _, sampled = spectral_sampling_grid(mask)
        self.assertGreater(stride, 1)
        self.assertLessEqual(sampled.size, MAX_ROI_SPECTRAL_WINDOW_PIXELS)

    def test_roi_removes_bad_sr_band_and_reports_nodata(self) -> None:
        with self.hdf5_fixture() as root:
            result = extract_roi_product_from_root(
                "ortho_sr",
                {"url": "memory://tiny"},
                root,
                self.full_grid_ring(),
                "hdf5",
            )
            inverse = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
            lon, lat = inverse.transform(500200, 800)
            point = extract_product_from_root(
                "ortho_sr",
                {"url": "memory://tiny"},
                root,
                lat,
                lon,
                0,
                "hdf5",
            )
        self.assertEqual(result["roi"]["selected_pixel_count"], 16)
        self.assertEqual(result["roi"]["data_pixel_count"], 15)
        self.assertEqual(result["good_wavelengths"], [True, False, True])
        self.assertIsNone(result["values"][1])
        self.assertEqual(result["statistics"]["valid_count"][1], 0)
        self.assertIn("mean", result["statistics"])
        self.assertNotIn("q25", result["statistics"])
        self.assertEqual(result["roi"]["analysis_stride"], 1)
        self.assertAlmostEqual(result["qa"]["nodata_fraction"], 1 / 16)
        self.assertEqual(point["good_wavelengths"], [True, False, True])
        self.assertIsNone(point["values"][1])

    def test_composer_outputs_png_and_rejects_bad_sr_band(self) -> None:
        with self.hdf5_fixture() as root:
            png, metadata = composite_from_root(
                "ortho_sr",
                {"url": "memory://tiny"},
                root,
                (500.0, 700.0, 500.0),
                2,
                98,
                160,
                "hdf5",
            )
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(metadata["preview_shape"], [4, 4])
            self.assertEqual(metadata["render_mode"], "rgb")

            sr = root["HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance"]
            sr.attrs.modify("good_wavelengths", np.asarray([1, 1, 1], dtype=np.uint8))
            index_png, index_metadata = composite_from_root(
                "ortho_sr",
                {"url": "memory://tiny"},
                root,
                (500.0, 700.0, 500.0),
                2,
                98,
                160,
                "hdf5",
                "ndwi",
            )
            self.assertTrue(index_png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(index_metadata["render_mode"], "index")
            self.assertEqual(index_metadata["recipe_label"], "NDWI")

            sr.attrs.modify("good_wavelengths", np.asarray([1, 0, 1], dtype=np.uint8))
            with self.assertRaises(SpectrumError) as context:
                composite_from_root(
                    "ortho_sr",
                    {"url": "memory://tiny"},
                    root,
                    (600.0, 700.0, 500.0),
                    2,
                    98,
                    160,
                    "hdf5",
                )
        self.assertIn("bad band", context.exception.message)

    def test_coastal_products_and_collection_eligibility(self) -> None:
        scene = {
            "collection": "coastal-water-bodies",
            "collections": ["coastal-water-bodies"],
            "centroid_lon": 117.5,
            "centroid_lat": 0.8,
        }
        self.assertTrue(scene_is_coastal(scene))
        self.assertTrue(scene_is_eligible(scene))
        outside_sea = {**scene, "centroid_lon": -122.4, "centroid_lat": 37.8}
        self.assertTrue(scene_is_eligible(outside_sea))
        self.assertFalse(scene_is_eligible({**scene, "collections": ["urban"]}))

        path = Path(__file__).with_name(f"_coastal_test_{uuid.uuid4().hex}.h5")
        try:
            with self.make_coastal_hdf5(path) as root:
                relative = coastal_from_root(
                    {"url": "memory://coastal"},
                    root,
                    max_size=160,
                    workflow="relative",
                    source_kind="hdf5",
                )
                fnu = coastal_from_root(
                    {"url": "memory://coastal"},
                    root,
                    max_size=160,
                    workflow="fnu",
                    source_kind="hdf5",
                )
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(
            [product["key"] for product in relative["products"]],
            ["relative_turbidity", "relative_cdom", "ndci"],
        )
        self.assertEqual([product["key"] for product in fnu["products"]], ["turbidity_fnu"])
        self.assertGreater(relative["qa"]["water_pixel_count"], 25)
        for product in [*relative["products"], *fnu["products"]]:
            self.assertTrue(product["image"].startswith("data:image/png;base64,"))
            self.assertTrue(np.isfinite(product["median"]))
            self.assertEqual(len(product["legend"]["colors"]), 5)
            self.assertEqual(len(product["legend"]["ticks"]), 3)
        self.assertIn("p95", fnu["products"][0])
        self.assertEqual(relative["georeferencing"]["epsg"], 32631)
        self.assertEqual(relative["georeferencing"]["grid_shape"], [12, 12])
        self.assertEqual(relative["georeferencing"]["pixel_size_m"], [30.0, 30.0])
        self.assertEqual(len(relative["georeferencing"]["corners"]), 4)
        self.assertLess(relative["georeferencing"]["bounds"][0], relative["georeferencing"]["bounds"][2])
        self.assertLess(relative["georeferencing"]["bounds"][1], relative["georeferencing"]["bounds"][3])


if __name__ == "__main__":
    unittest.main()
