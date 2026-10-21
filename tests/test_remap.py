# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-process tests for the two properties ``ConservativeRemap`` promises.

Straight from the module's own docstring: **consistency** (a constant field
stays constant) and **conservation** (the integral over the sphere is
unchanged) are the two properties that make a remapping "conservative" rather
than merely an interpolation, and they are not the same property -- an
interpolation can have the first without the second. These tests check both
directly, for both source kinds, rather than just checking the output is
finite or reasonable-looking.

The distributed operator (``tests/distributed/test_distributed_remap.py``)
is checked against *this* serial one, not against these properties directly;
if the distributed and serial outputs agree exactly (as that suite checks),
whatever holds here holds there too. So this is where a regression in the
underlying remapping math -- as opposed to the distributed plumbing around
it -- would actually show up.
"""

import unittest
from parameterized import parameterized_class

import numpy as np
import torch

from makani.utils.remap import ConservativeRemap, cell_bounds_longitude, cell_bounds_sin

from .testutils import compare_tensors, disable_tf32, set_seed

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def _fibonacci_mesh(n_cells: int) -> tuple:
    """A quasi-uniform mesh of ``n_cells`` points on the sphere (Fibonacci spiral)."""
    index = np.arange(n_cells, dtype=np.float64)
    lat = np.degrees(np.arcsin(1.0 - 2.0 * (index + 0.5) / n_cells))
    lon = np.mod(np.degrees(index * np.pi * (3.0 - np.sqrt(5.0))), 360.0)
    return lat, lon


def _raster_cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Area (steradians) of each cell of a lat/lon grid: ``dA = d(sin phi) dlambda``.

    Independent of (and simpler than) anything inside ``remap.py`` -- this is
    the textbook spherical-quadrature weight, used here purely as a ground
    truth to check the operator's output against, not reused from the
    module under test.
    """
    dsin = np.abs(np.diff(cell_bounds_sin(lat)))
    dlon = np.radians(np.abs(np.diff(cell_bounds_longitude(lon))))
    return dsin[:, None] * dlon[None, :]


@parameterized_class(("device",), _devices)
class TestRasterProperties(unittest.TestCase):
    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_consistency_constant_field(self):
        """A constant field in gives the same constant back out, coarsening or refining."""
        cases = [(48, 96, 24, 48), (24, 48, 48, 96), (17, 32, 17, 32)]
        for n_src_lat, n_src_lon, n_tgt_lat, n_tgt_lon in cases:
            with self.subTest(src=(n_src_lat, n_src_lon), tgt=(n_tgt_lat, n_tgt_lon)):
                source_lat = np.linspace(90.0, -90.0, n_src_lat)
                source_lon = np.linspace(0.0, 360.0, n_src_lon, endpoint=False)
                target_lat = np.linspace(90.0, -90.0, n_tgt_lat)
                target_lon = np.linspace(0.0, 360.0, n_tgt_lon, endpoint=False)
                remap = ConservativeRemap.from_raster(source_lat, source_lon, target_lat, target_lon).to(self.device)

                field = torch.full((n_src_lat, n_src_lon), 3.7, device=self.device)
                out = remap(field)
                self.assertTrue(
                    compare_tensors("constant field", out, torch.full_like(out, 3.7), atol=1e-5, verbose=False)
                )

    def test_conservation(self):
        """The area-weighted integral of the field is unchanged by the remap.

        Exact (to double-precision rounding) for a raster source: the two
        weight matrices are true overlap areas normalised by the target's own
        extent, which is a mathematical identity, not an empirical property
        that could hold only approximately.
        """
        cases = [(48, 96, 24, 48), (24, 48, 48, 96)]
        for n_src_lat, n_src_lon, n_tgt_lat, n_tgt_lon in cases:
            with self.subTest(src=(n_src_lat, n_src_lon), tgt=(n_tgt_lat, n_tgt_lon)):
                source_lat = np.linspace(90.0, -90.0, n_src_lat)
                source_lon = np.linspace(0.0, 360.0, n_src_lon, endpoint=False)
                target_lat = np.linspace(90.0, -90.0, n_tgt_lat)
                target_lon = np.linspace(0.0, 360.0, n_tgt_lon, endpoint=False)
                remap = ConservativeRemap.from_raster(
                    source_lat, source_lon, target_lat, target_lon, dtype=torch.float64
                ).to(self.device)

                rng = np.random.default_rng(42)
                field = torch.as_tensor(
                    rng.normal(size=(n_src_lat, n_src_lon)), dtype=torch.float64, device=self.device
                )
                out = remap(field)

                src_area = torch.as_tensor(_raster_cell_areas(source_lat, source_lon), device=self.device)
                tgt_area = torch.as_tensor(_raster_cell_areas(target_lat, target_lon), device=self.device)

                total_in = (field * src_area).sum()
                total_out = (out * tgt_area).sum()
                self.assertTrue(
                    compare_tensors("conserved integral", total_out, total_in, atol=1e-9, rtol=1e-9, verbose=False)
                )

    def test_refine_takes_exact_value(self):
        """Where the target is finer than the source, a target cell just takes its source cell's value.

        ``(n_src_lat - 1) * 3 + 1`` target nodes make every target cell nest
        exactly inside one source cell (verified below via the weight
        matrices themselves being clean 0/1 indicators, rather than assumed
        -- the pole caps are half-width, so which source cell a target row
        lands in is not a uniform "repeat every row 3 times" pattern). Pole
        rows are excluded from the per-cell check: those are intentionally
        forced to their row's zonal mean regardless of nesting (see
        ``test_pole_rows_constant_across_longitude``), so nesting alone does
        not predict their value.
        """
        n_src_lat, n_src_lon = 4, 8
        n_tgt_lat, n_tgt_lon = (n_src_lat - 1) * 3 + 1, n_src_lon * 3
        source_lat = np.linspace(90.0, -90.0, n_src_lat)
        source_lon = np.linspace(0.0, 360.0, n_src_lon, endpoint=False)
        target_lat = np.linspace(90.0, -90.0, n_tgt_lat)
        target_lon = np.linspace(0.0, 360.0, n_tgt_lon, endpoint=False)
        remap = ConservativeRemap.from_raster(source_lat, source_lon, target_lat, target_lon, dtype=torch.float64).to(
            self.device
        )

        # exact nesting: every row of each weight matrix is a one-hot indicator
        lat_max, lat_idx = remap.latitude_weights.max(dim=1)
        lon_max, lon_idx = remap.longitude_weights.max(dim=1)
        self.assertTrue(
            compare_tensors("latitude weight row max", lat_max, torch.ones_like(lat_max), atol=1e-8, verbose=False)
        )
        self.assertTrue(
            compare_tensors("longitude weight row max", lon_max, torch.ones_like(lon_max), atol=1e-8, verbose=False)
        )

        field = torch.arange(n_src_lat * n_src_lon, dtype=torch.float64, device=self.device).reshape(
            n_src_lat, n_src_lon
        )
        out = remap(field)

        expected = field[lat_idx][:, lon_idx]
        non_pole = ~remap.pole_mask
        self.assertTrue(
            compare_tensors("nested cell value", out[non_pole], expected[non_pole], atol=1e-8, verbose=False)
        )

    def test_pole_rows_constant_across_longitude(self):
        """A target row sitting on a pole is one physical point -- every longitude must agree."""
        n_src_lat, n_src_lon, n_tgt_lat, n_tgt_lon = (
            33,
            64,
            17,
            32,
        )  # odd counts so a target row lands exactly on a pole
        source_lat = np.linspace(90.0, -90.0, n_src_lat)
        source_lon = np.linspace(0.0, 360.0, n_src_lon, endpoint=False)
        target_lat = np.linspace(90.0, -90.0, n_tgt_lat)
        target_lon = np.linspace(0.0, 360.0, n_tgt_lon, endpoint=False)
        remap = ConservativeRemap.from_raster(source_lat, source_lon, target_lat, target_lon).to(self.device)

        rng = np.random.default_rng(7)
        field = torch.as_tensor(rng.normal(size=(n_src_lat, n_src_lon)), dtype=torch.float32, device=self.device)
        out = remap(field)

        pole_rows = torch.as_tensor(np.flatnonzero(np.isclose(np.abs(target_lat), 90.0)))
        self.assertTrue(pole_rows.numel() > 0, "test setup should put a target row on a pole")
        for row in pole_rows.tolist():
            self.assertTrue(
                compare_tensors(f"pole row {row}", out[row], out[row, :1].expand_as(out[row]), atol=1e-5, verbose=False)
            )


@parameterized_class(("device",), _devices)
class TestMeshProperties(unittest.TestCase):
    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_consistency_constant_field(self):
        """A constant field in gives the same constant back out, including through the empty-target fallback."""
        n_cells, n_lat, n_lon = 20000, 24, 48
        cell_lat, cell_lon = _fibonacci_mesh(n_cells)
        cell_area = np.full(n_cells, 4.0 * np.pi / n_cells)
        target_lat = np.linspace(90.0, -90.0, n_lat)
        target_lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
        remap = ConservativeRemap.from_mesh(cell_lat, cell_lon, cell_area, target_lat, target_lon).to(self.device)

        values = torch.full((n_cells,), -2.4, device=self.device)
        out = remap(values)
        self.assertTrue(compare_tensors("constant field", out, torch.full_like(out, -2.4), atol=1e-5, verbose=False))

    def test_conservation(self):
        """The area-weighted integral is unchanged, when every target cell catches at least one mesh cell.

        Two different notions of "target area" are worth distinguishing here.
        A mesh source has no predetermined geometric target-cell area the way
        a raster source's weight matrices do -- ``target_weight`` (the actual
        divisor the operator uses) is the *caught mesh-cell area*, which only
        converges to a target cell's true geometric area as the mesh gets
        dense relative to the target grid. So there are two checks:

        * exact (to float64 rounding), against ``target_weight`` -- this is
          the identity the scatter-add-then-divide arithmetic guarantees by
          construction, and is what would actually catch a regression (an
          off-by-one target index, a double-counted cell, a broken pole
          reduce);
        * approximate, against the true geometric target-cell area -- this
          checks the operator is actually doing the right *physical* thing,
          not just being internally consistent, but can only ever be
          approximate for a finite mesh (checked loosely, at a tolerance
          matching the expected O(1/sqrt(cells per target cell)) discretisation
          noise -- not exact, and not the primary regression guard above).
        """
        n_cells, n_lat, n_lon = 20000, 16, 32
        cell_lat, cell_lon = _fibonacci_mesh(n_cells)
        cell_area = np.full(n_cells, 4.0 * np.pi / n_cells)
        target_lat = np.linspace(90.0, -90.0, n_lat)
        target_lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
        remap = ConservativeRemap.from_mesh(
            cell_lat, cell_lon, cell_area, target_lat, target_lon, dtype=torch.float64
        ).to(self.device)
        self.assertEqual(remap.fallback_fraction, 0.0, "test setup should leave no target cell empty")

        rng = np.random.default_rng(11)
        values = torch.as_tensor(rng.normal(size=n_cells), dtype=torch.float64, device=self.device)
        out = remap(values)

        mesh_area = torch.as_tensor(cell_area, dtype=torch.float64, device=self.device)
        total_in = (values * mesh_area).sum()

        total_out_exact = (out.reshape(-1) * remap.target_weight).sum()
        self.assertTrue(
            compare_tensors(
                "conserved integral (exact, vs. target_weight)",
                total_out_exact,
                total_in,
                atol=1e-9,
                rtol=1e-9,
                verbose=False,
            )
        )

        tgt_area = torch.as_tensor(_raster_cell_areas(target_lat, target_lon), dtype=torch.float64, device=self.device)
        total_out_geometric = (out * tgt_area).sum()
        cells_per_target = n_cells / (n_lat * n_lon)
        self.assertTrue(
            compare_tensors(
                "conserved integral (approximate, vs. geometric area)",
                total_out_geometric,
                total_in,
                atol=0.0,
                rtol=10.0 / np.sqrt(cells_per_target),
                verbose=False,
            )
        )

    def test_pole_rows_constant_across_longitude(self):
        """A target row sitting on a pole is one physical point -- every longitude must agree."""
        n_cells, n_lat, n_lon = 20000, 17, 32  # odd n_lat so a target row lands exactly on a pole
        cell_lat, cell_lon = _fibonacci_mesh(n_cells)
        cell_area = np.full(n_cells, 4.0 * np.pi / n_cells)
        target_lat = np.linspace(90.0, -90.0, n_lat)
        target_lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
        remap = ConservativeRemap.from_mesh(cell_lat, cell_lon, cell_area, target_lat, target_lon).to(self.device)

        rng = np.random.default_rng(13)
        values = torch.as_tensor(rng.normal(size=n_cells), dtype=torch.float32, device=self.device)
        out = remap(values)

        pole_rows = torch.as_tensor(np.flatnonzero(np.isclose(np.abs(target_lat), 90.0)))
        self.assertTrue(pole_rows.numel() > 0, "test setup should put a target row on a pole")
        for row in pole_rows.tolist():
            self.assertTrue(
                compare_tensors(f"pole row {row}", out[row], out[row, :1].expand_as(out[row]), atol=1e-5, verbose=False)
            )


class TestMeshConstructionMetadata(unittest.TestCase):
    """Construction-time attributes with no forward pass involved -- not device-relevant."""

    def setUp(self):
        set_seed(333)

    def test_fallback_fraction_reflects_target_resolution(self):
        """A target much finer than the mesh leaves cells empty; a coarse one does not."""
        n_cells = 2000
        cell_lat, cell_lon = _fibonacci_mesh(n_cells)
        cell_area = np.full(n_cells, 4.0 * np.pi / n_cells)

        coarse_lat, coarse_lon = np.linspace(90.0, -90.0, 8), np.linspace(0.0, 360.0, 16, endpoint=False)
        coarse = ConservativeRemap.from_mesh(cell_lat, cell_lon, cell_area, coarse_lat, coarse_lon)
        self.assertEqual(coarse.fallback_fraction, 0.0)

        fine_lat, fine_lon = np.linspace(90.0, -90.0, 200), np.linspace(0.0, 360.0, 400, endpoint=False)
        fine = ConservativeRemap.from_mesh(cell_lat, cell_lon, cell_area, fine_lat, fine_lon)
        self.assertGreater(fine.fallback_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
