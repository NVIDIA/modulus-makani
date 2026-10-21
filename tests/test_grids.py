# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import os
import sys
import unittest

import numpy as np
import torch
from parameterized import parameterized

from makani.utils.grid_types import expected_latitudes, matching_grid_types, verify_grid_type
from makani.utils.grids import grid_to_quadrature_rule, GridConverter, GridQuadrature, BandLimitMask

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

# Small grid that runs fast on CPU
_H, _W = 64, 128


# ---------------------------------------------------------------------------
# grid_to_quadrature_rule
# ---------------------------------------------------------------------------


class TestGridToQuadratureRule(unittest.TestCase):

    @parameterized.expand(
        [
            ("euclidean", "uniform"),
            ("equiangular", "naive"),
            ("legendre-gauss", "legendre-gauss"),
            ("clenshaw-curtiss", "clenshaw-curtiss"),
            ("weatherbench2", "weatherbench2"),
        ]
    )
    def test_known_grid_types(self, grid_type, expected_rule):
        self.assertEqual(grid_to_quadrature_rule(grid_type), expected_rule)

    def test_unknown_grid_type_raises(self):
        with self.assertRaises(NotImplementedError):
            grid_to_quadrature_rule("unknown_grid")


# ---------------------------------------------------------------------------
# GridConverter
# ---------------------------------------------------------------------------


class TestGridConverter(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.H, self.W = _H, _W
        # equiangular source grid: north (+π/2) to south (-π/2), longitudes [0, 2π)
        self.lat_rad = torch.linspace(math.pi / 2, -math.pi / 2, self.H)
        self.lon_rad = torch.arange(self.W) * (2.0 * math.pi / self.W)

    def test_passthrough_same_grid_returns_input(self):
        """forward() is the identity (same object) when src == dst."""
        converter = GridConverter("equiangular", "equiangular", self.lat_rad, self.lon_rad)
        x = torch.randn(2, 3, self.H, self.W)
        out = converter(x)
        self.assertIs(out, x)

    def test_passthrough_coords_unchanged(self):
        """src == dst: dst coords are equal to src coords."""
        converter = GridConverter("equiangular", "equiangular", self.lat_rad, self.lon_rad)
        src_lat, src_lon = converter.get_src_coords()
        dst_lat, dst_lon = converter.get_dst_coords()
        self.assertTrue(torch.equal(src_lat, dst_lat))
        self.assertTrue(torch.equal(src_lon, dst_lon))

    def test_equiangular_to_legendre_gauss_output_shape(self):
        """equiangular → legendre-gauss output has the same spatial shape as input."""
        converter = GridConverter("equiangular", "legendre-gauss", self.lat_rad, self.lon_rad)
        x = torch.randn(2, 3, self.H, self.W)
        out = converter(x)
        self.assertEqual(out.shape, (2, 3, self.H, self.W))

    def test_equiangular_to_legendre_gauss_dst_lats_within_src_range(self):
        """Legendre-Gauss destination nodes must lie strictly inside the source lat range."""
        converter = GridConverter("equiangular", "legendre-gauss", self.lat_rad, self.lon_rad)
        dst_lat, _ = converter.get_dst_coords()
        self.assertTrue((dst_lat > self.lat_rad[-1]).all(), "dst_lat has values at or below the south pole")
        self.assertTrue((dst_lat < self.lat_rad[0]).all(), "dst_lat has values at or above the north pole")

    def test_equiangular_to_legendre_gauss_linear_function_exact(self, verbose=False):
        """Linear interpolation recovers a latitude-linear field exactly."""
        converter = GridConverter("equiangular", "legendre-gauss", self.lat_rad, self.lon_rad)

        # Field whose value at row i is lat_rad[i], constant across lon
        data = self.lat_rad.unsqueeze(-1).expand(self.H, self.W)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        out = converter(data)  # (1, 1, H_dst, W)

        dst_lat, _ = converter.get_dst_coords()
        expected = dst_lat.unsqueeze(-1).expand(-1, self.W).unsqueeze(0).unsqueeze(0).to(out.dtype)

        self.assertTrue(
            compare_tensors("linear function interpolation", out, expected, atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    def test_unsupported_dst_raises(self):
        """An unsupported destination grid raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            GridConverter("equiangular", "unsupported_dst", self.lat_rad, self.lon_rad)


# ---------------------------------------------------------------------------
# GridQuadrature
# ---------------------------------------------------------------------------


class TestGridQuadrature(unittest.TestCase):

    def setUp(self):
        disable_tf32()

    # --- weight correctness --------------------------------------------------

    @parameterized.expand(
        [
            ("naive",),
            ("clenshaw-curtiss",),
            ("legendre-gauss",),
            ("weatherbench2",),
            ("uniform",),
        ]
    )
    def test_weights_sum_to_4pi(self, quadrature_rule):
        """Quadrature weights must integrate the constant function 1 to 4π (unit-sphere area)."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W))
        total = gq.quad_weight.sum().item()
        self.assertAlmostEqual(
            total,
            4.0 * math.pi,
            delta=1e-4,
            msg=f"{quadrature_rule}: weights sum to {total:.6f}, expected 4π≈{4*math.pi:.6f}",
        )

    @parameterized.expand(
        [
            ("naive",),
            ("clenshaw-curtiss",),
            ("legendre-gauss",),
            ("weatherbench2",),
            ("uniform",),
        ]
    )
    def test_weights_sum_to_one_when_normalized(self, quadrature_rule):
        """With normalize=True the weights must sum to 1."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W), normalize=True)
        total = gq.quad_weight.sum().item()
        self.assertAlmostEqual(
            total,
            1.0,
            delta=1e-5,
            msg=f"{quadrature_rule} normalized: weights sum to {total:.8f}, expected 1.0",
        )

    @parameterized.expand(
        [
            ("naive",),
            ("clenshaw-curtiss",),
            ("legendre-gauss",),
            ("weatherbench2",),
            ("uniform",),
        ]
    )
    def test_weights_all_non_negative(self, quadrature_rule):
        """All quadrature weights must be non-negative."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W))
        self.assertTrue(
            (gq.quad_weight >= 0).all(),
            msg=f"{quadrature_rule}: negative weights found",
        )

    # --- forward pass --------------------------------------------------------

    @parameterized.expand(
        [
            ("naive",),
            ("clenshaw-curtiss",),
            ("legendre-gauss",),
            ("weatherbench2",),
            ("uniform",),
        ]
    )
    def test_forward_constant_one_integrates_to_4pi(self, quadrature_rule, verbose=False):
        """Integrating f≡1 over the sphere yields 4π for every quadrature rule."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W))
        B, C = 2, 3
        x = torch.ones(B, C, _H, _W)
        out = gq(x)
        expected = torch.full((B, C), 4.0 * math.pi)
        self.assertTrue(
            compare_tensors(
                f"constant integral ({quadrature_rule})",
                out,
                expected,
                atol=1e-3,
                rtol=1e-4,
                verbose=verbose,
            )
        )

    def test_forward_output_shape(self):
        """forward() reduces the spatial (H, W) dims and returns (B, C)."""
        gq = GridQuadrature("legendre-gauss", img_shape=(_H, _W))
        x = torch.randn(4, 5, _H, _W)
        out = gq(x)
        self.assertEqual(out.shape, (4, 5))

    def test_forward_linearity(self, verbose=False):
        """forward() must be linear: Q(a*x + b*y) == a*Q(x) + b*Q(y)."""
        gq = GridQuadrature("legendre-gauss", img_shape=(_H, _W))
        set_seed(333)
        x = torch.randn(2, 3, _H, _W)
        y = torch.randn(2, 3, _H, _W)
        a, b = 2.5, -1.3
        lhs = gq(a * x + b * y)
        rhs = a * gq(x) + b * gq(y)
        self.assertTrue(compare_tensors("linearity", lhs, rhs, atol=1e-4, rtol=1e-5, verbose=verbose))

    # --- construction edge cases ---------------------------------------------

    def test_unknown_rule_raises(self):
        """An unknown quadrature rule raises ValueError."""
        with self.assertRaises(ValueError):
            GridQuadrature("unknown_rule", img_shape=(_H, _W))

    def test_crop_reduces_total_weight(self):
        """A spatially cropped quadrature has strictly less total weight than the full sphere."""
        gq_full = GridQuadrature("legendre-gauss", img_shape=(_H, _W))
        gq_crop = GridQuadrature(
            "legendre-gauss",
            img_shape=(_H, _W),
            crop_shape=(_H // 2, _W),
            crop_offset=(0, 0),
        )
        total_full = gq_full.quad_weight.sum().item()
        total_crop = gq_crop.quad_weight.sum().item()
        self.assertGreater(total_crop, 0.0)
        self.assertLess(total_crop, total_full)

    def test_crop_shape_equals_img_shape_matches_full(self, verbose=False):
        """crop_shape == img_shape (with zero offset) is equivalent to no crop."""
        gq_full = GridQuadrature("clenshaw-curtiss", img_shape=(_H, _W))
        gq_crop = GridQuadrature(
            "clenshaw-curtiss",
            img_shape=(_H, _W),
            crop_shape=(_H, _W),
            crop_offset=(0, 0),
        )
        self.assertTrue(
            compare_tensors(
                "full crop vs no crop", gq_full.quad_weight, gq_crop.quad_weight, atol=0.0, rtol=0.0, verbose=verbose
            )
        )


class TestGridTypeVerification(unittest.TestCase):
    """Checking that a declared grid type matches the coordinates it came with.

    The declaration decides the quadrature. Weights that do not belong to the
    grid mis-weight every area averaged loss and metric, by an amount that looks
    like a modelling difference rather than a bug -- and since the coordinates
    are always available, the claim never has to be taken on trust.
    """

    def test_equiangular_latitudes_span_pole_to_pole(self):
        latitudes = expected_latitudes("equiangular", 5)
        self.assertTrue(np.allclose(latitudes, [-90.0, -45.0, 0.0, 45.0, 90.0]))

    def test_the_equally_spaced_grids_share_their_nodes(self):
        """Three grid types, one set of nodes.

        Clenshaw-Curtiss and WeatherBench2 both sit on the equally spaced
        latitudes and differ from equiangular only in how they weight them. So
        no coordinate can separate the three, and anything claiming to identify
        a grid from its latitudes has to say so rather than pick one.
        """
        for grid_type in ("clenshaw-curtiss", "weatherbench2"):
            with self.subTest(grid_type=grid_type):
                self.assertTrue(np.allclose(expected_latitudes(grid_type, 9), expected_latitudes("equiangular", 9)))

    def test_gauss_nodes_are_not_the_equiangular_ones(self):
        # the premise of the whole check: the grids are distinguishable
        gauss = expected_latitudes("legendre-gauss", 32)
        equiangular = expected_latitudes("equiangular", 32)

        self.assertGreater(np.max(np.abs(gauss - equiangular)), 1.0)

    def test_gauss_nodes_stay_inside_the_poles(self):
        gauss = expected_latitudes("legendre-gauss", 16)

        self.assertGreater(gauss[0], -90.0)
        self.assertLess(gauss[-1], 90.0)

    def test_a_grid_off_the_sphere_has_no_latitudes(self):
        self.assertIsNone(expected_latitudes("euclidean", 8))

    def test_an_unknown_grid_type_is_refused(self):
        with self.assertRaises(NotImplementedError):
            expected_latitudes("cubed-sphere", 8)

    def test_equally_spaced_nodes_match_every_grid_that_uses_them(self):
        # all three, because the coordinates genuinely do not distinguish them
        self.assertEqual(
            matching_grid_types(expected_latitudes("equiangular", 24)),
            ["equiangular", "clenshaw-curtiss", "weatherbench2"],
        )

    def test_gauss_nodes_match_only_the_gauss_grid(self):
        # the one separation coordinates can make, and the one worth making
        self.assertEqual(matching_grid_types(expected_latitudes("legendre-gauss", 24)), ["legendre-gauss"])

    def test_latitudes_that_are_no_known_grid_match_nothing(self):
        self.assertEqual(matching_grid_types(np.linspace(-30.0, 30.0, 12)), [])

    def test_a_truthful_declaration_passes(self):
        for grid_type in ("equiangular", "legendre-gauss", "clenshaw-curtiss", "weatherbench2"):
            with self.subTest(grid_type=grid_type):
                verify_grid_type(grid_type, expected_latitudes(grid_type, 24))

    def test_orientation_is_not_part_of_the_claim(self):
        # makani stores latitudes north to south, the quadrature routines
        # produce them south to north; what is compared is where the nodes are
        verify_grid_type("legendre-gauss", np.flip(expected_latitudes("legendre-gauss", 24)))

    def test_a_false_declaration_is_refused_and_says_what_it_looks_like(self):
        with self.assertRaises(ValueError) as ctx:
            verify_grid_type("legendre-gauss", expected_latitudes("equiangular", 24), source="data.json")

        message = str(ctx.exception)
        self.assertIn("data.json", message)
        self.assertIn("legendre-gauss", message)
        self.assertIn("equiangular", message)

    def test_a_swap_within_the_equally_spaced_grids_cannot_be_caught(self):
        """Stated as a test because it is a limit, not an oversight.

        Declaring Clenshaw-Curtiss over equiangular nodes is accepted: the nodes
        are identical and the coordinates hold no evidence either way. Only the
        weights differ, and no dataset carries those.
        """
        verify_grid_type("clenshaw-curtiss", expected_latitudes("equiangular", 24))
        verify_grid_type("equiangular", expected_latitudes("clenshaw-curtiss", 24))

    def test_float32_coordinates_still_verify(self):
        # coordinates come out of files as float32, so the tolerance has to
        # absorb that while still separating the grids
        latitudes = expected_latitudes("equiangular", 721).astype(np.float32).astype(np.float64)
        verify_grid_type("equiangular", latitudes)

    def test_a_grid_off_the_sphere_is_not_checked(self):
        verify_grid_type("euclidean", np.linspace(0.0, 1.0, 8))


# ---------------------------------------------------------------------------
# BandLimitMask
# ---------------------------------------------------------------------------


class TestBandLimitMask(unittest.TestCase):
    """BandLimitMask keeps the modes below the band limit and removes the ones above."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_sht_removes_modes_above_lmax(self, verbose=False):
        import torch_harmonics as th

        nlat, nlon, lmax = 16, 32, 6
        isht = th.InverseRealSHT(nlat, nlon, grid="legendre-gauss")

        # random real field with modes up to the grid's full band limit
        coeffs = torch.randn(2, 3, isht.lmax, isht.mmax, dtype=torch.complex64)
        ls = torch.arange(isht.lmax).reshape(-1, 1)
        ms = torch.arange(isht.mmax).reshape(1, -1)
        coeffs = torch.where(ms <= ls, coeffs, torch.zeros_like(coeffs))
        coeffs[..., 0] = coeffs[..., 0].real.to(coeffs.dtype)
        low = torch.where(ls < lmax, coeffs, torch.zeros_like(coeffs))

        field = isht(coeffs)
        expected = isht(low)

        mask = BandLimitMask((nlat, nlon), "legendre-gauss", lmax=lmax, type="sht")
        self.assertTrue(compare_tensors("band limited", mask(field), expected, atol=1e-5, rtol=1e-5, verbose=verbose))
        # a field that is already band limited passes unchanged
        self.assertTrue(compare_tensors("unchanged", mask(expected), expected, atol=1e-5, rtol=1e-5, verbose=verbose))

    def test_fft_cutoff_shrinks_towards_the_poles(self, verbose=False):
        from torch_harmonics.quadrature import precompute_latitudes

        nlat, nlon, lmax, m = 16, 32, 8, 3
        lon = torch.arange(nlon, dtype=torch.float32) * 2 * math.pi / nlon
        wave = torch.cos(m * lon).reshape(1, 1, 1, nlon).expand(1, 1, nlat, nlon)
        constant = torch.ones(1, 1, nlat, nlon)

        mask = BandLimitMask((nlat, nlon), "equiangular", lmax=lmax, type="fft")

        # the zonal mean is kept on every latitude
        self.assertTrue(compare_tensors("m=0", mask(constant), constant, atol=1e-5, verbose=verbose))

        # wavenumber m is kept where the latitude circle resolves it at the equator's grid spacing
        colats, _ = precompute_latitudes(nlat, grid="equiangular")
        keep = m <= torch.ceil((lmax - 1) * torch.sin(colats.float()))
        self.assertTrue(keep.any() and not keep.all(), "the test should cover kept and removed latitudes")
        expected = torch.where(keep.reshape(1, 1, nlat, 1), wave, torch.zeros_like(wave))
        self.assertTrue(compare_tensors(f"m={m}", mask(wave), expected, atol=1e-5, verbose=verbose))

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            BandLimitMask((16, 32), "equiangular", lmax=8, type="wavelet")


if __name__ == "__main__":
    unittest.main()
