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

import unittest
import tempfile

import h5py
import numpy as np
import torch

from makani.models.preprocessor_helpers import get_bias_correction, get_static_features
from makani.utils.features import get_auxiliary_channels, get_channel_groups, is_static_aux_channel

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import set_seed, get_default_parameters

# Grid dimensions used across all tests — small enough to be fast on CPU.
IMG_H = 8
IMG_W = 16


# ===========================================================================
# 1. get_bias_correction
# ===========================================================================
class TestGetBiasCorrection(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = IMG_H
        self.params.img_shape_y = IMG_W

    def test_no_bias_correction_returns_none(self):
        """Returns None when 'bias_correction' is absent from params."""
        self.assertIsNone(get_bias_correction(self.params))

    def test_missing_file_raises_ioerror(self):
        """Raises IOError when bias_correction points to a non-existent file."""
        self.params.bias_correction = "/nonexistent/bias.npy"
        with self.assertRaises(IOError):
            get_bias_correction(self.params)


# ===========================================================================
# 2. get_static_features
# ===========================================================================
class TestGetStaticFeatures(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = IMG_H
        self.params.img_shape_y = IMG_W

    # -----------------------------------------------------------------------
    # 2a. Trivial / None case
    # -----------------------------------------------------------------------
    def test_no_features_returns_none(self):
        """Returns None when all feature flags are False (default state)."""
        self.assertIsNone(get_static_features(self.params))

    # -----------------------------------------------------------------------
    # 2b. add_grid — raw gridtype
    # -----------------------------------------------------------------------
    def test_add_grid_raw_shape(self):
        """Raw grid produces a (1, 2, H, W) tensor: one channel per spatial dim."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        out = get_static_features(self.params)
        self.assertIsNotNone(out)
        self.assertEqual(tuple(out.shape), (1, 2, IMG_H, IMG_W))

    def test_add_grid_raw_values_in_unit_interval(self):
        """Raw grid coordinates come from linspace, so all values are in [0, 1)."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        out = get_static_features(self.params)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLess(out.max().item(), 1.0)

    # -----------------------------------------------------------------------
    # 2c. add_grid — sinusoidal gridtype, channel-count arithmetic
    # -----------------------------------------------------------------------
    def test_add_grid_sinusoidal_1freq_with_cos_channels(self):
        """num_freq=1, add_cos=True: {sin,cos}(grid) × 2 spatial dims → 4 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 1
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 4, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_1freq_no_cos_channels(self):
        """num_freq=1, add_cos=False: sin(grid) × 2 spatial dims → 2 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 1
        self.params.add_cos_to_grid = False
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 2, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_2freq_with_cos_channels(self):
        """num_freq=2, add_cos=True: {sin,cos}(k*grid) k=1,2 × 2 dims → 8 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 2
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 8, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_2freq_no_cos_channels(self):
        """num_freq=2, add_cos=False: sin(k*grid) k=1,2 × 2 dims → 4 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 2
        self.params.add_cos_to_grid = False
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 4, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_values_bounded(self):
        """All sinusoidal-encoded values must lie in [-1, 1]."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 2
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertGreaterEqual(out.min().item(), -1.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_add_grid_sinusoidal_finite(self):
        """Sinusoidal grid features must be finite (no NaN or Inf)."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 1
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    # -----------------------------------------------------------------------
    # 2d. Spatial sharding
    # -----------------------------------------------------------------------
    def test_add_grid_spatial_sharding_height(self):
        """img_local_offset_x / img_local_shape_x slice the H dimension."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.img_local_offset_x = 2
        self.params.img_local_shape_x = 4  # rows 2..5 → 4 rows
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], 4)
        self.assertEqual(out.shape[-1], IMG_W)

    def test_add_grid_spatial_sharding_width(self):
        """img_local_offset_y / img_local_shape_y slice the W dimension."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.img_local_offset_y = 4
        self.params.img_local_shape_y = 8  # cols 4..11 → 8 cols
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H)
        self.assertEqual(out.shape[-1], 8)

    def test_add_grid_sharding_clamped_to_grid_boundary(self):
        """offset + local_shape > grid_size is clamped: only the remaining rows."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.img_local_offset_x = 6
        self.params.img_local_shape_x = 8  # would overshoot → clamped to 2 rows
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H - 6)

    # -----------------------------------------------------------------------
    # 2e. Subsampling
    # -----------------------------------------------------------------------
    def test_add_grid_subsampling_halves_spatial_dims(self):
        """subsampling_factor=2 halves both spatial dimensions."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.subsampling_factor = 2
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H // 2)
        self.assertEqual(out.shape[-1], IMG_W // 2)

    def test_add_grid_subsampling_factor_1_unchanged(self):
        """subsampling_factor=1 leaves both spatial dimensions unchanged."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.subsampling_factor = 1
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H)
        self.assertEqual(out.shape[-1], IMG_W)

    # -----------------------------------------------------------------------
    # 2f. IOError guards for file-dependent features
    # -----------------------------------------------------------------------
    def test_invariants_missing_file_raises_ioerror(self):
        """Raises IOError when invariants_path does not exist."""
        self.params.invariants = [{"channel": "z", "encoding": "normalize"}]
        self.params.invariants_path = "/nonexistent/invariants.h5"
        with self.assertRaises(IOError):
            get_static_features(self.params)

    def test_legacy_invariant_options_raise(self):
        """The discontinued per-file options are refused rather than silently ignored."""
        for option in ["add_orography", "add_landmask", "add_soiltype"]:
            with self.subTest(option=option):
                params = get_default_parameters()
                params.img_shape_x = IMG_H
                params.img_shape_y = IMG_W
                setattr(params, option, True)
                with self.assertRaises(ValueError):
                    get_static_features(params)
                with self.assertRaises(ValueError):
                    get_auxiliary_channels(**params.to_dict())

    def test_add_copernicus_emb_missing_file_raises_ioerror(self):
        """Raises IOError when copernicus_emb_path does not exist."""
        self.params.add_copernicus_emb = True
        self.params.copernicus_emb_path = "/nonexistent/copernicus.npy"
        with self.assertRaises(IOError):
            get_static_features(self.params)


# ===========================================================================
# 3. Invariants read from an invariants file
# ===========================================================================
_LSM = [{"channel": "lsm", "encoding": "onehot", "num_classes": 2, "rounding": "floor"}]


def _write_invariants_file(path, fields, channels, valid=None):
    """An invariants file laid out like data_process/convert_era5_invariants_to_makani_input.py writes it."""
    with h5py.File(path, "w") as f:
        f["fields"] = np.asarray(fields, dtype=np.float32)
        f["channel"] = np.array(channels, dtype="S")
        f["lat"] = np.linspace(90.0, -90.0, IMG_H)
        f["lon"] = np.linspace(0.0, 360.0, IMG_W, endpoint=False)
        f["valid_data"] = np.ones(len(channels), dtype=np.int32) if valid is None else np.asarray(valid)


class TestInvariantFeatures(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = IMG_H
        self.params.img_shape_y = IMG_W

        rng = np.random.default_rng(333)
        self.z = rng.normal(size=(IMG_H, IMG_W)) * 1000.0
        # fractional land, so that floor and round differ
        self.lsm = rng.uniform(size=(IMG_H, IMG_W))
        self.lsm[0, :] = 1.0
        self.slt = rng.integers(0, 8, size=(IMG_H, IMG_W)).astype(np.float32)

        self._tmpdir = tempfile.TemporaryDirectory()
        self.params.invariants_path = os.path.join(self._tmpdir.name, "invariants.h5")
        _write_invariants_file(self.params.invariants_path, [self.z, self.lsm, self.slt], ["z", "lsm", "slt"])

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_channels_follow_the_configured_order(self):
        self.params.invariants = [
            {"channel": "slt", "encoding": "raw"},
            {"channel": "z", "encoding": "normalize"},
        ]
        out = get_static_features(self.params)

        self.assertEqual(tuple(out.shape), (1, 2, IMG_H, IMG_W))
        torch.testing.assert_close(out[0, 0], torch.as_tensor(self.slt, dtype=torch.float32))
        self.assertAlmostEqual(out[0, 1].mean().item(), 0.0, places=5)
        self.assertAlmostEqual(out[0, 1].std().item(), 1.0, places=4)

    def test_onehot_floor_marks_only_full_land(self):
        self.params.invariants = _LSM
        out = get_static_features(self.params)

        # class 1, land, only where the land fraction is exactly one
        land = torch.as_tensor(self.lsm == 1.0, dtype=torch.float32)
        torch.testing.assert_close(out[0, 1], land)
        torch.testing.assert_close(out[0, 0], 1.0 - land)

    def test_onehot_round(self):
        self.params.invariants = [{"channel": "lsm", "encoding": "onehot", "num_classes": 2}]
        out = get_static_features(self.params)

        land = torch.as_tensor(np.round(self.lsm) == 1.0, dtype=torch.float32)
        torch.testing.assert_close(out[0, 1], land)

    def test_onehot_classes_out_of_range_raise(self):
        self.params.invariants = [{"channel": "slt", "encoding": "onehot", "num_classes": 4}]
        with self.assertRaises(ValueError):
            get_static_features(self.params)

    def test_channel_absent_from_file_raises(self):
        self.params.invariants = [{"channel": "cvh", "encoding": "raw"}]
        with self.assertRaises(ValueError):
            get_static_features(self.params)

    def test_channel_marked_invalid_raises(self):
        _write_invariants_file(self.params.invariants_path, [self.z], ["z"], valid=[0])
        self.params.invariants = [{"channel": "z", "encoding": "normalize"}]
        with self.assertRaises(ValueError):
            get_static_features(self.params)

    def test_grid_mismatch_raises(self):
        self.params.img_shape_x = IMG_H * 2
        self.params.invariants = [{"channel": "z", "encoding": "normalize"}]
        with self.assertRaises(ValueError):
            get_static_features(self.params)

    def test_invalid_configuration_raises(self):
        for invariants in [
            [{"channel": "lsm", "encoding": "onehot"}],
            [{"channel": "z", "encoding": "zscore"}],
            [{"channel": "z", "encoding": "raw", "num_classes": 2}],
            [{"channel": "z", "encoding": "raw"}, {"channel": "z", "encoding": "normalize"}],
        ]:
            with self.subTest(invariants=invariants):
                self.params.invariants = invariants
                with self.assertRaises(ValueError):
                    get_static_features(self.params)

    def test_sharding_and_subsampling(self):
        self.params.invariants = [{"channel": "slt", "encoding": "raw"}]
        self.params.img_local_offset_x = 2
        self.params.img_local_shape_x = 4
        self.params.subsampling_factor = 2
        out = get_static_features(self.params)

        torch.testing.assert_close(out[0, 0], torch.as_tensor(self.slt[2:6:2, ::2], dtype=torch.float32))


# ===========================================================================
# 4. Synthetic-data stand-ins for the invariants
# ===========================================================================
class TestSyntheticStaticFeatures(unittest.TestCase):
    """On synthetic data the invariants file does not exist, so it is stood in for."""

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = IMG_H
        self.params.img_shape_y = IMG_W
        self.params.enable_synthetic_data = True
        self.params.invariants_path = "/nonexistent/invariants.h5"

    def test_invariants(self):
        """The fcn3 configuration: orography plus a one-hot land-sea mask."""
        self.params.invariants = [{"channel": "z", "encoding": "normalize"}] + _LSM

        out = get_static_features(self.params)

        self.assertEqual(tuple(out.shape), (1, 3, IMG_H, IMG_W))
        # one-hot: exactly one channel is set per grid point
        torch.testing.assert_close(out[:, 1:].sum(dim=1), torch.ones(1, IMG_H, IMG_W))

    def test_copernicus_emb(self):
        self.params.add_copernicus_emb = True
        self.params.copernicus_emb_path = "/nonexistent/copernicus.npy"

        out = get_static_features(self.params)

        self.assertEqual(tuple(out.shape), (1, 8, IMG_H, IMG_W))

    def test_real_data_still_raises(self):
        """Without synthetic data a missing invariants file is still a hard error."""
        self.params.enable_synthetic_data = False
        self.params.invariants = [{"channel": "z", "encoding": "normalize"}]

        with self.assertRaises(IOError):
            get_static_features(self.params)


# ===========================================================================
# 5. Auxiliary channel names against the static features
# ===========================================================================
class TestAuxiliaryChannelNames(unittest.TestCase):
    """The driver counts the static channels from their names, so the names have to match the tensor."""

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = IMG_H
        self.params.img_shape_y = IMG_W
        self.params.enable_synthetic_data = True
        self.params.invariants_path = "/nonexistent/invariants.h5"

    def _check(self):
        names = get_auxiliary_channels(**self.params.to_dict())
        out = get_static_features(self.params)
        num_static = 0 if out is None else out.shape[1]
        self.assertEqual(len([name for name in names if is_static_aux_channel(name)]), num_static)
        return names

    def test_static_counts_match(self):
        configurations = [
            dict(add_grid=True, gridtype="raw"),
            dict(add_grid=True, gridtype="sinusoidal", grid_num_frequencies=2, add_cos_to_grid=True),
            dict(add_grid=True, gridtype="sinusoidal", grid_num_frequencies=2, add_cos_to_grid=False),
            dict(invariants=[{"channel": "z", "encoding": "normalize"}] + _LSM),
            dict(add_copernicus_emb=True, copernicus_emb_path="/nonexistent/copernicus.npy"),
        ]
        for configuration in configurations:
            with self.subTest(**{k: str(v) for k, v in configuration.items()}):
                self.setUp()
                for key, value in configuration.items():
                    setattr(self.params, key, value)
                self._check()

    def test_names_and_order(self):
        self.params.add_zenith = True
        self.params.n_noise_chan = 2
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.invariants = [{"channel": "z", "encoding": "normalize"}] + _LSM

        names = self._check()

        self.assertEqual(names, ["xd_zen", "xd_noise0", "xd_noise1", "xs_lat", "xs_lon", "xs_z", "xs_lsm0", "xs_lsm1"])

    def test_channel_groups_split_by_prefix(self):
        _, _, dyn, stat, _ = get_channel_groups(["t2m"], ["xd_zen", "xs_lat", "xs_lon", "xs_lsm1"])
        self.assertEqual(dyn, [1])
        self.assertEqual(stat, [2, 3, 4])

    def test_channel_groups_reject_unprefixed_aux_channels(self):
        with self.assertRaises(ValueError):
            get_channel_groups(["t2m"], ["xzen"])


if __name__ == "__main__":
    unittest.main()
