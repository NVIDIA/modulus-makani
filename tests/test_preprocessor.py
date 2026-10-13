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

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
import torch

from makani.models.preprocessor import Preprocessor2D

from .testutils import set_seed, get_default_parameters, compare_tensors, IMG_SIZE_H, IMG_SIZE_W


class TestPreprocessor2DBasic(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.pp = Preprocessor2D(self.params)
        self.B = self.params.batch_size  # 1
        self.C = self.params.N_in_channels  # NUM_CHANNELS
        self.H = self.params.img_shape_x  # IMG_SIZE_H
        self.W = self.params.img_shape_y  # IMG_SIZE_W

    def _rand(self, *shape):
        return torch.randn(*shape)

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def test_construction_does_not_raise(self):
        """Minimal params construct a Preprocessor2D without error."""
        pp = Preprocessor2D(get_default_parameters())
        self.assertIsInstance(pp, Preprocessor2D)

    # -----------------------------------------------------------------------
    # flatten_history / expand_history
    # -----------------------------------------------------------------------

    def test_flatten_history_4d_noop(self, verbose=False):
        """flatten_history on a 4-D tensor returns it unchanged."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.flatten_history(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(compare_tensors("flatten_history 4D noop", out, x, verbose=verbose))

    def test_flatten_history_5d_collapses_time(self):
        """flatten_history on (B,T,C,H,W) gives (B, T*C, H, W)."""
        T = 3
        x = self._rand(self.B, T, self.C, self.H, self.W)
        out = self.pp.flatten_history(x)
        self.assertEqual(tuple(out.shape), (self.B, T * self.C, self.H, self.W))

    def test_expand_history_5d_noop(self):
        """expand_history on a 5-D tensor returns it unchanged."""
        T = 2
        x = self._rand(self.B, T, self.C, self.H, self.W)
        out = self.pp.expand_history(x, nhist=T)
        self.assertEqual(out.shape, x.shape)

    def test_expand_history_4d_adds_time(self):
        """expand_history on (B, T*C, H, W) gives (B, T, C, H, W)."""
        T = 3
        x = self._rand(self.B, T * self.C, self.H, self.W)
        out = self.pp.expand_history(x, nhist=T)
        self.assertEqual(tuple(out.shape), (self.B, T, self.C, self.H, self.W))

    def test_flatten_expand_roundtrip(self, verbose=False):
        """flatten_history(expand_history(x, T)) is bit-identical to x."""
        T = 2
        x = self._rand(self.B, T * self.C, self.H, self.W)
        out = self.pp.flatten_history(self.pp.expand_history(x, nhist=T))
        self.assertTrue(compare_tensors("flatten/expand roundtrip", out, x, verbose=verbose))

    # -----------------------------------------------------------------------
    # Static features
    # -----------------------------------------------------------------------

    def test_add_static_features_noop_when_none(self, verbose=False):
        """Without static features, add_static_features is identity."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.add_static_features(x)
        self.assertTrue(compare_tensors("add_static noop", out, x, verbose=verbose))

    def test_remove_static_features_noop_when_none(self, verbose=False):
        """Without static features, remove_static_features is identity."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.remove_static_features(x)
        self.assertTrue(compare_tensors("remove_static noop", out, x, verbose=verbose))

    def test_add_static_features_increases_channels(self):
        """Raw 2-channel grid appends 2 channels to the input."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.add_static_features(x)
        self.assertEqual(out.shape[1], self.C + 2)

    def test_add_remove_static_features_roundtrip_shape(self):
        """remove_static_features restores original channel count."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.remove_static_features(pp.add_static_features(x))
        self.assertEqual(out.shape, x.shape)

    def test_add_remove_static_features_roundtrip_values(self, verbose=False):
        """remove_static_features leaves the original channels bit-identical."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.remove_static_features(pp.add_static_features(x))
        self.assertTrue(compare_tensors("add/remove static roundtrip", out, x, verbose=verbose))

    def test_get_static_features_returns_tensor_when_enabled(self):
        """get_static_features returns a non-None tensor when grid is enabled."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        sf = pp.get_static_features()
        self.assertIsNotNone(sf)
        self.assertIsInstance(sf, torch.Tensor)

    def test_get_static_features_returns_none_when_disabled(self):
        """get_static_features returns None when no features are configured."""
        self.assertIsNone(self.pp.get_static_features())

    def test_static_feature_spatial_shape_matches_grid(self):
        """The (H, W) of the stored static feature matches the image shape."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        sf = pp.get_static_features()
        self.assertEqual(sf.shape[-2], IMG_SIZE_H)
        self.assertEqual(sf.shape[-1], IMG_SIZE_W)

    # -----------------------------------------------------------------------
    # append_history
    # -----------------------------------------------------------------------

    def test_append_history_n_history_0_returns_x2(self, verbose=False):
        """When n_history=0, append_history returns x2 unchanged."""
        x1 = self._rand(self.B, self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.append_history(x1, x2, step=0)
        self.assertTrue(compare_tensors("append_history n_history=0", out, x2, verbose=verbose))

    def test_append_history_n_history_1_shape(self):
        """When n_history=1, append_history returns (B, 2*C, H, W)."""
        params = get_default_parameters()
        params.n_history = 1
        pp = Preprocessor2D(params)
        T = 2
        x1 = self._rand(self.B, T * self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        out = pp.append_history(x1, x2, step=0)
        self.assertEqual(tuple(out.shape), (self.B, T * self.C, self.H, self.W))

    def test_append_history_n_history_1_rolls_content(self, verbose=False):
        """append_history drops the oldest time-step and appends x2 as newest."""
        params = get_default_parameters()
        params.n_history = 1
        pp = Preprocessor2D(params)
        T = 2
        x1 = self._rand(self.B, T * self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)

        out = pp.append_history(x1, x2, step=0)
        out_5d = pp.expand_history(out, nhist=T)
        x1_5d = pp.expand_history(x1, nhist=T)

        self.assertTrue(compare_tensors("append_history newest slot", out_5d[:, -1], x2, verbose=verbose))
        self.assertTrue(
            compare_tensors("append_history oldest slot rolled", out_5d[:, 0], x1_5d[:, 1], verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # cache_unpredicted_features / get_unpredicted_features
    # -----------------------------------------------------------------------

    def test_get_unpredicted_features_none_without_cache(self):
        """Before caching, get_unpredicted_features returns (None, None)."""
        self.pp.eval()
        inp, tar = self.pp.get_unpredicted_features()
        self.assertIsNone(inp)
        self.assertIsNone(tar)

    def test_cache_unpredicted_features_stores_and_retrieves_train(self, verbose=False):
        """Cached xz/yz are retrievable from get_unpredicted_features (train mode)."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        y = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, y, xz=xz, yz=yz)
        inp, tar = self.pp.get_unpredicted_features()
        self.assertIsNotNone(inp)
        self.assertIsNotNone(tar)
        self.assertTrue(compare_tensors("cache unpredicted inp (train)", inp, xz, verbose=verbose))
        self.assertTrue(compare_tensors("cache unpredicted tar (train)", tar, yz, verbose=verbose))

    def test_cache_unpredicted_features_stores_and_retrieves_eval(self, verbose=False):
        """Cached xz/yz are retrievable in eval mode."""
        self.pp.eval()
        x = self._rand(self.B, self.C, self.H, self.W)
        y = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, y, xz=xz, yz=yz)
        inp, tar = self.pp.get_unpredicted_features()
        self.assertIsNotNone(inp)
        self.assertIsNotNone(tar)
        self.assertTrue(compare_tensors("cache unpredicted inp (eval)", inp, xz, verbose=verbose))
        self.assertTrue(compare_tensors("cache unpredicted tar (eval)", tar, yz, verbose=verbose))

    def test_cache_unpredicted_features_returns_x_y_unchanged(self, verbose=False):
        """cache_unpredicted_features returns x and y bit-identically."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        y = self._rand(self.B, self.C, self.H, self.W)
        xr, yr = self.pp.cache_unpredicted_features(x, y)
        self.assertTrue(compare_tensors("cache returns x unchanged", xr, x, verbose=verbose))
        self.assertTrue(compare_tensors("cache returns y unchanged", yr, y, verbose=verbose))

    # -----------------------------------------------------------------------
    # correct_bias
    # -----------------------------------------------------------------------

    def test_correct_bias_noop_without_bias(self, verbose=False):
        """correct_bias returns input unchanged when no bias correction is registered."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.correct_bias(x)
        self.assertTrue(compare_tensors("correct_bias noop", out, x, verbose=verbose))

    def test_correct_bias_subtracts_registered_bias(self, verbose=False):
        """correct_bias subtracts the bias_correction buffer from the input."""
        pp = Preprocessor2D(get_default_parameters())
        bias = torch.ones(1, self.C, 1, 1)
        pp.register_buffer("bias_correction", bias, persistent=False)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.correct_bias(x)
        self.assertTrue(compare_tensors("correct_bias subtract", out, x - bias, verbose=verbose))

    def test_correct_bias_broadcasts_over_batch_and_spatial(self, verbose=False):
        """A (1, C, 1, 1) bias broadcasts correctly over (B, C, H, W)."""
        pp = Preprocessor2D(get_default_parameters())
        bias = torch.arange(self.C, dtype=torch.float32).reshape(1, self.C, 1, 1)
        pp.register_buffer("bias_correction", bias, persistent=False)
        x = torch.zeros(self.B, self.C, self.H, self.W)
        out = pp.correct_bias(x)
        expected = -bias.expand(self.B, self.C, self.H, self.W)
        self.assertTrue(compare_tensors("correct_bias broadcast", out, expected, verbose=verbose))

    # -----------------------------------------------------------------------
    # Internal state accessors — no noise
    # -----------------------------------------------------------------------

    def test_get_base_seed_default_no_noise(self):
        """Without noise, get_base_seed returns the supplied default value."""
        self.assertEqual(self.pp.get_base_seed(default=42), 42)

    def test_get_internal_rng_no_noise_returns_none(self):
        """Without noise, both get_internal_rng(gpu=True/False) return None."""
        self.assertIsNone(self.pp.get_internal_rng(gpu=True))
        self.assertIsNone(self.pp.get_internal_rng(gpu=False))

    def test_get_internal_state_no_noise_rng(self):
        """Without noise, get_internal_state(tensor=False) returns (None, None)."""
        state = self.pp.get_internal_state(tensor=False)
        self.assertEqual(state, (None, None))

    def test_get_internal_state_no_noise_tensor(self):
        """Without noise, get_internal_state(tensor=True) returns None."""
        self.assertIsNone(self.pp.get_internal_state(tensor=True))

    def test_set_internal_state_no_noise_noop(self):
        """Without noise, set_internal_state(None) does not raise."""
        self.pp.set_internal_state(None)

    def test_update_internal_state_no_noise_noop(self):
        """Without noise, update_internal_state() does not raise."""
        self.pp.update_internal_state()

    # -----------------------------------------------------------------------
    # Internal state accessors — with DummyNoiseS2
    # -----------------------------------------------------------------------

    def _make_pp_with_noise(self, input_noise_mode="concatenate", n_channels=1):
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": input_noise_mode, "n_channels": n_channels}
        return Preprocessor2D(params)

    def test_construction_with_dummy_noise(self):
        """Preprocessor2D with dummy noise constructs without error."""
        pp = self._make_pp_with_noise()
        self.assertTrue(hasattr(pp, "input_noise"))

    def test_get_base_seed_with_noise_is_int(self):
        """With noise, get_base_seed returns an int (the noise_base_seed)."""
        pp = self._make_pp_with_noise()
        self.assertIsInstance(pp.get_base_seed(), int)

    def test_get_internal_rng_with_noise_cpu(self):
        """With noise, get_internal_rng(gpu=False) returns a CPU Generator."""
        pp = self._make_pp_with_noise()
        rng = pp.get_internal_rng(gpu=False)
        self.assertIsNotNone(rng)
        self.assertIsInstance(rng, torch.Generator)

    def test_get_internal_state_with_noise_rng_is_tuple(self):
        """With noise, get_internal_state(tensor=False) returns a 2-tuple."""
        pp = self._make_pp_with_noise()
        state = pp.get_internal_state(tensor=False)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)

    def test_get_internal_state_with_noise_tensor_is_tensor(self):
        """With noise, get_internal_state(tensor=True) returns a Tensor."""
        pp = self._make_pp_with_noise()
        state = pp.get_internal_state(tensor=True)
        self.assertIsInstance(state, torch.Tensor)

    def test_set_rng_with_noise_does_not_raise(self):
        """set_rng resets the noise RNG without raising."""
        pp = self._make_pp_with_noise()
        pp.set_rng(reset=True, seed=42)

    def test_set_internal_state_tensor_roundtrip(self, verbose=False):
        """Saving and restoring tensor state restores the stored noise field."""
        pp = self._make_pp_with_noise()
        pp.update_internal_state()
        tensor_state = pp.get_internal_state(tensor=True)
        # overwrite with zeros, then restore
        pp.input_noise.state.zero_()
        pp.set_internal_state(tensor_state)
        restored = pp.get_internal_state(tensor=True)
        self.assertTrue(compare_tensors("tensor state roundtrip", restored, tensor_state, verbose=verbose))

    def test_update_internal_state_with_noise_does_not_raise(self):
        """update_internal_state() with noise does not raise."""
        pp = self._make_pp_with_noise()
        pp.update_internal_state()

    # -----------------------------------------------------------------------
    # cache_unpredicted_features — copy_() vs rebind
    # -----------------------------------------------------------------------

    def test_cache_unpredicted_features_same_shape_reuses_buffer_train(self):
        """Second cache call with matching shape uses copy_(), preserving data_ptr (train)."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz1 = self._rand(self.B, 1, 1, self.H, self.W)
        xz2 = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz1, yz=yz)
        ptr1 = self.pp.unpredicted_inp_train.data_ptr()
        self.pp.cache_unpredicted_features(x, x, xz=xz2, yz=yz)
        ptr2 = self.pp.unpredicted_inp_train.data_ptr()
        self.assertEqual(ptr1, ptr2)

    def test_cache_unpredicted_features_same_shape_updates_values_train(self, verbose=False):
        """After the copy_() path the stored values reflect the new tensor (train)."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz1 = self._rand(self.B, 1, 1, self.H, self.W)
        xz2 = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz1, yz=yz)
        self.pp.cache_unpredicted_features(x, x, xz=xz2, yz=yz)
        inp, _ = self.pp.get_unpredicted_features()
        self.assertTrue(compare_tensors("copy path updated value", inp, xz2, verbose=verbose))

    def test_cache_unpredicted_features_same_shape_reuses_buffer_eval(self):
        """Second cache call with matching shape uses copy_(), preserving data_ptr (eval)."""
        self.pp.eval()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz1 = self._rand(self.B, 1, 1, self.H, self.W)
        xz2 = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz1, yz=yz)
        ptr1 = self.pp.unpredicted_inp_eval.data_ptr()
        self.pp.cache_unpredicted_features(x, x, xz=xz2, yz=yz)
        ptr2 = self.pp.unpredicted_inp_eval.data_ptr()
        self.assertEqual(ptr1, ptr2)

    def test_cache_unpredicted_features_shape_change_rebinds_train(self):
        """When the shape changes, cache rebinds rather than copy_()."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz1 = self._rand(self.B, 1, 1, self.H, self.W)
        xz2 = self._rand(self.B, 1, 2, self.H, self.W)  # different C_z
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz1, yz=yz)
        ptr1 = self.pp.unpredicted_inp_train.data_ptr()
        self.pp.cache_unpredicted_features(x, x, xz=xz2, yz=yz)
        ptr2 = self.pp.unpredicted_inp_train.data_ptr()
        self.assertNotEqual(ptr1, ptr2)
        self.assertEqual(tuple(self.pp.unpredicted_inp_train.shape), tuple(xz2.shape))

    def test_cache_unpredicted_features_none_xz_clears_cache_train(self):
        """Passing xz=None resets unpredicted_inp_train to None."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        self.assertIsNotNone(self.pp.unpredicted_inp_train)
        self.pp.cache_unpredicted_features(x, x, xz=None, yz=yz)
        self.assertIsNone(self.pp.unpredicted_inp_train)

    # -----------------------------------------------------------------------
    # append_unpredicted_features
    # -----------------------------------------------------------------------

    def test_append_unpredicted_features_noop_no_cache_train(self, verbose=False):
        """Without cached features, append_unpredicted_features is identity (train)."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.append_unpredicted_features(x, target=False)
        self.assertTrue(compare_tensors("noop train", out, x, verbose=verbose))

    def test_append_unpredicted_features_noop_no_cache_eval(self, verbose=False):
        """Without cached features, append_unpredicted_features is identity (eval)."""
        self.pp.eval()
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.append_unpredicted_features(x, target=False)
        self.assertTrue(compare_tensors("noop eval", out, x, verbose=verbose))

    def test_append_unpredicted_features_appends_inp_train(self):
        """With cached inp features, output has C + C_xz channels (train mode)."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = self.pp.append_unpredicted_features(x, target=False)
        self.assertEqual(out.shape[1], self.C + 1)

    def test_append_unpredicted_features_appends_inp_eval(self):
        """With cached inp features, output has C + C_xz channels (eval mode)."""
        self.pp.eval()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = self.pp.append_unpredicted_features(x, target=False)
        self.assertEqual(out.shape[1], self.C + 1)

    def test_append_unpredicted_features_target_uses_yz(self):
        """target=True appends yz features; inp and tar channel counts differ."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 2, self.H, self.W)  # 2 inp channels
        yz = self._rand(self.B, 1, 3, self.H, self.W)  # 3 tar channels
        self.pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out_inp = self.pp.append_unpredicted_features(x, target=False)
        out_tar = self.pp.append_unpredicted_features(x, target=True)
        self.assertEqual(out_inp.shape[1], self.C + 2)
        self.assertEqual(out_tar.shape[1], self.C + 3)

    def test_append_unpredicted_features_content_x_channels_preserved(self, verbose=False):
        """The original C channels of x appear unchanged in the first C output channels."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = self.pp.append_unpredicted_features(x, target=False)
        self.assertTrue(compare_tensors("original channels preserved", out[:, : self.C], x, verbose=verbose))

    def test_append_unpredicted_features_content_xz_channels_appended(self, verbose=False):
        """The appended channels equal the cached xz squeezed on the time dim."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = self.pp.append_unpredicted_features(x, target=False)
        # xz has shape (B, T=1, C_z=1, H, W); xz[:, 0] is (B, C_z=1, H, W)
        self.assertTrue(compare_tensors("xz appended", out[:, self.C :], xz[:, 0], verbose=verbose))

    # -----------------------------------------------------------------------
    # _append_channels with noise
    # -----------------------------------------------------------------------

    def test_append_channels_concatenate_noise_channel_count(self):
        """concatenate noise adds noise_channels beyond x + xz in the output."""
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 2}
        pp = Preprocessor2D(params)
        pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = pp.append_unpredicted_features(x, target=False)
        # C original + 1 xz + 2 noise channels
        self.assertEqual(out.shape[1], self.C + 1 + 2)

    def test_append_channels_concatenate_noise_original_channels_unchanged(self, verbose=False):
        """With concatenate noise the first C channels are bit-identical to x."""
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 1}
        pp = Preprocessor2D(params)
        pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = pp.append_unpredicted_features(x, target=False)
        self.assertTrue(compare_tensors("x channels unchanged with concat noise", out[:, : self.C], x, verbose=verbose))

    def test_append_channels_perturb_noise_channel_count_unchanged(self):
        """perturb noise does not add extra channels to the output."""
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "perturb", "perturb_channels": ["u10m"]}
        pp = Preprocessor2D(params)
        pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = pp.append_unpredicted_features(x, target=False)
        # C original + 1 xz, no extra noise channels
        self.assertEqual(out.shape[1], self.C + 1)

    def test_append_channels_perturb_noise_zero_leaves_x_unchanged(self, verbose=False):
        """With zero DummyNoiseS2 in perturb mode, the x channels are bit-identical."""
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "perturb", "perturb_channels": ["u10m"]}
        pp = Preprocessor2D(params)
        pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        out = pp.append_unpredicted_features(x, target=False)
        self.assertTrue(compare_tensors("x unchanged with zero perturb noise", out[:, : self.C], x, verbose=verbose))

    def test_concatenate_noise_without_cached_features(self, verbose=False):
        """Without zenith the noise is still appended, as the channel count expects."""
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 2}
        pp = Preprocessor2D(params)
        for mode in ["train", "eval"]:
            with self.subTest(mode=mode):
                getattr(pp, mode)()
                x = self._rand(self.B, self.C, self.H, self.W)
                pp.cache_unpredicted_features(x, x, xz=None, yz=None)
                out = pp.append_unpredicted_features(x, target=False)
                self.assertEqual(out.shape[1], self.C + 2)
                self.assertTrue(compare_tensors("x channels unchanged", out[:, : self.C], x, verbose=verbose))

    def test_concatenate_noise_without_cached_features_with_history(self):
        """With history, the noise is appended to every step."""
        params = get_default_parameters()
        params.n_history = 1
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 1}
        pp = Preprocessor2D(params)
        pp.train()
        x = self._rand(self.B, 2 * self.C, self.H, self.W)
        out = pp.append_unpredicted_features(x, target=False)
        self.assertEqual(out.shape[1], 2 * (self.C + 1))
        # per step: the step's channels first, then its noise channel
        self.assertTrue(torch.equal(out[:, : self.C], x[:, : self.C]))
        self.assertTrue(torch.equal(out[:, self.C + 1 : 2 * self.C + 1], x[:, self.C :]))

    # -----------------------------------------------------------------------
    # append_history with update_state
    # -----------------------------------------------------------------------

    def test_append_history_update_state_n_history_0_overwrites_inp(self, verbose=False):
        """With n_history=0, update_state copies yz[:, 0:1] into unpredicted_inp."""
        pp = Preprocessor2D(get_default_parameters())
        pp.train()
        x1 = self._rand(self.B, self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x1, x2, xz=xz, yz=yz)

        pp.append_history(x1, x2, step=0, update_state=True)

        inp, _ = pp.get_unpredicted_features()
        self.assertTrue(compare_tensors("n_history=0 overwrites inp", inp, yz[:, 0:1], verbose=verbose))

    def test_append_history_update_state_n_history_1_rolls_newest(self, verbose=False):
        """With n_history=1, after update the newest slot of inp equals yz[:, 0]."""
        params = get_default_parameters()
        params.n_history = 1
        pp = Preprocessor2D(params)
        pp.train()
        T = 2
        x1 = self._rand(self.B, T * self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, T, 1, self.H, self.W)
        yz = self._rand(self.B, T, 1, self.H, self.W)
        pp.cache_unpredicted_features(x1, x2, xz=xz, yz=yz)

        pp.append_history(x1, x2, step=0, update_state=True)

        inp, _ = pp.get_unpredicted_features()
        self.assertTrue(compare_tensors("newest slot = yz step 0", inp[:, -1], yz[:, 0], verbose=verbose))

    def test_append_history_update_state_n_history_1_rolls_oldest(self, verbose=False):
        """With n_history=1, after update the oldest slot of inp equals the old xz[:, 1]."""
        params = get_default_parameters()
        params.n_history = 1
        pp = Preprocessor2D(params)
        pp.train()
        T = 2
        x1 = self._rand(self.B, T * self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, T, 1, self.H, self.W)
        yz = self._rand(self.B, T, 1, self.H, self.W)
        pp.cache_unpredicted_features(x1, x2, xz=xz, yz=yz)

        pp.append_history(x1, x2, step=0, update_state=True)

        inp, _ = pp.get_unpredicted_features()
        self.assertTrue(compare_tensors("oldest slot = old xz[1]", inp[:, 0], xz[:, 1], verbose=verbose))

    def test_append_history_update_state_false_preserves_inp(self, verbose=False):
        """With update_state=False, cached unpredicted_inp is not modified."""
        pp = Preprocessor2D(get_default_parameters())
        pp.train()
        x1 = self._rand(self.B, self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x1, x2, xz=xz, yz=yz)

        pp.append_history(x1, x2, step=0, update_state=False)

        inp, _ = pp.get_unpredicted_features()
        self.assertTrue(compare_tensors("inp unchanged with update_state=False", inp, xz, verbose=verbose))

    # -----------------------------------------------------------------------
    # history_compute_stats / history_normalize / history_denormalize
    # -----------------------------------------------------------------------

    def test_history_normalize_none_mode_returns_x_unchanged(self, verbose=False):
        """history_normalization_mode='none': normalize is a no-op."""
        # default params have mode="none"
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.history_normalize(x)
        self.assertTrue(compare_tensors("normalize noop", out, x, verbose=verbose))

    def test_history_denormalize_none_mode_returns_x_unchanged(self, verbose=False):
        """history_normalization_mode='none': denormalize is a no-op."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.history_denormalize(x)
        self.assertTrue(compare_tensors("denormalize noop", out, x, verbose=verbose))

    def _make_pp_mean(self, n_history=1):
        """Build a Preprocessor2D in 'mean' normalization mode."""
        params = get_default_parameters()
        params.n_history = n_history
        params.history_normalization_mode = "mean"
        return Preprocessor2D(params)

    def test_history_normalize_denormalize_roundtrip_mean_mode(self, verbose=False):
        """normalize then denormalize recovers original tensor (mean mode, n_history=1)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(x)
        xn = pp.history_normalize(x)
        xr = pp.history_denormalize(xn)
        self.assertTrue(compare_tensors("roundtrip non-target", xr, x, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_normalize_denormalize_roundtrip_target_mean_mode(self, verbose=False):
        """normalize(target=True) then denormalize(target=True) recovers original (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        inp = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(inp)
        # target has only C channels (one time step)
        y = self._rand(self.B, self.C, self.H, self.W)
        yn = pp.history_normalize(y, target=True)
        yr = pp.history_denormalize(yn, target=True)
        self.assertTrue(compare_tensors("roundtrip target", yr, y, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_normalize_denormalize_roundtrip_5d_mean_mode(self, verbose=False):
        """Roundtrip works on 5-D (B, T, C, H, W) input as well (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        x5d = self._rand(self.B, T, self.C, self.H, self.W)
        # history_compute_stats expects 4-D or 5-D; use flattened form for stats
        pp.history_compute_stats(x5d)
        xn = pp.history_normalize(x5d)
        xr = pp.history_denormalize(xn)
        self.assertTrue(compare_tensors("roundtrip 5D", xr, x5d, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_normalize_denormalize_roundtrip_exponential_mode(self, verbose=False):
        """Roundtrip holds for exponential normalization mode."""
        params = get_default_parameters()
        params.n_history = 2
        params.history_normalization_mode = "exponential"
        params.history_normalization_decay = 0.5
        pp = Preprocessor2D(params)
        T = pp.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(x)
        xn = pp.history_normalize(x)
        xr = pp.history_denormalize(xn)
        self.assertTrue(compare_tensors("roundtrip exponential", xr, x, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_compute_stats_mean_is_finite(self):
        """After history_compute_stats, history_mean contains no NaN or Inf."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(x)
        self.assertIsNotNone(pp.history_mean)
        self.assertFalse(torch.any(torch.isnan(pp.history_mean)).item())
        self.assertFalse(torch.any(torch.isinf(pp.history_mean)).item())

    def test_history_compute_stats_std_is_finite_and_nonneg(self):
        """After history_compute_stats, history_std is finite and non-negative."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(x)
        self.assertIsNotNone(pp.history_std)
        self.assertFalse(torch.any(torch.isnan(pp.history_std)).item())
        self.assertFalse(torch.any(torch.isinf(pp.history_std)).item())
        self.assertTrue((pp.history_std >= 0).all().item())

    def test_history_compute_stats_shapes_mean_mode(self):
        """After history_compute_stats (mean mode), mean/std have shape (B, C, 1, 1)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(x)
        expected = (self.B, self.C, 1, 1)
        self.assertEqual(tuple(pp.history_mean.shape), expected)
        self.assertEqual(tuple(pp.history_std.shape), expected)

    # -----------------------------------------------------------------------
    # history_compute_stats — known-answer tests
    # -----------------------------------------------------------------------

    def test_history_compute_stats_constant_field_mean_mode(self, verbose=False):
        """Constant field value c → history_mean == c, history_std == 0 (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        value = 3.7
        x = torch.full((self.B, T * self.C, self.H, self.W), value, dtype=torch.float32)
        pp.history_compute_stats(x)
        self.assertTrue(
            compare_tensors(
                "constant-field mean",
                pp.history_mean,
                torch.full_like(pp.history_mean, value),
                atol=1e-5,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "constant-field std", pp.history_std, torch.zeros_like(pp.history_std), atol=1e-5, verbose=verbose
            )
        )

    def test_history_compute_stats_constant_field_exponential_mode(self, verbose=False):
        """Constant field → mean == value, std == 0 (exponential mode)."""
        params = get_default_parameters()
        params.n_history = 2
        params.history_normalization_mode = "exponential"
        params.history_normalization_decay = 0.5
        pp = Preprocessor2D(params)
        T = pp.n_history + 1
        value = -1.25
        x = torch.full((self.B, T * self.C, self.H, self.W), value, dtype=torch.float32)
        pp.history_compute_stats(x)
        self.assertTrue(
            compare_tensors(
                "constant-field mean (exponential)",
                pp.history_mean,
                torch.full_like(pp.history_mean, value),
                atol=1e-5,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "constant-field std (exponential)",
                pp.history_std,
                torch.zeros_like(pp.history_std),
                atol=1e-5,
                verbose=verbose,
            )
        )

    def test_history_compute_stats_per_channel_constants_mean_mode(self, verbose=False):
        """Per-channel constants: history_mean[:, c] equals channel c's constant value."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        # every spatial/time value of channel c is (c + 1.0)
        x5 = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for c in range(self.C):
            x5[:, :, c, :, :] = float(c + 1)
        x = pp.flatten_history(x5)
        pp.history_compute_stats(x)
        expected_mean = torch.arange(1, self.C + 1, dtype=torch.float32).reshape(1, self.C, 1, 1)
        self.assertTrue(compare_tensors("per-channel mean", pp.history_mean, expected_mean, atol=1e-5, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "per-channel std", pp.history_std, torch.zeros_like(pp.history_std), atol=1e-5, verbose=verbose
            )
        )

    def test_history_compute_stats_matches_analytical_weighted_mean_exponential(self, verbose=False):
        """Spatially-constant-per-step fields: history_mean recovers the analytical weighted temporal average."""
        params = get_default_parameters()
        params.n_history = 2
        params.history_normalization_mode = "exponential"
        params.history_normalization_decay = 0.5
        pp = Preprocessor2D(params)
        T = pp.n_history + 1
        # spatially constant per (time, channel) but distinct across time & channel
        values = (torch.arange(T * self.C, dtype=torch.float32) + 1.0).reshape(T, self.C)
        x5 = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for t in range(T):
            for c in range(self.C):
                x5[:, t, c, :, :] = values[t, c]
        x = pp.flatten_history(x5)
        pp.history_compute_stats(x)
        # reconstruct weights exactly as the preprocessor does: first element is oldest
        w = torch.exp(-0.5 * torch.arange(start=T - 1, end=-1, step=-1, dtype=torch.float32))
        w = w / w.sum()
        expected_mean = (w.unsqueeze(-1) * values).sum(dim=0).reshape(1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors(
                "analytical weighted mean", pp.history_mean, expected_mean, atol=1e-5, rtol=1e-5, verbose=verbose
            )
        )

    def test_history_compute_stats_linear_ramp_mean_mode_mean(self, verbose=False):
        """Linear ramp v_t = a_c + b_c * t (spatially constant): history_mean matches a_c + b_c * t_bar."""
        pp = self._make_pp_mean(n_history=2)
        T = pp.n_history + 1
        a = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float32)[: self.C]
        b = torch.tensor([1.0, -0.5, 2.0, 0.25, -1.0], dtype=torch.float32)[: self.C]
        x5 = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for t in range(T):
            x5[:, t, :, :, :] = (a + b * t).reshape(1, self.C, 1, 1)
        x = pp.flatten_history(x5)
        pp.history_compute_stats(x)
        t_vec = torch.arange(T, dtype=torch.float32)
        t_bar = t_vec.mean()
        expected_mean = (a + b * t_bar).reshape(1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors("linear ramp mean", pp.history_mean, expected_mean, atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    def test_history_compute_stats_linear_ramp_mean_mode_std(self, verbose=False):
        """Linear ramp: history_std = |b_c| * sqrt(Var_t) (uniform-weight variance of 0..T-1)."""
        pp = self._make_pp_mean(n_history=2)
        T = pp.n_history + 1
        a = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float32)[: self.C]
        b = torch.tensor([1.0, -0.5, 2.0, 0.25, -1.0], dtype=torch.float32)[: self.C]
        x5 = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for t in range(T):
            x5[:, t, :, :, :] = (a + b * t).reshape(1, self.C, 1, 1)
        x = pp.flatten_history(x5)
        pp.history_compute_stats(x)
        t_vec = torch.arange(T, dtype=torch.float32)
        t_bar = t_vec.mean()
        var_t = ((t_vec - t_bar) ** 2).mean()
        expected_std = (b.abs() * torch.sqrt(var_t)).reshape(1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors("linear ramp std", pp.history_std, expected_std, atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # history_normalize post-condition: weighted temporal mean ≈ 0, weighted L2 ≈ 1
    # -----------------------------------------------------------------------

    def test_history_normalize_post_condition_mean_mode(self, verbose=False):
        """After normalize (mean mode): weighted temporal mean ≈ 0, weighted L2 ≈ 1, per channel."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(x)
        xn = pp.history_normalize(x)
        xn5 = pp.expand_history(xn, nhist=T)
        # mean-mode weights are uniform 1/T: use plain temporal average of per-time area means
        quad_xn = pp.quadrature(xn5)  # (B, T, C)
        quad_xn2 = pp.quadrature(xn5**2)  # (B, T, C)
        weighted_mean = quad_xn.mean(dim=1)  # (B, C)
        weighted_l2 = quad_xn2.mean(dim=1)  # (B, C)
        self.assertTrue(
            compare_tensors(
                "normalize weighted mean ≈ 0 (mean mode)",
                weighted_mean,
                torch.zeros_like(weighted_mean),
                atol=1e-5,
                rtol=1e-4,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "normalize weighted L2 ≈ 1 (mean mode)",
                weighted_l2,
                torch.ones_like(weighted_l2),
                atol=1e-4,
                rtol=1e-4,
                verbose=verbose,
            )
        )

    def test_history_normalize_post_condition_exponential_mode(self, verbose=False):
        """After normalize (exponential mode): weighted temporal mean ≈ 0, weighted L2 ≈ 1, per channel."""
        params = get_default_parameters()
        params.n_history = 2
        params.history_normalization_mode = "exponential"
        params.history_normalization_decay = 0.5
        pp = Preprocessor2D(params)
        T = pp.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(x)
        xn = pp.history_normalize(x)
        xn5 = pp.expand_history(xn, nhist=T)
        # exponential weights: buffer shape is (1, T, 1, 1, 1)
        w = pp.history_normalization_weights.reshape(1, T, 1)  # broadcasts with (B, T, C)
        quad_xn = pp.quadrature(xn5)
        quad_xn2 = pp.quadrature(xn5**2)
        weighted_mean = (w * quad_xn).sum(dim=1)
        weighted_l2 = (w * quad_xn2).sum(dim=1)
        self.assertTrue(
            compare_tensors(
                "normalize weighted mean ≈ 0 (exponential)",
                weighted_mean,
                torch.zeros_like(weighted_mean),
                atol=1e-5,
                rtol=1e-4,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "normalize weighted L2 ≈ 1 (exponential)",
                weighted_l2,
                torch.ones_like(weighted_l2),
                atol=1e-4,
                rtol=1e-4,
                verbose=verbose,
            )
        )

    # -----------------------------------------------------------------------
    # Noise determinism end-to-end
    # -----------------------------------------------------------------------

    def _make_pp_with_constant_random_noise(self):
        """Build a preprocessor whose input_noise is a seeded DummyNoiseS2 in constant_random mode."""
        from makani.models.noise import DummyNoiseS2

        params = get_default_parameters()
        # construct with dummy-concatenate so the preprocessor is wired for noise
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 1}
        pp = Preprocessor2D(params)
        pp.train()
        # replace with a constant_random DummyNoise so set_rng actually matters
        pp.input_noise = DummyNoiseS2(
            img_shape=[self.H, self.W],
            batch_size=self.B,
            num_channels=1,
            num_time_steps=pp.n_history + 1,
            mode="constant_random",
        )
        return pp

    def test_noise_determinism_same_seed_produces_equal_outputs(self, verbose=False):
        """set_rng(seed=42) twice, each followed by a forward pass, yields identical outputs."""
        pp = self._make_pp_with_constant_random_noise()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)

        pp.set_rng(reset=True, seed=42)
        pp.update_internal_state()
        out1 = pp.append_unpredicted_features(x, target=False)

        pp.set_rng(reset=True, seed=42)
        pp.update_internal_state()
        out2 = pp.append_unpredicted_features(x, target=False)

        self.assertTrue(compare_tensors("noise same seed", out1, out2, verbose=verbose))

    def test_noise_determinism_different_seed_produces_different_outputs(self, verbose=False):
        """Two different seeds give different outputs (the noise channels differ)."""
        pp = self._make_pp_with_constant_random_noise()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)

        pp.set_rng(reset=True, seed=42)
        pp.update_internal_state()
        out1 = pp.append_unpredicted_features(x, target=False)

        pp.set_rng(reset=True, seed=999)
        pp.update_internal_state()
        out2 = pp.append_unpredicted_features(x, target=False)

        self.assertFalse(compare_tensors("noise different seeds", out1, out2, verbose=verbose))

    # -----------------------------------------------------------------------
    # append_history with out-of-range step
    # -----------------------------------------------------------------------

    def test_append_history_update_state_out_of_range_step_is_noop(self, verbose=False):
        """With step >= unpredicted_tar_train.shape[1], the cached inp is not modified (silent no-op)."""
        pp = Preprocessor2D(get_default_parameters())
        pp.train()
        x1 = self._rand(self.B, self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)  # shape[1] == 1
        pp.cache_unpredicted_features(x1, x2, xz=xz, yz=yz)

        inp_before, _ = pp.get_unpredicted_features()
        # step=5 is far beyond yz.shape[1] == 1 — current behavior is a silent no-op
        pp.append_history(x1, x2, step=5, update_state=True)
        inp_after, _ = pp.get_unpredicted_features()

        self.assertTrue(
            compare_tensors("inp unchanged after out-of-range step", inp_after, inp_before, verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # Gradient flow through train-path ops
    # -----------------------------------------------------------------------

    def test_append_channels_grad_flows_concatenate_noise(self):
        """_append_channels with concatenate noise preserves autograd w.r.t. x."""
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 1}
        pp = Preprocessor2D(params)
        pp.train()
        x_base = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x_base, x_base, xz=xz, yz=yz)
        x = x_base.clone().requires_grad_(True)
        out = pp.append_unpredicted_features(x, target=False)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        # Each element of x appears exactly once in the output → gradient is 1
        self.assertTrue(torch.all(x.grad == 1.0).item())

    def test_append_channels_grad_flows_perturb_noise(self):
        """_append_channels with perturb noise still preserves autograd w.r.t. x."""
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "perturb", "perturb_channels": ["u10m"]}
        pp = Preprocessor2D(params)
        pp.train()
        x_base = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x_base, x_base, xz=xz, yz=yz)
        x = x_base.clone().requires_grad_(True)
        out = pp.append_unpredicted_features(x, target=False)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.all(x.grad == 1.0).item())

    def test_history_normalize_grad_flows_mean_mode(self):
        """history_normalize is differentiable w.r.t. its input (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        stats_x = self._rand(self.B, T * self.C, self.H, self.W)
        pp.history_compute_stats(stats_x)
        x = self._rand(self.B, T * self.C, self.H, self.W).requires_grad_(True)
        out = pp.history_normalize(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        # d/dx of (x - m)/s summed = 1/s broadcast; grad must be non-zero everywhere
        self.assertTrue((x.grad.abs() > 0).all().item())

    def test_history_normalize_grad_flows_5d_mean_mode(self):
        """history_normalize is differentiable on 5-D input too."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        stats_x = self._rand(self.B, T, self.C, self.H, self.W)
        pp.history_compute_stats(stats_x)
        x = self._rand(self.B, T, self.C, self.H, self.W).requires_grad_(True)
        out = pp.history_normalize(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue((x.grad.abs() > 0).all().item())

    # -----------------------------------------------------------------------
    # CPU ↔ GPU numerical consistency
    # -----------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_append_unpredicted_features_cpu_gpu_match(self, verbose=False):
        """append_unpredicted_features matches on CPU and GPU (no noise)."""
        pp_cpu = Preprocessor2D(get_default_parameters())
        pp_gpu = Preprocessor2D(get_default_parameters()).to("cuda")
        pp_cpu.train()
        pp_gpu.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp_cpu.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        pp_gpu.cache_unpredicted_features(x.to("cuda"), x.to("cuda"), xz=xz.to("cuda"), yz=yz.to("cuda"))
        out_cpu = pp_cpu.append_unpredicted_features(x, target=False)
        out_gpu = pp_gpu.append_unpredicted_features(x.to("cuda"), target=False).cpu()
        self.assertTrue(
            compare_tensors("append_unpredicted cpu vs gpu", out_cpu, out_gpu, atol=1e-6, rtol=1e-5, verbose=verbose)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_history_compute_stats_cpu_gpu_match(self, verbose=False):
        """history_compute_stats produces matching mean/std on CPU and GPU (mean mode)."""
        pp_cpu = self._make_pp_mean(n_history=1)
        pp_gpu = self._make_pp_mean(n_history=1).to("cuda")
        T = pp_cpu.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp_cpu.history_compute_stats(x)
        pp_gpu.history_compute_stats(x.to("cuda"))
        self.assertTrue(
            compare_tensors(
                "history_mean cpu vs gpu",
                pp_cpu.history_mean,
                pp_gpu.history_mean.cpu(),
                atol=1e-5,
                rtol=1e-5,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "history_std cpu vs gpu",
                pp_cpu.history_std,
                pp_gpu.history_std.cpu(),
                atol=1e-5,
                rtol=1e-5,
                verbose=verbose,
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_history_normalize_cpu_gpu_match(self, verbose=False):
        """history_normalize produces matching output on CPU and GPU (mean mode)."""
        pp_cpu = self._make_pp_mean(n_history=1)
        pp_gpu = self._make_pp_mean(n_history=1).to("cuda")
        T = pp_cpu.n_history + 1
        x = self._rand(self.B, T * self.C, self.H, self.W)
        pp_cpu.history_compute_stats(x)
        pp_gpu.history_compute_stats(x.to("cuda"))
        xn_cpu = pp_cpu.history_normalize(x)
        xn_gpu = pp_gpu.history_normalize(x.to("cuda")).cpu()
        self.assertTrue(
            compare_tensors("history_normalize cpu vs gpu", xn_cpu, xn_gpu, atol=1e-5, rtol=1e-4, verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # Hardening: controlled-fail shape / batch checks
    # -----------------------------------------------------------------------

    def test_expand_history_raises_on_non_divisible_channels(self):
        """expand_history raises when the channel dim isn't divisible by nhist.

        The check is a torch._check (compile-safe), so it surfaces as RuntimeError.
        """
        params = get_default_parameters()
        params.n_history = 2  # nhist = 3
        pp = Preprocessor2D(params)
        # channels = 5, not divisible by 3
        x_bad = torch.randn(self.B, 5, self.H, self.W)
        with self.assertRaises(RuntimeError) as cm:
            pp.expand_history(x_bad, nhist=pp.n_history + 1)
        self.assertIn("not divisible", str(cm.exception))

    def test_append_channels_raises_on_input_cache_batch_mismatch(self):
        """_append_channels raises when input and cached xz have different batch sizes.

        The check is a torch._check (compile-safe), so it surfaces as RuntimeError.
        """
        self.pp.train()
        # cache xz/yz at the preprocessor's configured batch
        x_cache = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x_cache, x_cache, xz=xz, yz=yz)
        # now pass an input with a DIFFERENT batch size
        x_mismatch = self._rand(self.B + 1, self.C, self.H, self.W)
        with self.assertRaises(RuntimeError) as cm:
            self.pp.append_unpredicted_features(x_mismatch, target=False)
        self.assertIn("batch mismatch", str(cm.exception))
        self.assertIn("cached unpredicted features", str(cm.exception))

    def test_append_channels_raises_on_noise_state_batch_mismatch(self):
        """_append_channels raises when input_noise state and input batch disagree.

        The check is a torch._check (compile-safe), so it surfaces as RuntimeError.
        """
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": "concatenate", "n_channels": 1}
        pp = Preprocessor2D(params)
        pp.train()
        # cache xz at batch B and pass input at batch B so the first check passes;
        # bump noise state to a different batch to hit the second (noise) check.
        x = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        pp.cache_unpredicted_features(x, x, xz=xz, yz=yz)
        pp.update_internal_state(replace_state=True, batch_size=self.B + 2)

        with self.assertRaises(RuntimeError) as cm:
            pp.append_unpredicted_features(x, target=False)
        msg = str(cm.exception)
        self.assertIn("batch mismatch", msg)
        self.assertIn("input_noise state", msg)


if __name__ == "__main__":
    unittest.main()
