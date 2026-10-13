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
from makani.utils.dataloaders.batch import INPUT_DYNAMIC_FEATURES, TARGET_DYNAMIC_FEATURES, ZENITH

from .testutils import set_seed, get_default_parameters, compare_tensors, IMG_SIZE_H, IMG_SIZE_W


def _make_pp(n_history=0, add_zenith=False, **overrides):
    params = get_default_parameters()
    params.n_history = n_history
    params.add_zenith = add_zenith
    for key, value in overrides.items():
        setattr(params, key, value)
    return Preprocessor2D(params)


def _dynamic(inp=None, tar=None):
    """The dynamic part of a batch, with the zenith angle as the only feature."""
    batch = {INPUT_DYNAMIC_FEATURES: {} if inp is None else {ZENITH: inp}}
    if tar is not None:
        batch[TARGET_DYNAMIC_FEATURES] = {ZENITH: tar}
    return batch


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

    def _field(self, T=1, C=None, B=None):
        return self._rand(B or self.B, T, C or self.C, self.H, self.W)

    def _zenith(self, T=1, B=None):
        return self._rand(B or self.B, T, 1, self.H, self.W)

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def test_construction_does_not_raise(self):
        """Minimal params construct a Preprocessor2D without error."""
        pp = Preprocessor2D(get_default_parameters())
        self.assertIsInstance(pp, Preprocessor2D)

    def test_dynamic_features_follow_the_config(self):
        """The model takes the zenith angle exactly when the config adds it."""
        self.assertEqual(_make_pp(add_zenith=False).dynamic_features, [])
        self.assertEqual(_make_pp(add_zenith=True).dynamic_features, [ZENITH])

    # -----------------------------------------------------------------------
    # flatten_history / expand_history, the boundary to flat channel stacks
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
        out = self.pp.flatten_history(self._field(T))
        self.assertEqual(tuple(out.shape), (self.B, T * self.C, self.H, self.W))

    def test_expand_history_5d_noop(self):
        """expand_history on a 5-D tensor returns it unchanged."""
        T = 2
        x = self._field(T)
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
    # pack_model_input and static features
    # -----------------------------------------------------------------------

    def test_pack_model_input_without_static_features_flattens(self, verbose=False):
        """Without static features, packing only folds the time steps into the channels."""
        pp = _make_pp(n_history=1)
        x = self._field(2)
        out = pp.pack_model_input(x)
        self.assertTrue(compare_tensors("pack without static", out, pp.flatten_history(x), verbose=verbose))

    def test_pack_model_input_appends_static_features_after_all_steps(self, verbose=False):
        """The static channels, which have no time axis, follow the stacked time steps."""
        pp = _make_pp(n_history=1, add_grid=True, gridtype="raw")
        x = self._field(2)
        out = pp.pack_model_input(x)
        self.assertEqual(tuple(out.shape), (self.B, 2 * self.C + 2, self.H, self.W))
        self.assertTrue(compare_tensors("steps first", out[:, : 2 * self.C], pp.flatten_history(x), verbose=verbose))
        static = pp.get_static_features().expand(self.B, -1, -1, -1)
        self.assertTrue(compare_tensors("static last", out[:, 2 * self.C :], static, verbose=verbose))

    def test_pack_model_input_needs_the_time_axis(self):
        """A flat input is refused rather than packed with the wrong layout."""
        with self.assertRaises(RuntimeError):
            self.pp.pack_model_input(self._rand(self.B, self.C, self.H, self.W))

    def test_get_static_features_returns_tensor_when_enabled(self):
        """get_static_features returns a non-None tensor when grid is enabled."""
        pp = _make_pp(add_grid=True, gridtype="raw")
        sf = pp.get_static_features()
        self.assertIsNotNone(sf)
        self.assertIsInstance(sf, torch.Tensor)

    def test_get_static_features_returns_none_when_disabled(self):
        """get_static_features returns None when no features are configured."""
        self.assertIsNone(self.pp.get_static_features())

    def test_static_feature_spatial_shape_matches_grid(self):
        """The (H, W) of the stored static feature matches the image shape."""
        pp = _make_pp(add_grid=True, gridtype="raw")
        sf = pp.get_static_features()
        self.assertEqual(sf.shape[-2], IMG_SIZE_H)
        self.assertEqual(sf.shape[-1], IMG_SIZE_W)

    # -----------------------------------------------------------------------
    # append_history
    # -----------------------------------------------------------------------

    def test_append_history_n_history_0_returns_the_prediction(self, verbose=False):
        """When n_history=0, the next input is the prediction."""
        x = self._field(1)
        pred = self._field(1)
        out = self.pp.append_history(x, pred, step=0)
        self.assertTrue(compare_tensors("append_history n_history=0", out, pred, verbose=verbose))

    def test_append_history_n_history_1_shape(self):
        """When n_history=1, the window keeps its two steps."""
        pp = _make_pp(n_history=1)
        out = pp.append_history(self._field(2), self._field(1), step=0)
        self.assertEqual(tuple(out.shape), (self.B, 2, self.C, self.H, self.W))

    def test_append_history_n_history_1_rolls_content(self, verbose=False):
        """append_history drops the oldest time step and appends the prediction as newest."""
        pp = _make_pp(n_history=1)
        x = self._field(2)
        pred = self._field(1)
        out = pp.append_history(x, pred, step=0)
        self.assertTrue(compare_tensors("newest slot", out[:, -1], pred[:, 0], verbose=verbose))
        self.assertTrue(compare_tensors("oldest slot rolled", out[:, 0], x[:, 1], verbose=verbose))

    def test_append_history_needs_the_time_axis(self):
        """Flat tensors are refused, they used to be unfolded here."""
        pp = _make_pp(n_history=1)
        with self.assertRaises(RuntimeError):
            pp.append_history(self._rand(self.B, 2 * self.C, self.H, self.W), self._field(1), step=0)
        with self.assertRaises(RuntimeError):
            pp.append_history(self._field(2), self._rand(self.B, self.C, self.H, self.W), step=0)

    # -----------------------------------------------------------------------
    # cache_dynamic_features / get_dynamic_features
    # -----------------------------------------------------------------------

    def test_get_dynamic_features_empty_without_cache(self):
        """Before caching, there are no features on either side."""
        self.pp.eval()
        self.assertEqual(self.pp.get_dynamic_features(), ({}, {}))

    def test_cache_dynamic_features_stores_and_retrieves(self, verbose=False):
        """Cached features are retrievable, in train and in eval mode."""
        pp = _make_pp(add_zenith=True)
        for mode in ["train", "eval"]:
            with self.subTest(mode=mode):
                getattr(pp, mode)()
                xz, yz = self._zenith(1), self._zenith(2)
                pp.cache_dynamic_features(_dynamic(xz, yz))
                inp, tar = pp.get_dynamic_features()
                self.assertTrue(compare_tensors(f"input ({mode})", inp[ZENITH], xz, verbose=verbose))
                self.assertTrue(compare_tensors(f"target ({mode})", tar[ZENITH], yz, verbose=verbose))

    def test_train_and_eval_caches_are_separate(self, verbose=False):
        """Evaluation does not clobber the features of an in-flight training step."""
        pp = _make_pp(add_zenith=True)
        pp.train()
        xz_train = self._zenith(1)
        pp.cache_dynamic_features(_dynamic(xz_train, self._zenith(1)))
        pp.eval()
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        pp.train()
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("train cache kept", inp[ZENITH], xz_train, verbose=verbose))

    def test_get_dynamic_features_returns_copies(self, verbose=False):
        """Changing what get_dynamic_features returns leaves the cache alone."""
        pp = _make_pp(add_zenith=True)
        xz = self._zenith(1)
        pp.cache_dynamic_features(_dynamic(xz))
        inp, _ = pp.get_dynamic_features()
        inp[ZENITH].zero_()
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("cache untouched", inp[ZENITH], xz, verbose=verbose))

    def test_cache_dynamic_features_same_shape_reuses_buffer(self):
        """A second batch of the same shape is copied into the cached memory."""
        pp = _make_pp(add_zenith=True)
        for mode in ["train", "eval"]:
            with self.subTest(mode=mode):
                getattr(pp, mode)()
                pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
                ptr1 = pp._dynamic_cache[mode]["input"][ZENITH].data_ptr()
                pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
                ptr2 = pp._dynamic_cache[mode]["input"][ZENITH].data_ptr()
                self.assertEqual(ptr1, ptr2)

    def test_cache_dynamic_features_same_shape_updates_values(self, verbose=False):
        """After the copy, the cache holds the values of the new batch."""
        pp = _make_pp(add_zenith=True)
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        xz2 = self._zenith(1)
        pp.cache_dynamic_features(_dynamic(xz2, self._zenith(1)))
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("copy path updated value", inp[ZENITH], xz2, verbose=verbose))

    def test_cache_dynamic_features_shape_change_rebinds(self):
        """When the shape changes, for example the batch size, the cache is replaced."""
        pp = _make_pp(add_zenith=True)
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        ptr1 = pp._dynamic_cache["train"]["input"][ZENITH].data_ptr()
        xz2 = self._zenith(1, B=self.B + 1)
        pp.cache_dynamic_features(_dynamic(xz2, self._zenith(1, B=self.B + 1)))
        cached = pp._dynamic_cache["train"]["input"][ZENITH]
        self.assertNotEqual(ptr1, cached.data_ptr())
        self.assertEqual(tuple(cached.shape), tuple(xz2.shape))

    def test_cache_without_target_clears_the_target_side(self):
        """An inference batch carries no target, so no stale target features remain."""
        pp = _make_pp(add_zenith=True)
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        pp.cache_dynamic_features(_dynamic(self._zenith(1)))
        _, tar = pp.get_dynamic_features()
        self.assertEqual(tar, {})

    def test_cache_rejects_a_missing_feature(self):
        """A model built with the zenith angle cannot take a batch without it."""
        pp = _make_pp(add_zenith=True)
        with self.assertRaisesRegex(ValueError, "zenith"):
            pp.cache_dynamic_features(_dynamic())

    def test_cache_rejects_an_extra_feature(self):
        """A feature the model does not take would be silently dropped."""
        with self.assertRaisesRegex(ValueError, "zenith"):
            self.pp.cache_dynamic_features(_dynamic(self._zenith(1)))

    def test_cache_rejects_a_feature_without_time_axis(self):
        """Features keep their time axis, like everything else in the preprocessor."""
        pp = _make_pp(add_zenith=True)
        with self.assertRaisesRegex(ValueError, "time axis"):
            pp.cache_dynamic_features({INPUT_DYNAMIC_FEATURES: {ZENITH: self._rand(self.B, 1, self.H, self.W)}})

    def test_cache_rejects_an_input_feature_of_the_wrong_window(self):
        """The input features span the input window, n_history + 1 steps."""
        pp = _make_pp(n_history=1, add_zenith=True)
        with self.assertRaisesRegex(ValueError, "time steps"):
            pp.cache_dynamic_features(_dynamic(self._zenith(1)))

    # -----------------------------------------------------------------------
    # append_dynamic_features
    # -----------------------------------------------------------------------

    def test_append_dynamic_features_noop_without_features(self, verbose=False):
        """Without dynamic features or noise, the input comes back unchanged."""
        for mode in ["train", "eval"]:
            with self.subTest(mode=mode):
                getattr(self.pp, mode)()
                x = self._field(1)
                out = self.pp.append_dynamic_features(x)
                self.assertTrue(compare_tensors(f"noop {mode}", out, x, verbose=verbose))

    def test_append_dynamic_features_appends_the_zenith_angle(self, verbose=False):
        """The cached feature follows the input channels of its step."""
        pp = _make_pp(add_zenith=True)
        for mode in ["train", "eval"]:
            with self.subTest(mode=mode):
                getattr(pp, mode)()
                x, xz = self._field(1), self._zenith(1)
                pp.cache_dynamic_features(_dynamic(xz, self._zenith(1)))
                out = pp.append_dynamic_features(x)
                self.assertEqual(tuple(out.shape), (self.B, 1, self.C + 1, self.H, self.W))
                self.assertTrue(compare_tensors("x first", out[:, :, : self.C], x, verbose=verbose))
                self.assertTrue(compare_tensors("zenith after", out[:, :, self.C :], xz, verbose=verbose))

    def test_append_dynamic_features_per_step_with_history(self, verbose=False):
        """With history, every step gets the feature of that step."""
        pp = _make_pp(n_history=1, add_zenith=True)
        x, xz = self._field(2), self._zenith(2)
        pp.cache_dynamic_features(_dynamic(xz))
        out = pp.append_dynamic_features(x)
        for t in range(2):
            self.assertTrue(compare_tensors(f"x step {t}", out[:, t, : self.C], x[:, t], verbose=verbose))
            self.assertTrue(compare_tensors(f"zenith step {t}", out[:, t, self.C], xz[:, t, 0], verbose=verbose))

    def test_append_dynamic_features_needs_the_cache(self):
        """A model with the zenith angle refuses to run without it."""
        pp = _make_pp(add_zenith=True)
        with self.assertRaisesRegex(RuntimeError, "cache_dynamic_features"):
            pp.append_dynamic_features(self._field(1))

    def test_append_dynamic_features_needs_the_time_axis(self):
        """A flat input is refused."""
        with self.assertRaises(RuntimeError):
            self.pp.append_dynamic_features(self._rand(self.B, self.C, self.H, self.W))

    def test_concatenate_noise_channel_count(self):
        """Concatenated noise adds its channels after the dynamic features."""
        pp = _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 2})
        x = self._field(1)
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        out = pp.append_dynamic_features(x)
        # C original + 1 zenith + 2 noise channels
        self.assertEqual(out.shape[2], self.C + 1 + 2)

    def test_channel_order_is_input_features_noise(self, verbose=False):
        """[x | dynamic features | noise], the order the network was trained with."""
        from makani.models.noise import DummyNoiseS2

        pp = _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 1})
        pp.input_noise = DummyNoiseS2(
            img_shape=[self.H, self.W], batch_size=self.B, num_channels=1, num_time_steps=1, mode="constant_random"
        )
        pp.update_internal_state()
        x, xz = self._field(1), self._zenith(1)
        pp.cache_dynamic_features(_dynamic(xz))
        out = pp.append_dynamic_features(x)
        self.assertTrue(compare_tensors("x", out[:, :, : self.C], x, verbose=verbose))
        self.assertTrue(compare_tensors("zenith", out[:, :, self.C : self.C + 1], xz, verbose=verbose))
        self.assertTrue(compare_tensors("noise", out[:, :, self.C + 1 :], pp.input_noise(), verbose=verbose))

    def test_concatenate_noise_original_channels_unchanged(self, verbose=False):
        """With concatenated noise the input channels are bit-identical to x."""
        pp = _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 1})
        x = self._field(1)
        pp.cache_dynamic_features(_dynamic(self._zenith(1)))
        out = pp.append_dynamic_features(x)
        self.assertTrue(compare_tensors("x unchanged", out[:, :, : self.C], x, verbose=verbose))

    def test_perturb_noise_channel_count_unchanged(self):
        """Perturbing noise adds no channels of its own."""
        pp = _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "perturb", "perturb_channels": ["u10m"]})
        pp.cache_dynamic_features(_dynamic(self._zenith(1)))
        out = pp.append_dynamic_features(self._field(1))
        # C original + 1 zenith, no extra noise channels
        self.assertEqual(out.shape[2], self.C + 1)

    def test_perturb_noise_zero_leaves_x_unchanged(self, verbose=False):
        """With zero DummyNoiseS2 in perturb mode, the x channels are bit-identical."""
        pp = _make_pp(input_noise={"type": "dummy", "mode": "perturb", "perturb_channels": ["u10m"]})
        x = self._field(1)
        out = pp.append_dynamic_features(x)
        self.assertTrue(compare_tensors("x unchanged with zero perturb noise", out, x, verbose=verbose))

    def test_concatenate_noise_without_dynamic_features(self, verbose=False):
        """Without the zenith angle the noise is still appended, as the channel count expects."""
        pp = _make_pp(input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 2})
        for mode in ["train", "eval"]:
            with self.subTest(mode=mode):
                getattr(pp, mode)()
                x = self._field(1)
                pp.cache_dynamic_features(_dynamic())
                out = pp.append_dynamic_features(x)
                self.assertEqual(out.shape[2], self.C + 2)
                self.assertTrue(compare_tensors("x channels unchanged", out[:, :, : self.C], x, verbose=verbose))

    def test_concatenate_noise_with_history(self):
        """With history, the noise is appended to every step."""
        pp = _make_pp(n_history=1, input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 1})
        x = self._field(2)
        out = pp.append_dynamic_features(x)
        self.assertEqual(tuple(out.shape), (self.B, 2, self.C + 1, self.H, self.W))
        self.assertTrue(torch.equal(out[:, :, : self.C], x))

    def test_stack_dynamic_features_in_model_order(self, verbose=False):
        """Models that pack their input themselves get the features in the same order."""
        pp = _make_pp(add_zenith=True)
        xz = self._zenith(1)
        self.assertTrue(compare_tensors("stacked", pp.stack_dynamic_features({ZENITH: xz}), xz, verbose=verbose))
        self.assertIsNone(self.pp.stack_dynamic_features({}))

    # -----------------------------------------------------------------------
    # append_history advancing the dynamic features
    # -----------------------------------------------------------------------

    def test_append_history_n_history_0_takes_the_target_step(self, verbose=False):
        """With n_history=0, the input feature becomes the target's feature of this step."""
        pp = _make_pp(add_zenith=True)
        yz = self._zenith(2)
        pp.cache_dynamic_features(_dynamic(self._zenith(1), yz))
        pp.append_history(self._field(1), self._field(1), step=1, update_state=True)
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("input feature = target step 1", inp[ZENITH], yz[:, 1:2], verbose=verbose))

    def test_append_history_n_history_1_rolls_newest(self, verbose=False):
        """With n_history=1, the newest step of the input feature is the target's step."""
        pp = _make_pp(n_history=1, add_zenith=True)
        yz = self._zenith(2)
        pp.cache_dynamic_features(_dynamic(self._zenith(2), yz))
        pp.append_history(self._field(2), self._field(1), step=0, update_state=True)
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("newest = target step 0", inp[ZENITH][:, -1], yz[:, 0], verbose=verbose))

    def test_append_history_n_history_1_rolls_oldest(self, verbose=False):
        """With n_history=1, the oldest step of the input feature is its former newest."""
        pp = _make_pp(n_history=1, add_zenith=True)
        xz = self._zenith(2)
        pp.cache_dynamic_features(_dynamic(xz, self._zenith(2)))
        pp.append_history(self._field(2), self._field(1), step=0, update_state=True)
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("oldest = former newest", inp[ZENITH][:, 0], xz[:, 1], verbose=verbose))

    def test_append_history_update_state_false_preserves_the_features(self, verbose=False):
        """With update_state=False, the cached features stay as they are."""
        pp = _make_pp(add_zenith=True)
        xz = self._zenith(1)
        pp.cache_dynamic_features(_dynamic(xz, self._zenith(1)))
        pp.append_history(self._field(1), self._field(1), step=0, update_state=False)
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("unchanged", inp[ZENITH], xz, verbose=verbose))

    def test_append_history_out_of_range_step_is_a_noop(self, verbose=False):
        """Steps beyond the cached target window leave the features untouched."""
        pp = _make_pp(add_zenith=True)
        xz = self._zenith(1)
        pp.cache_dynamic_features(_dynamic(xz, self._zenith(1)))
        pp.append_history(self._field(1), self._field(1), step=5, update_state=True)
        inp, _ = pp.get_dynamic_features()
        self.assertTrue(compare_tensors("unchanged after out-of-range step", inp[ZENITH], xz, verbose=verbose))

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
        return _make_pp(input_noise={"type": "dummy", "mode": input_noise_mode, "n_channels": n_channels})

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
    # history_compute_stats / history_normalize / history_denormalize
    # -----------------------------------------------------------------------

    def test_history_normalize_none_mode_returns_x_unchanged(self, verbose=False):
        """history_normalization_mode='none': normalize is a no-op."""
        # default params have mode="none"
        x = self._field(1)
        out = self.pp.history_normalize(x)
        self.assertTrue(compare_tensors("normalize noop", out, x, verbose=verbose))

    def test_history_denormalize_none_mode_returns_x_unchanged(self, verbose=False):
        """history_normalization_mode='none': denormalize is a no-op."""
        x = self._field(1)
        out = self.pp.history_denormalize(x)
        self.assertTrue(compare_tensors("denormalize noop", out, x, verbose=verbose))

    def _make_pp_mean(self, n_history=1):
        """Build a Preprocessor2D in 'mean' normalization mode."""
        return _make_pp(n_history=n_history, history_normalization_mode="mean")

    def _make_pp_exponential(self, n_history=2):
        return _make_pp(n_history=n_history, history_normalization_mode="exponential", history_normalization_decay=0.5)

    def test_history_normalization_needs_the_time_axis(self):
        """Statistics and normalization refuse flat tensors instead of unfolding them."""
        pp = self._make_pp_mean(n_history=1)
        flat = self._rand(self.B, 2 * self.C, self.H, self.W)
        with self.assertRaises(RuntimeError):
            pp.history_compute_stats(flat)
        pp.history_compute_stats(self._field(2))
        with self.assertRaises(RuntimeError):
            pp.history_normalize(flat)
        with self.assertRaises(RuntimeError):
            pp.history_denormalize(flat)

    def test_history_normalize_denormalize_roundtrip_mean_mode(self, verbose=False):
        """normalize then denormalize recovers the original tensor (mean mode, n_history=1)."""
        pp = self._make_pp_mean(n_history=1)
        x = self._field(pp.n_history + 1)
        pp.history_compute_stats(x)
        xr = pp.history_denormalize(pp.history_normalize(x))
        self.assertTrue(compare_tensors("roundtrip", xr, x, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_normalize_denormalize_roundtrip_target_mean_mode(self, verbose=False):
        """normalize(target=True) then denormalize(target=True) recovers the original (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        # the input carries one more channel than the target, as with the zenith angle
        pp.history_compute_stats(self._field(pp.n_history + 1, C=self.C + 1))
        y = self._field(1)
        yr = pp.history_denormalize(pp.history_normalize(y, target=True), target=True)
        self.assertTrue(compare_tensors("roundtrip target", yr, y, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_normalize_target_uses_the_leading_statistics(self, verbose=False):
        """A target holds the predicted channels, which lead the input channels."""
        pp = self._make_pp_mean(n_history=1)
        pp.history_compute_stats(self._field(2, C=self.C + 1))
        y = self._field(1)
        yn = pp.history_normalize(y, target=True)
        expected = (y - pp.history_mean[:, :, : self.C]) / pp.history_std[:, :, : self.C]
        self.assertTrue(compare_tensors("target statistics", yn, expected, verbose=verbose))

    def test_history_normalize_uses_the_same_statistics_for_every_step(self, verbose=False):
        """The statistics are per channel and broadcast over the time steps."""
        pp = self._make_pp_mean(n_history=1)
        x = self._field(2)
        pp.history_compute_stats(x)
        xn = pp.history_normalize(x)
        for t in range(2):
            expected = (x[:, t] - pp.history_mean[:, 0]) / pp.history_std[:, 0]
            self.assertTrue(compare_tensors(f"step {t}", xn[:, t], expected, verbose=verbose))

    def test_history_normalize_denormalize_roundtrip_exponential_mode(self, verbose=False):
        """Roundtrip holds for exponential normalization mode."""
        pp = self._make_pp_exponential(n_history=2)
        x = self._field(pp.n_history + 1)
        pp.history_compute_stats(x)
        xr = pp.history_denormalize(pp.history_normalize(x))
        self.assertTrue(compare_tensors("roundtrip exponential", xr, x, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_compute_stats_mean_is_finite(self):
        """After history_compute_stats, history_mean contains no NaN or Inf."""
        pp = self._make_pp_mean(n_history=1)
        pp.history_compute_stats(self._field(pp.n_history + 1))
        self.assertIsNotNone(pp.history_mean)
        self.assertFalse(torch.any(torch.isnan(pp.history_mean)).item())
        self.assertFalse(torch.any(torch.isinf(pp.history_mean)).item())

    def test_history_compute_stats_std_is_finite_and_nonneg(self):
        """After history_compute_stats, history_std is finite and non-negative."""
        pp = self._make_pp_mean(n_history=1)
        pp.history_compute_stats(self._field(pp.n_history + 1))
        self.assertIsNotNone(pp.history_std)
        self.assertFalse(torch.any(torch.isnan(pp.history_std)).item())
        self.assertFalse(torch.any(torch.isinf(pp.history_std)).item())
        self.assertTrue((pp.history_std >= 0).all().item())

    def test_history_compute_stats_shapes_mean_mode(self):
        """Mean and std are (B, 1, C, 1, 1), broadcasting over time and grid."""
        pp = self._make_pp_mean(n_history=1)
        pp.history_compute_stats(self._field(pp.n_history + 1))
        expected = (self.B, 1, self.C, 1, 1)
        self.assertEqual(tuple(pp.history_mean.shape), expected)
        self.assertEqual(tuple(pp.history_std.shape), expected)

    # -----------------------------------------------------------------------
    # history_compute_stats — known-answer tests
    # -----------------------------------------------------------------------

    def _check_constant_field(self, pp, value, verbose=False):
        T = pp.n_history + 1
        x = torch.full((self.B, T, self.C, self.H, self.W), value, dtype=torch.float32)
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

    def test_history_compute_stats_constant_field_mean_mode(self, verbose=False):
        """Constant field value c -> history_mean == c, history_std == 0 (mean mode)."""
        self._check_constant_field(self._make_pp_mean(n_history=1), 3.7, verbose=verbose)

    def test_history_compute_stats_constant_field_exponential_mode(self, verbose=False):
        """Constant field -> mean == value, std == 0 (exponential mode)."""
        self._check_constant_field(self._make_pp_exponential(n_history=2), -1.25, verbose=verbose)

    def test_history_compute_stats_per_channel_constants_mean_mode(self, verbose=False):
        """Per-channel constants: history_mean of channel c equals channel c's constant value."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        # every spatial/time value of channel c is (c + 1.0)
        x = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for c in range(self.C):
            x[:, :, c, :, :] = float(c + 1)
        pp.history_compute_stats(x)
        expected_mean = torch.arange(1, self.C + 1, dtype=torch.float32).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(compare_tensors("per-channel mean", pp.history_mean, expected_mean, atol=1e-5, verbose=verbose))
        self.assertTrue(
            compare_tensors(
                "per-channel std", pp.history_std, torch.zeros_like(pp.history_std), atol=1e-5, verbose=verbose
            )
        )

    def test_history_compute_stats_matches_analytical_weighted_mean_exponential(self, verbose=False):
        """Spatially-constant-per-step fields: history_mean recovers the analytical weighted temporal average."""
        pp = self._make_pp_exponential(n_history=2)
        T = pp.n_history + 1
        # spatially constant per (time, channel) but distinct across time & channel
        values = (torch.arange(T * self.C, dtype=torch.float32) + 1.0).reshape(T, self.C)
        x = values.reshape(1, T, self.C, 1, 1).expand(self.B, T, self.C, self.H, self.W).contiguous()
        pp.history_compute_stats(x)
        # reconstruct weights exactly as the preprocessor does: first element is oldest
        w = torch.exp(-0.5 * torch.arange(start=T - 1, end=-1, step=-1, dtype=torch.float32))
        w = w / w.sum()
        expected_mean = (w.unsqueeze(-1) * values).sum(dim=0).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors(
                "analytical weighted mean", pp.history_mean, expected_mean, atol=1e-5, rtol=1e-5, verbose=verbose
            )
        )

    def _linear_ramp(self, T):
        a = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float32)[: self.C]
        b = torch.tensor([1.0, -0.5, 2.0, 0.25, -1.0], dtype=torch.float32)[: self.C]
        x = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for t in range(T):
            x[:, t, :, :, :] = (a + b * t).reshape(1, self.C, 1, 1)
        return a, b, x

    def test_history_compute_stats_linear_ramp_mean_mode_mean(self, verbose=False):
        """Linear ramp v_t = a_c + b_c * t (spatially constant): history_mean matches a_c + b_c * t_bar."""
        pp = self._make_pp_mean(n_history=2)
        T = pp.n_history + 1
        a, b, x = self._linear_ramp(T)
        pp.history_compute_stats(x)
        t_bar = torch.arange(T, dtype=torch.float32).mean()
        expected_mean = (a + b * t_bar).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors("linear ramp mean", pp.history_mean, expected_mean, atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    def test_history_compute_stats_linear_ramp_mean_mode_std(self, verbose=False):
        """Linear ramp: history_std = |b_c| * sqrt(Var_t) (uniform-weight variance of 0..T-1)."""
        pp = self._make_pp_mean(n_history=2)
        T = pp.n_history + 1
        _, b, x = self._linear_ramp(T)
        pp.history_compute_stats(x)
        t_vec = torch.arange(T, dtype=torch.float32)
        var_t = ((t_vec - t_vec.mean()) ** 2).mean()
        expected_std = (b.abs() * torch.sqrt(var_t)).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors("linear ramp std", pp.history_std, expected_std, atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # history_normalize post-condition: weighted temporal mean ≈ 0, weighted L2 ≈ 1
    # -----------------------------------------------------------------------

    def _check_post_condition(self, pp, weights, verbose=False):
        x = self._field(pp.n_history + 1)
        pp.history_compute_stats(x)
        xn = pp.history_normalize(x)
        quad_xn = pp.quadrature(xn)  # (B, T, C)
        quad_xn2 = pp.quadrature(xn**2)  # (B, T, C)
        weighted_mean = (weights * quad_xn).sum(dim=1)  # (B, C)
        weighted_l2 = (weights * quad_xn2).sum(dim=1)  # (B, C)
        self.assertTrue(
            compare_tensors(
                "weighted mean ≈ 0",
                weighted_mean,
                torch.zeros_like(weighted_mean),
                atol=1e-5,
                rtol=1e-4,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "weighted L2 ≈ 1", weighted_l2, torch.ones_like(weighted_l2), atol=1e-4, rtol=1e-4, verbose=verbose
            )
        )

    def test_history_normalize_post_condition_mean_mode(self, verbose=False):
        """After normalize (mean mode): weighted temporal mean ≈ 0, weighted L2 ≈ 1, per channel."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        # mean-mode weights are uniform 1/T
        self._check_post_condition(pp, torch.full((1, T, 1), 1.0 / T), verbose=verbose)

    def test_history_normalize_post_condition_exponential_mode(self, verbose=False):
        """After normalize (exponential mode): weighted temporal mean ≈ 0, weighted L2 ≈ 1, per channel."""
        pp = self._make_pp_exponential(n_history=2)
        T = pp.n_history + 1
        # exponential weights: buffer shape is (1, T, 1, 1, 1)
        self._check_post_condition(pp, pp.history_normalization_weights.reshape(1, T, 1), verbose=verbose)

    # -----------------------------------------------------------------------
    # Noise determinism end-to-end
    # -----------------------------------------------------------------------

    def _make_pp_with_constant_random_noise(self):
        """Build a preprocessor whose input_noise is a seeded DummyNoiseS2 in constant_random mode."""
        from makani.models.noise import DummyNoiseS2

        # construct with dummy-concatenate so the preprocessor is wired for noise
        pp = _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 1})
        pp.train()
        # replace with a constant_random DummyNoise so set_rng actually matters
        pp.input_noise = DummyNoiseS2(
            img_shape=[self.H, self.W],
            batch_size=self.B,
            num_channels=1,
            num_time_steps=pp.n_history + 1,
            mode="constant_random",
        )
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        return pp

    def test_noise_determinism_same_seed_produces_equal_outputs(self, verbose=False):
        """set_rng(seed=42) twice, each followed by a forward pass, yields identical outputs."""
        pp = self._make_pp_with_constant_random_noise()
        x = self._field(1)

        pp.set_rng(reset=True, seed=42)
        pp.update_internal_state()
        out1 = pp.append_dynamic_features(x)

        pp.set_rng(reset=True, seed=42)
        pp.update_internal_state()
        out2 = pp.append_dynamic_features(x)

        self.assertTrue(compare_tensors("noise same seed", out1, out2, verbose=verbose))

    def test_noise_determinism_different_seed_produces_different_outputs(self, verbose=False):
        """Two different seeds give different outputs (the noise channels differ)."""
        pp = self._make_pp_with_constant_random_noise()
        x = self._field(1)

        pp.set_rng(reset=True, seed=42)
        pp.update_internal_state()
        out1 = pp.append_dynamic_features(x)

        pp.set_rng(reset=True, seed=999)
        pp.update_internal_state()
        out2 = pp.append_dynamic_features(x)

        self.assertFalse(compare_tensors("noise different seeds", out1, out2, verbose=verbose))

    # -----------------------------------------------------------------------
    # Gradient flow through train-path ops
    # -----------------------------------------------------------------------

    def _check_grad_flows(self, pp):
        pp.train()
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        x = self._field(1).requires_grad_(True)
        out = pp.append_dynamic_features(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        # each element of x appears exactly once in the output, so the gradient is 1
        self.assertTrue(torch.all(x.grad == 1.0).item())

    def test_append_dynamic_features_grad_flows_concatenate_noise(self):
        """Appending with concatenated noise preserves autograd w.r.t. x."""
        self._check_grad_flows(
            _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 1})
        )

    def test_append_dynamic_features_grad_flows_perturb_noise(self):
        """Appending with perturbing noise preserves autograd w.r.t. x."""
        self._check_grad_flows(
            _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "perturb", "perturb_channels": ["u10m"]})
        )

    def test_history_normalize_grad_flows_mean_mode(self):
        """history_normalize is differentiable w.r.t. its input (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        pp.history_compute_stats(self._field(T))
        x = self._field(T).requires_grad_(True)
        out = pp.history_normalize(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        # d/dx of (x - m)/s summed = 1/s broadcast; grad must be non-zero everywhere
        self.assertTrue((x.grad.abs() > 0).all().item())

    # -----------------------------------------------------------------------
    # CPU ↔ GPU numerical consistency
    # -----------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_append_dynamic_features_cpu_gpu_match(self, verbose=False):
        """append_dynamic_features matches on CPU and GPU (no noise)."""
        pp_cpu = _make_pp(add_zenith=True)
        pp_gpu = _make_pp(add_zenith=True).to("cuda")
        x, xz, yz = self._field(1), self._zenith(1), self._zenith(1)
        pp_cpu.cache_dynamic_features(_dynamic(xz, yz))
        pp_gpu.cache_dynamic_features(_dynamic(xz.to("cuda"), yz.to("cuda")))
        out_cpu = pp_cpu.append_dynamic_features(x)
        out_gpu = pp_gpu.append_dynamic_features(x.to("cuda")).cpu()
        self.assertTrue(
            compare_tensors(
                "append_dynamic_features cpu vs gpu", out_cpu, out_gpu, atol=1e-6, rtol=1e-5, verbose=verbose
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_history_compute_stats_cpu_gpu_match(self, verbose=False):
        """history_compute_stats produces matching mean/std on CPU and GPU (mean mode)."""
        pp_cpu = self._make_pp_mean(n_history=1)
        pp_gpu = self._make_pp_mean(n_history=1).to("cuda")
        x = self._field(pp_cpu.n_history + 1)
        pp_cpu.history_compute_stats(x)
        pp_gpu.history_compute_stats(x.to("cuda"))
        for name in ("history_mean", "history_std"):
            self.assertTrue(
                compare_tensors(
                    f"{name} cpu vs gpu",
                    getattr(pp_cpu, name),
                    getattr(pp_gpu, name).cpu(),
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
        x = self._field(pp_cpu.n_history + 1)
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
        pp = _make_pp(n_history=2)  # nhist = 3
        # channels = 5, not divisible by 3
        x_bad = torch.randn(self.B, 5, self.H, self.W)
        with self.assertRaises(RuntimeError) as cm:
            pp.expand_history(x_bad, nhist=pp.n_history + 1)
        self.assertIn("not divisible", str(cm.exception))

    def test_append_dynamic_features_raises_on_feature_batch_mismatch(self):
        """The input and the cached features must have the same batch size.

        The check is a torch._check (compile-safe), so it surfaces as RuntimeError.
        """
        pp = _make_pp(add_zenith=True)
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        with self.assertRaises(RuntimeError) as cm:
            pp.append_dynamic_features(self._field(1, B=self.B + 1))
        self.assertIn("different batch size", str(cm.exception))

    def test_append_dynamic_features_raises_on_noise_state_batch_mismatch(self):
        """The input_noise state and the input must have the same batch size.

        The check is a torch._check (compile-safe), so it surfaces as RuntimeError.
        """
        pp = _make_pp(add_zenith=True, input_noise={"type": "dummy", "mode": "concatenate", "n_channels": 1})
        # cache the features at batch B and pass input at batch B so the first check passes;
        # bump noise state to a different batch to hit the second (noise) check.
        pp.cache_dynamic_features(_dynamic(self._zenith(1), self._zenith(1)))
        pp.update_internal_state(replace_state=True, batch_size=self.B + 2)

        with self.assertRaises(RuntimeError) as cm:
            pp.append_dynamic_features(self._field(1))
        msg = str(cm.exception)
        self.assertIn("batch mismatch", msg)
        self.assertIn("input_noise state", msg)


if __name__ == "__main__":
    unittest.main()
