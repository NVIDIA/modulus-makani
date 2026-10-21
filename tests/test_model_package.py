# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Tests for packaging and loading models: model packages and the wrappers they
are built from, the steppers, the model registry entry points and PhysicsNeMo
compatibility, and loading parameters that predate the resampled shapes.
"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import sys
from parameterized import parameterized
import warnings
from importlib.metadata import entry_points

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.models.model_package import (
    LocalPackage,
    ModelWrapper,
    load_model_package,
    load_time_loop,
    save_model_package,
)
from makani.models.stepper import StepWrapper
from makani.utils.driver import Driver
from makani.utils.dataloaders.batch import make_batch
from makani.utils.dataloaders.rollout import Rollout
from .testutils import set_seed, get_default_parameters, compare_tensors, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W
from makani.utils.YParams import ParamsBase, ensure_resampled_shapes
from makani.models import model_registry

# earth2mip is optional; load_time_loop builds one of its TimeLoops
_have_earth2mip = importlib.util.find_spec("earth2mip") is not None

# ---------------------------------------------------------------------------
# Model packages: makani.models.model_package
# ---------------------------------------------------------------------------


class _LeadingChannelsModel(nn.Module):
    """
    Dummy backbone returning ``scale`` times the leading ``n_out_chans`` channels.

    The wrapper may append dynamic (zenith) and static features, so we slice
    from the FRONT -- those are the data channels of the oldest history step and
    are unaffected by whatever gets appended behind them.
    """

    def __init__(self, n_out_chans: int, scale: float = 2.0):
        super().__init__()
        self.n_out_chans = n_out_chans
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return self.scale * x[..., : self.n_out_chans, :, :]

    def encode_process(self, x):
        return self.forward(x)


class _ModelPackageTestBase(unittest.TestCase):
    """Builds a ModelWrapper around a dummy backbone without a real package on disk."""

    @classmethod
    def setUpClass(cls):
        # ModelWrapper needs real normalization arrays; "none" makes
        # get_data_normalization return (None, None), which the wrapper cannot index.
        cls.tmpdir = tempfile.mkdtemp()
        means = np.zeros((1, NUM_CHANNELS, 1, 1), dtype=np.float64)
        stds = np.ones((1, NUM_CHANNELS, 1, 1), dtype=np.float64)
        cls.means_path = os.path.join(cls.tmpdir, "global_means.npy")
        cls.stds_path = os.path.join(cls.tmpdir, "global_stds.npy")
        np.save(cls.means_path, means)
        np.save(cls.stds_path, stds)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        set_seed(333)
        self.C = NUM_CHANNELS
        self.H = IMG_SIZE_H
        self.W = IMG_SIZE_W

    def _make_params(self, n_history=0, add_zenith=True, input_noise=None):
        params = get_default_parameters()
        params.n_history = n_history
        params.add_zenith = add_zenith
        params.dhours = 6
        # identity normalization (mean 0, std 1) so forward stays an exact comparison
        params.normalization = "zscore"
        params.global_means_path = self.means_path
        params.global_stds_path = self.stds_path
        if input_noise is not None:
            params.input_noise = input_noise
        return params

    def _make_wrapper(self, n_history=0, add_zenith=True, input_noise=None):
        params = self._make_params(n_history=n_history, add_zenith=add_zenith, input_noise=input_noise)
        stepper = StepWrapper(params, lambda: _LeadingChannelsModel(n_out_chans=NUM_CHANNELS, scale=2.0))
        wrapper = ModelWrapper(stepper, params)
        # the packaged model is always used in eval mode
        wrapper.eval()
        return wrapper

    def _times(self, n):
        base = dt.datetime(2020, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
        return np.array([base + dt.timedelta(hours=6 * i) for i in range(n)])


class TestZenithFeatureShapes(_ModelPackageTestBase):
    """
    The zenith feature must be (B, n_history+1, 1, H, W).

    That is a dynamic feature as the preprocessor takes it: batch, time and
    channel leading, the grid trailing. Getting either of the two leading axes
    wrong trips the batch or the time-window check when the step wrapper
    assembles its input.
    """

    def _check(self, n_history, batch, n_times):
        wrapper = self._make_wrapper(n_history=n_history)
        x = torch.randn(batch, (n_history + 1) * self.C, self.H, self.W)
        z = wrapper._zenith_features(x, self._times(n_times))
        self.assertEqual(tuple(z.shape), (batch, n_history + 1, 1, self.H, self.W))

    def test_single_sample_no_history(self):
        self._check(n_history=0, batch=1, n_times=1)

    def test_batched_one_time_per_member(self):
        # the case main got wrong: dim 0 must be batch, not a prepended singleton
        self._check(n_history=0, batch=4, n_times=4)

    def test_batched_shared_time(self):
        # a perturbed-IC ensemble: every member valid at the same time
        self._check(n_history=0, batch=4, n_times=1)

    def test_single_sample_with_history(self):
        # this worked before the latent-API PR and must keep working: dim 1 is history
        self._check(n_history=2, batch=1, n_times=3)

    def test_batched_with_history_per_member(self):
        # B * nhist times, ordered member-major
        self._check(n_history=2, batch=4, n_times=12)

    def test_batched_with_history_shared_window(self):
        # one history window broadcast across the batch
        self._check(n_history=2, batch=4, n_times=3)

    def test_member_major_ordering(self):
        # with B*nhist times the reshape must group by member, not interleave them:
        # member b's history window is times [b*nhist : (b+1)*nhist]
        n_history, batch = 2, 3
        nhist = n_history + 1
        wrapper = self._make_wrapper(n_history=n_history)
        x = torch.randn(batch, nhist * self.C, self.H, self.W)
        times = self._times(batch * nhist)

        z = wrapper._zenith_features(x, times)

        from makani.third_party.climt.zenith_angle_v2 import cos_zenith_angle

        for b in range(batch):
            expected = cos_zenith_angle(times[b * nhist : (b + 1) * nhist], wrapper.lon_grid, wrapper.lat_grid)
            self.assertTrue(
                compare_tensors(
                    f"member_major_b{b}",
                    z[b, :, 0],
                    torch.as_tensor(expected.astype(np.float32)),
                    verbose=False,
                )
            )

    def test_shared_window_is_identical_across_members(self):
        wrapper = self._make_wrapper(n_history=1)
        x = torch.randn(5, 2 * self.C, self.H, self.W)
        z = wrapper._zenith_features(x, self._times(2))
        for b in range(1, 5):
            self.assertTrue(torch.equal(z[0], z[b]))

    def test_mismatched_time_count_raises(self):
        # 5 times for batch 4 / nhist 1 matches neither convention
        wrapper = self._make_wrapper(n_history=0)
        x = torch.randn(4, self.C, self.H, self.W)
        with self.assertRaises(ValueError) as cm:
            wrapper._zenith_features(x, self._times(5))
        # the message must name both acceptable counts (4 per-member, 1 shared)
        msg = str(cm.exception)
        self.assertIn("add_zenith", msg)
        self.assertIn("Pass either 4 times", msg)

    def test_input_with_time_axis_is_accepted(self):
        # the history may come flattened into the channels or on its own axis, with
        # or without the zenith angle; the step wrapper gets it with the time axis
        for add_zenith in (True, False):
            with self.subTest(add_zenith=add_zenith):
                wrapper = self._make_wrapper(n_history=1, add_zenith=add_zenith)
                x = torch.randn(2, 2, self.C, self.H, self.W)
                out = wrapper._prepare_input(x, self._times(4), normalized_data=True)
                self.assertTrue(compare_tensors("time axis kept", out, x, verbose=False))

    def test_flat_input_gets_its_time_axis(self):
        wrapper = self._make_wrapper(n_history=1)
        x = torch.randn(2, 2 * self.C, self.H, self.W)
        out = wrapper._prepare_input(x, self._times(4), normalized_data=True)
        self.assertEqual(tuple(out.shape), (2, 2, self.C, self.H, self.W))
        self.assertTrue(compare_tensors("unfolded", out[:, 1], x[:, self.C :], verbose=False))


class TestModelPackageForward(_ModelPackageTestBase):
    """End-to-end: the wrapper must run for any batch size, with and without zenith."""

    def _run(self, wrapper, batch, n_history, n_times, **kwargs):
        x = torch.randn(batch, (n_history + 1) * self.C, self.H, self.W)
        out = wrapper(x, self._times(n_times), **kwargs)
        self.assertEqual(tuple(out.shape), (batch, self.C, self.H, self.W))
        # identity normalization + leading-channel dummy => exactly 2 * leading slice
        self.assertTrue(compare_tensors("model_package_forward", out, 2.0 * x[:, : self.C], verbose=False))

    def test_forward_batch_one(self):
        self._run(self._make_wrapper(), batch=1, n_history=0, n_times=1)

    def test_forward_batched(self):
        # the whole point: B > 1 through the packaged path
        self._run(self._make_wrapper(), batch=4, n_history=0, n_times=4)

    def test_forward_batched_shared_time(self):
        self._run(self._make_wrapper(), batch=4, n_history=0, n_times=1)

    def test_forward_with_history(self):
        self._run(self._make_wrapper(n_history=2), batch=1, n_history=2, n_times=3)

    def test_forward_batched_with_history(self):
        self._run(self._make_wrapper(n_history=2), batch=3, n_history=2, n_times=9)

    def test_forward_without_zenith_ignores_time(self):
        wrapper = self._make_wrapper(add_zenith=False)
        self._run(wrapper, batch=4, n_history=0, n_times=1)

    def test_forward_denormalizes_when_not_normalized(self):
        # with mean 0 / std 1 the round trip is the identity, so the result is unchanged
        wrapper = self._make_wrapper()
        x = torch.randn(2, self.C, self.H, self.W)
        out = wrapper(x, self._times(2), normalized_data=False)
        self.assertTrue(compare_tensors("denorm_roundtrip", out, 2.0 * x, verbose=False))

    def test_normalization_with_history(self):
        # the statistics are per channel and have to be applied to every history step
        wrapper = self._make_wrapper(n_history=1)
        means = torch.arange(self.C, dtype=torch.float32).reshape(1, self.C, 1, 1)
        stds = (1.0 + torch.arange(self.C, dtype=torch.float32)).reshape(1, self.C, 1, 1)
        wrapper.in_bias.copy_(means)
        wrapper.in_scale.copy_(stds)

        x = torch.randn(2, 2 * self.C, self.H, self.W)
        out = wrapper._prepare_input(x, self._times(4), normalized_data=False)

        expected = torch.stack([(x[:, : self.C] - means) / stds, (x[:, self.C :] - means) / stds], dim=1)
        self.assertTrue(compare_tensors("normalization_with_history", out, expected, verbose=False))

    def test_encode_process_matches_forward_preprocessing(self):
        # the dummy's encode_process mirrors its forward, so with an identity
        # denormalization the staged path must reproduce forward exactly
        wrapper = self._make_wrapper()
        x = torch.randn(4, self.C, self.H, self.W)
        times = self._times(4)

        features = wrapper.encode_process(x, times)
        expected = wrapper(x, times)

        self.assertTrue(compare_tensors("model_package_encode_process", features, expected, verbose=False))


class TestNoiseBatchGuard(_ModelPackageTestBase):
    """
    Resizing a stateful noise state mid-sequence must fail loudly.

    The state carries an n_history+1 AR history; reallocating zeroes it, so
    continuing the sequence would silently restart the noise from zero instead of
    the intended stationary distribution.
    """

    DIFFUSION = {"type": "diffusion", "mode": "concatenate", "n_channels": 1}
    WHITE = {"type": "white", "mode": "concatenate", "n_channels": 1}

    def test_resize_during_ar_sequence_raises(self):
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        pp = wrapper.model.preprocessor
        self.assertEqual(pp.input_noise.state.shape[0], 1)

        with self.assertRaises(RuntimeError) as cm:
            pp.update_internal_state(replace_state=False, batch_size=4)
        self.assertIn("replace_state=True", str(cm.exception))

    def test_resize_with_replace_state_is_allowed(self):
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        pp = wrapper.model.preprocessor

        pp.update_internal_state(replace_state=True, batch_size=4)

        self.assertEqual(pp.input_noise.state.shape[0], 4)
        # the history axis is config-derived and must be untouched by the resize
        self.assertEqual(pp.input_noise.state.shape[1], pp.n_history + 1)

    def test_ar_step_at_fixed_batch_is_allowed(self):
        # this is the ensemble/inference pattern: prime once, then roll forward
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        pp = wrapper.model.preprocessor

        pp.update_internal_state(replace_state=True, batch_size=4)
        pp.update_internal_state(replace_state=False, batch_size=4)

        self.assertEqual(pp.input_noise.state.shape[0], 4)

    def test_stateless_noise_is_not_guarded(self):
        # white noise redraws every step, so a resize destroys nothing
        wrapper = self._make_wrapper(input_noise=self.WHITE)
        pp = wrapper.model.preprocessor
        self.assertFalse(pp.input_noise.is_stateful())

        pp.update_internal_state(replace_state=False, batch_size=4)

        self.assertEqual(pp.input_noise.state.shape[0], 4)

    def test_batched_forward_after_priming(self):
        # the documented recipe: prime at the target batch, then run batched
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        wrapper.update_state(replace_state=True, batch_size=4)

        x = torch.randn(4, self.C, self.H, self.W)
        out = wrapper(x, self._times(4), replace_state=False)

        self.assertEqual(tuple(out.shape), (4, self.C, self.H, self.W))

    def test_batched_forward_without_priming_raises(self):
        # noise state is still at params.batch_size == 1; forward defaults to
        # replace_state=None (falsy), so this must be refused rather than silently
        # restarting the AR sequence from zero
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        x = torch.randn(4, self.C, self.H, self.W)

        with self.assertRaises(RuntimeError):
            wrapper(x, self._times(4))


class TestSaveModelPackage(unittest.TestCase):
    """A synthetic-data run has no statistics or invariants to package."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

        self.params = get_default_parameters()
        self.params.experiment_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_synthetic_data_writes_nothing(self):
        self.params.enable_synthetic_data = True
        self.params.invariants = [{"channel": "z", "encoding": "normalize"}]
        self.params.invariants_path = "/nonexistent/invariants.h5"

        save_model_package(self.params)

        self.assertEqual(os.listdir(self.tmpdir.name), [])

    def test_real_data_still_writes_the_config(self):
        self.params.enable_synthetic_data = False

        save_model_package(self.params)

        self.assertIn("config.json", os.listdir(self.tmpdir.name))


class TestLoadStaticData(unittest.TestCase):
    """LocalPackage._load_static_data points the params at the package's canonical files."""

    def test_set_paths_are_redirected_into_the_package(self):
        package = LocalPackage("/package")
        params = ParamsBase()
        params.invariants = [{"channel": "z", "encoding": "normalize"}]
        params.invariants_path = "/elsewhere/orography.h5"
        params.global_means_path = "/elsewhere/means.npy"
        params.global_stds_path = "/elsewhere/stds.npy"
        params.min_path = "/elsewhere/mins.npy"
        params.max_path = "/elsewhere/maxs.npy"

        LocalPackage._load_static_data(package, params)

        self.assertEqual(params.invariants_path, os.path.join("/package", LocalPackage.INVARIANTS_FILE))
        self.assertEqual(params.global_means_path, os.path.join("/package", LocalPackage.MEANS_FILE))
        self.assertEqual(params.global_stds_path, os.path.join("/package", LocalPackage.STDS_FILE))
        self.assertEqual(params.min_path, os.path.join("/package", LocalPackage.MINS_FILE))
        self.assertEqual(params.max_path, os.path.join("/package", LocalPackage.MAXS_FILE))

    def test_unset_paths_stay_unset(self):
        # the package only holds the files the run had, so absent statistics must not appear
        package = LocalPackage("/package")
        params = ParamsBase()
        params.invariants = None
        params.global_means_path = "/elsewhere/means.npy"
        params.global_stds_path = "/elsewhere/stds.npy"
        params.min_path = None

        LocalPackage._load_static_data(package, params)

        self.assertFalse(hasattr(params, "invariants_path"))
        self.assertIsNone(params.min_path)
        self.assertIsNone(params.get("max_path", None))
        self.assertEqual(params.global_means_path, os.path.join("/package", LocalPackage.MEANS_FILE))


class TestLoadModelPackage(_ModelPackageTestBase):
    """
    save_model_package followed by load_model_package gives back the same model.

    The package is written the way training writes it: the config and the statistics through
    save_model_package, the weights as the legacy checkpoint at the canonical package path.
    The statistics come from files with non-canonical names, so the round trip also checks that
    they are copied into the package and found there again.
    """

    def setUp(self):
        super().setUp()
        self.pkgdir = tempfile.TemporaryDirectory()
        self.srcdir = tempfile.TemporaryDirectory()

        # non-trivial statistics, so normalization errors show in the output
        means = np.arange(self.C, dtype=np.float64).reshape(1, self.C, 1, 1)
        stds = (1.0 + 0.5 * np.arange(self.C, dtype=np.float64)).reshape(1, self.C, 1, 1)
        self.means_src = os.path.join(self.srcdir.name, "my_means.npy")
        self.stds_src = os.path.join(self.srcdir.name, "my_stds.npy")
        self.mins_src = os.path.join(self.srcdir.name, "my_mins.npy")
        self.maxs_src = os.path.join(self.srcdir.name, "my_maxs.npy")
        np.save(self.means_src, means)
        np.save(self.stds_src, stds)
        np.save(self.mins_src, means - 3.0 * stds)
        np.save(self.maxs_src, means + 3.0 * stds)

        params = get_default_parameters()
        params.nettype = "SFNO"
        params.embed_dim = 16
        params.num_layers = 2
        params.normalization = "zscore"
        params.global_means_path = self.means_src
        params.global_stds_path = self.stds_src
        params.min_path = self.mins_src
        params.max_path = self.maxs_src
        params.data_channel_names = list(params.channel_names)
        params.dhours = 6
        params.img_crop_shape_x = params.img_shape_x
        params.img_crop_shape_y = params.img_shape_y
        params.img_crop_offset_x = 0
        params.img_crop_offset_y = 0
        params.experiment_dir = self.pkgdir.name
        self.params = params

        self.package = LocalPackage(self.pkgdir.name)

        # the package config is written first, as json.dumps fails on anything the model build
        # might add to the params
        save_model_package(params)

        self.model = model_registry.get_model(params)
        # move the weights away from any fresh initialization, so a model that was not restored
        # cannot match by chance
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(0.1 * torch.randn_like(p))

        checkpoint_path = self.package.get(LocalPackage.MODEL_PACKAGE_CHECKPOINT_PATH)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        Driver.save_checkpoint(checkpoint_path, self.model)

    def tearDown(self):
        self.pkgdir.cleanup()
        self.srcdir.cleanup()

    def _reference_wrapper(self):
        wrapper = ModelWrapper(self.model, self.params)
        wrapper.eval()
        return wrapper

    def test_package_holds_the_canonical_files(self):
        files = os.listdir(self.pkgdir.name)
        for fname in [
            "config.json",
            "metadata.json",
            LocalPackage.MEANS_FILE,
            LocalPackage.STDS_FILE,
            LocalPackage.MINS_FILE,
            LocalPackage.MAXS_FILE,
        ]:
            self.assertIn(fname, files)

    def test_params_survive(self):
        loaded = load_model_package(self.package, pretrained=True, device="cpu")

        self.assertEqual(loaded.params.nettype, "SFNO")
        self.assertEqual(loaded.params.embed_dim, 16)
        self.assertEqual(loaded.params.num_layers, 2)
        self.assertEqual(loaded.params.channel_names, self.params.channel_names)
        self.assertEqual(loaded.params.in_channels, self.params.in_channels)
        self.assertEqual(loaded.params.out_channels, self.params.out_channels)
        # the statistics are read from the package, not from where the training run had them
        self.assertEqual(loaded.params.global_means_path, self.package.get(LocalPackage.MEANS_FILE))
        self.assertEqual(loaded.params.global_stds_path, self.package.get(LocalPackage.STDS_FILE))
        self.assertEqual(loaded.params.min_path, self.package.get(LocalPackage.MINS_FILE))
        self.assertEqual(loaded.params.max_path, self.package.get(LocalPackage.MAXS_FILE))

    def test_weights_and_statistics_survive(self):
        loaded = load_model_package(self.package, pretrained=True, device="cpu")
        reference = self._reference_wrapper()

        self.assertFalse(loaded.training)

        state_ref = self.model.state_dict()
        state_loaded = loaded.model.state_dict()
        self.assertEqual(sorted(state_loaded.keys()), sorted(state_ref.keys()))
        for key, tensor in state_ref.items():
            self.assertTrue(torch.equal(state_loaded[key], tensor), key)

        for name in ["in_bias", "in_scale", "out_bias", "out_scale"]:
            self.assertTrue(torch.equal(getattr(loaded, name), getattr(reference, name)), name)

    def test_forward_matches_the_saved_model(self):
        loaded = load_model_package(self.package, pretrained=True, device="cpu")
        reference = self._reference_wrapper()

        x = torch.randn(2, self.C, self.H, self.W)
        times = self._times(2)
        with torch.no_grad():
            out_ref = reference(x, times, normalized_data=False)
            out_loaded = loaded(x, times, normalized_data=False)

        self.assertTrue(compare_tensors("load_model_package_forward", out_loaded, out_ref, shape_check=True))

    def test_without_pretrained_the_weights_are_not_restored(self):
        loaded = load_model_package(self.package, pretrained=False, device="cpu")

        state_ref = self.model.state_dict()
        differs = any(
            not torch.equal(t, state_ref[k]) for k, t in loaded.model.state_dict().items() if t.is_floating_point()
        )
        self.assertTrue(differs)

    def test_multistep_is_deprecated_but_loads_the_same_model(self):
        with self.assertWarns(FutureWarning):
            loaded = load_model_package(self.package, pretrained=True, device="cpu", multistep=True)
        reference = self._reference_wrapper()

        x = torch.randn(1, self.C, self.H, self.W)
        times = self._times(1)
        with torch.no_grad():
            out_ref = reference(x, times)
            out_loaded = loaded(x, times)

        self.assertTrue(compare_tensors("load_model_package_multistep", out_loaded, out_ref, shape_check=True))

    @unittest.skipUnless(_have_earth2mip, "earth2mip is not installed")
    def test_load_time_loop(self):
        from earth2mip.networks import Inference

        time_loop = load_time_loop(self.package, device="cpu")

        self.assertIsInstance(time_loop, Inference)
        self.assertEqual(list(time_loop.channel_names), self.params.data_channel_names)
        self.assertEqual(time_loop.n_history, self.params.n_history)

    @unittest.skipUnless(_have_earth2mip, "earth2mip is not installed")
    def test_load_time_loop_rejects_unequal_channels(self):
        # the time loop feeds the output back in, so it needs the same channels in and out
        with open(self.package.get("config.json")) as f:
            config = json.load(f)
        config["out_channels"] = config["out_channels"][:-1]
        with open(self.package.get("config.json"), "w") as f:
            json.dump(config, f)

        with self.assertRaises(NotImplementedError):
            load_time_loop(self.package, device="cpu")


# ---------------------------------------------------------------------------
# Steppers: makani.models.stepper
# ---------------------------------------------------------------------------


class _ScaleModel(nn.Module):
    """
    Dummy model that scales the most-recent timestep slice by a learnable factor.

    The wrapper packs (B, n_history+1, C, H, W) into (B, (n_history+1)*C, H, W)
    for the network and expects (B, C, H, W) back, so we slice the trailing C
    channels (the latest timestep, oldest first) and multiply by ``scale``. With
    scale=2 the rollout produces the geometric sequence
    pred_k = 2^(k+1) * (latest step of the initial input), which makes every
    assertion in this file an exact equality check.
    """

    def __init__(self, n_out_chans: int, scale: float = 2.0):
        super().__init__()
        self.n_out_chans = n_out_chans
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return self.scale * x[..., -self.n_out_chans :, :, :]


class _StagedScaleModel(_ScaleModel):
    def encode_process(self, x):
        return self.scale * x[..., -self.n_out_chans :, :, :]


class TestStepper(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.B = 1
        self.C = NUM_CHANNELS
        self.H = IMG_SIZE_H
        self.W = IMG_SIZE_W

    def _make_params(self, n_history=0, n_future=0, push_forward=False):
        params = get_default_parameters()
        params.n_history = n_history
        params.n_future = n_future
        params.multistep = {"push_forward": push_forward}
        return params

    def _make_handle(self):
        return lambda: _ScaleModel(n_out_chans=self.C, scale=2.0)

    def _inp(self, n_steps=1, batch=None, ones=False):
        shape = (batch or self.B, n_steps, self.C, self.H, self.W)
        return torch.ones(*shape) if ones else torch.randn(*shape)

    def _rollout(self, inp, n_steps):
        # a rollout from the window inp over n_steps; the targets only set the number of steps
        batch, n_window = inp.shape[:2]
        return Rollout(
            make_batch(
                inp=inp,
                inp_timestamp=torch.zeros(batch, n_window, dtype=torch.float64),
                tar=torch.zeros(batch, n_steps, *inp.shape[2:]),
                tar_timestamp=torch.zeros(batch, n_steps, dtype=torch.float64),
            )
        )

    # ------------------------------------------------------------------
    # a single step: predict and encode
    # ------------------------------------------------------------------

    def test_single_step_no_history(self):
        params = self._make_params(n_history=0)
        wrapper = StepWrapper(params, self._make_handle())
        wrapper.train()
        inp = self._inp()
        out = wrapper.predict(inp, {})
        self.assertEqual(out.shape, inp.shape)
        self.assertTrue(compare_tensors("single_step_no_history", out, 2.0 * inp, verbose=False))

    def test_single_step_with_history(self):
        n_history = 2
        params = self._make_params(n_history=n_history)
        wrapper = StepWrapper(params, self._make_handle())
        wrapper.train()
        inp = self._inp(n_history + 1)
        out = wrapper.predict(inp, {})
        self.assertEqual(out.shape, (self.B, 1, self.C, self.H, self.W))
        # only the most-recent timestep is consumed by the dummy model
        self.assertTrue(compare_tensors("single_step_with_history", out, 2.0 * inp[:, -1:], verbose=False))

    def test_single_step_batched(self):
        # any batch, not only params.batch_size (1), e.g. an ensemble folded into the batch
        params = self._make_params(n_history=0)
        wrapper = StepWrapper(params, self._make_handle())
        batch = 4
        inp = self._inp(batch=batch)

        out = wrapper.predict(inp, {})

        self.assertEqual(out.shape, (batch, 1, self.C, self.H, self.W))
        self.assertTrue(compare_tensors("single_step_batched", out, 2.0 * inp, verbose=False))

    def test_steppers_need_the_time_axis(self):
        # a flat input would be packed with the wrong layout, so it is refused
        wrapper = StepWrapper(self._make_params(n_history=1), self._make_handle())
        with self.assertRaises(RuntimeError):
            wrapper.predict(torch.randn(self.B, 2 * self.C, self.H, self.W), {})

    def test_forward_takes_a_rollout(self):
        # a tensor is not a rollout; a single step goes through predict
        wrapper = StepWrapper(self._make_params(), self._make_handle())
        with self.assertRaisesRegex(TypeError, "predict"):
            wrapper(self._inp())

    def test_encode_uses_the_preprocessing_of_predict(self):
        # encode must agree with predict on everything up to the decoder.
        # _StagedScaleModel.encode_process mirrors its forward, so with a
        # bias-correction/denormalization no-op config the two must match exactly.
        params = self._make_params(n_history=0)
        wrapper = StepWrapper(params, lambda: _StagedScaleModel(n_out_chans=self.C, scale=2.0))
        inp = self._inp()

        # the backbone's latent features are flat, the prediction has its time axis
        features = wrapper.encode(inp, {})
        expected = wrapper.predict(inp, {}).squeeze(1)

        self.assertEqual(features.shape, (self.B, self.C, self.H, self.W))
        self.assertTrue(compare_tensors("encode", features, expected, verbose=False))

    def test_encode_batched(self):
        params = self._make_params(n_history=0)
        wrapper = StepWrapper(params, lambda: _StagedScaleModel(n_out_chans=self.C, scale=2.0))
        batch = 4
        inp = self._inp(batch=batch)

        features = wrapper.encode(inp, {})

        self.assertEqual(features.shape, (batch, self.C, self.H, self.W))
        self.assertTrue(compare_tensors("encode_batched", features, 2.0 * inp[:, 0], verbose=False))

    def test_encode_unsupported_backbone(self):
        # a backbone without encode_process must fail with a clear error rather
        # than an AttributeError from deep inside the call
        params = self._make_params(n_history=0)
        wrapper = StepWrapper(params, self._make_handle())

        with self.assertRaises(NotImplementedError):
            wrapper.encode(self._inp(), {})

    # ------------------------------------------------------------------
    # a rollout
    # ------------------------------------------------------------------

    @parameterized.expand([(0,), (1,)])
    def test_rollout_geometric_sequence(self, n_history):
        n_future = 3
        params = self._make_params(n_history=n_history, n_future=n_future)
        wrapper = StepWrapper(params, self._make_handle())
        wrapper.train()

        inp = self._inp(n_history + 1)
        out = wrapper(self._rollout(inp, n_future + 1))

        # rollout output: (B, n_future+1, C, H, W), one step per rollout step
        self.assertEqual(out.shape, (self.B, n_future + 1, self.C, self.H, self.W))

        # step k must equal 2^(k+1) * (latest step of inp): each step scales the
        # previous prediction by 2, and the rollout places that prediction at
        # the most-recent position for the next step
        last = inp[:, -1]
        for k in range(n_future + 1):
            block = out[:, k]
            self.assertTrue(
                compare_tensors(
                    f"rollout_step_{k}_h{n_history}",
                    block,
                    (2.0 ** (k + 1)) * last,
                    verbose=False,
                )
            )

    def test_rollout_is_the_same_in_train_and_eval(self):
        # the mode only switches the training options, the steps are the same
        params = self._make_params(n_history=0, n_future=2)
        wrapper = StepWrapper(params, self._make_handle())
        inp = self._inp()

        wrapper.train()
        out_train = wrapper(self._rollout(inp, 3))

        wrapper.eval()
        out_eval = wrapper(self._rollout(inp, 3))

        self.assertEqual(out_train.shape[1], 3)
        self.assertTrue(compare_tensors("train vs eval", out_train, out_eval, verbose=False))

    # ------------------------------------------------------------------
    # push_forward: same numerics, but a strictly truncated gradient through
    # the rollout (each step's input is detached, so gradients only flow one
    # step at a time)
    # ------------------------------------------------------------------

    def test_push_forward_matches_no_push(self):
        n_future = 2
        inp = self._inp()

        wrapper_off = StepWrapper(self._make_params(n_future=n_future, push_forward=False), self._make_handle())
        wrapper_on = StepWrapper(self._make_params(n_future=n_future, push_forward=True), self._make_handle())
        wrapper_off.train()
        wrapper_on.train()

        out_off = wrapper_off(self._rollout(inp, n_future + 1))
        out_on = wrapper_on(self._rollout(inp, n_future + 1))
        self.assertTrue(compare_tensors("push_forward_values", out_off, out_on, verbose=False))

    def test_push_forward_truncates_gradient(self):
        # use ones() so the gradient takes a known closed form and we can
        # check exact values rather than just an inequality
        n_future = 2
        inp = self._inp(ones=True)

        wrapper_off = StepWrapper(self._make_params(n_future=n_future, push_forward=False), self._make_handle())
        wrapper_on = StepWrapper(self._make_params(n_future=n_future, push_forward=True), self._make_handle())
        wrapper_off.train()
        wrapper_on.train()

        # loss = sum of all rollout outputs. With dummy model `pred = scale * last_C(inp)`
        # and inp=ones, loss reduces to (scale + scale^2 + scale^3) * B*C*H*W.
        # d/d(scale) without push_forward = (1 + 2*scale + 3*scale^2) * B*C*H*W
        # d/d(scale) with push_forward    = (1 +   scale +   scale^2) * B*C*H*W
        # (push_forward detaches each step's input, so dpred_k/dscale only sees
        #  the direct multiplication, not the chain through earlier scales)
        wrapper_off(self._rollout(inp, n_future + 1)).sum().backward()
        wrapper_on(self._rollout(inp, n_future + 1)).sum().backward()

        bchw = self.B * self.C * self.H * self.W
        scale = 2.0
        expected_off = (1.0 + 2.0 * scale + 3.0 * scale * scale) * bchw
        expected_on = (1.0 + scale + scale * scale) * bchw

        g_off = wrapper_off.model.scale.grad.item()
        g_on = wrapper_on.model.scale.grad.item()

        self.assertAlmostEqual(g_off, expected_off, places=3)
        self.assertAlmostEqual(g_on, expected_on, places=3)
        # sanity: truncating the rollout strictly reduces gradient magnitude
        self.assertGreater(g_off, g_on)


# ---------------------------------------------------------------------------
# Model entry points and PhysicsNeMo compatibility
# ---------------------------------------------------------------------------


class TestEntryPoints(unittest.TestCase):

    def setUp(self):
        self.model_entry_points = {
            entry_point.name: entry_point
            for entry_point in entry_points(group="physicsnemo.models")
            if not entry_point.value.startswith("physicsnemo.experimental.models")
        }

    @parameterized.expand(["SFNO"])
    def test_model_entry_points(self, model_name):
        """Test model entry points"""

        # Check the model entry point.
        model_ep = self.model_entry_points.get(model_name)
        with self.subTest(desc="model entry point is not None"):
            self.assertIsNotNone(model_ep)

        # Try loading the model type.
        model_type = model_ep.load()
        with self.subTest(desc="model type is not None"):
            self.assertIsNotNone(model_type)

        # Create the model.
        model = model_type()
        with self.subTest(desc="model is not None"):
            self.assertIsNotNone(model)


class TestPhysicsNeMoCompat(unittest.TestCase):
    """Guards the PhysicsNeMo 1.x/2.x compatibility contract.

    PhysicsNeMo 2.0 made ``Module.from_torch`` registration opt-in and changed
    the generated class name. Both changes are silent -- the old call still
    succeeds, it just stops registering -- so nothing else in the suite would
    catch a regression here. These tests assert the behavior makani relies on,
    which :mod:`makani.models.physicsnemo_compat` normalizes across versions.
    """

    @parameterized.expand(
        [
            ("SFNO", "makani.models.networks.sfnonet", "SphericalFourierNeuralOperatorNet"),
            ("FNO", "makani.models.networks.sfnonet", "FourierNeuralOperatorNet"),
            ("FCN3", "makani.models.networks.fourcastnet3", "AtmoSphericNeuralOperatorNet"),
            ("FCN3", "makani.models.networks.fourcastnet3_1", "AtmoSphericNeuralOperatorNet31"),
        ]
    )
    def test_registered_under_legacy_name(self, attr, module_name, torch_class_name):
        """The wrapped class keeps its 1.x name and stays in the model registry."""
        import importlib

        from makani.models.physicsnemo_compat import get_model_registry, legacy_registered_name

        ModelRegistry = get_model_registry()

        module = importlib.import_module(module_name)
        wrapped = getattr(module, attr)
        expected = legacy_registered_name(getattr(module, torch_class_name))

        with self.subTest(desc="class name matches the PhysicsNeMo 1.x name"):
            self.assertEqual(wrapped.__name__, expected)

        # Registration is what from_checkpoint resolves against; on 2.x it only
        # happens because the compat helper passes register=True.
        with self.subTest(desc="class is registered"):
            self.assertIn(expected, ModelRegistry().list_models())

    def test_metadata_does_not_set_deprecated_name(self):
        """Constructing the metadata must not emit a DeprecationWarning.

        ``ModelMetaData.name`` is deprecated and inert on 2.x. makani keeps it
        off the dataclasses and applies it via the compat helper instead.
        """
        from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNetMetaData

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SphericalFourierNeuralOperatorNetMetaData()

        offenders = [w for w in caught if issubclass(w.category, DeprecationWarning) and "name" in str(w.message)]
        self.assertEqual(offenders, [], f"metadata set a deprecated field: {[str(w.message) for w in offenders]}")


# ---------------------------------------------------------------------------
# Loading models without resampled shapes
# ---------------------------------------------------------------------------


class TestResampledShapeFallback(unittest.TestCase):
    """``img_shape_{x,y}_resampled`` must be optional outside training.

    The resampled shapes are populated at runtime from the dataset (Driver copies
    them from the dataloader), so they are absent from model packages written
    before resampling existed and from params assembled by external callers such
    as earth2studio, which do not pass input shapes at all.

    Consumers read them unconditionally -- ``model_registry.get_model``,
    ``Preprocessor2D`` and ``ModelWrapper`` -- so without a fallback loading an
    older SFNO package fails with

        AttributeError: 'ParamsBase' object has no attribute 'img_shape_x_resampled'

    Falling back to the unresampled shape is correct in exactly these cases,
    since no resampling took place.
    """

    def setUp(self):
        set_seed(333)

    def _params_without_resampled(self, nettype):
        """Params as an older package / an external caller would supply them."""
        params = get_default_parameters()
        params.nettype = nettype
        params.img_shape_x = 36
        params.img_shape_y = 72
        params.img_local_shape_x = params.img_crop_shape_x = params.img_shape_x
        params.img_local_shape_y = params.img_crop_shape_y = params.img_shape_y
        # deliberately NOT set: img_shape_x_resampled / img_shape_y_resampled
        for key in ("img_shape_x_resampled", "img_shape_y_resampled"):
            if hasattr(params, key):
                delattr(params, key)
            params.params.pop(key, None)
        return params

    # -- the helper itself ---------------------------------------------------

    def test_fills_from_unresampled(self):
        params = self._params_without_resampled("SFNO")
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 36)
        self.assertEqual(params.img_shape_y_resampled, 72)

    def test_does_not_overwrite_explicit_values(self):
        """A genuinely resampled config must survive untouched."""
        params = self._params_without_resampled("SFNO")
        params.img_shape_x_resampled = 18
        params.img_shape_y_resampled = 36
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 18)
        self.assertEqual(params.img_shape_y_resampled, 36)

    def test_is_idempotent(self):
        params = self._params_without_resampled("SFNO")
        ensure_resampled_shapes(params)
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 36)

    def test_none_is_treated_as_absent(self):
        """Driver leaves these as None when no dataset is attached."""
        params = self._params_without_resampled("SFNO")
        params.img_shape_x_resampled = None
        params.img_shape_y_resampled = None
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 36)
        self.assertEqual(params.img_shape_y_resampled, 72)

    def test_missing_both_raises_clearly(self):
        """With no shape at all, fail with an actionable message rather than an
        AttributeError from deep inside model construction."""
        params = ParamsBase()
        params.update_params({"nettype": "SFNO"})
        with self.assertRaises(AttributeError) as cm:
            ensure_resampled_shapes(params)
        self.assertIn("img_shape_x", str(cm.exception))

    # -- the reported failure ------------------------------------------------

    @parameterized.expand([("SFNO",), ("FNO",), ("FCN3",)])
    def test_get_model_without_resampled_shapes(self, nettype):
        """Regression: this is the exact path that raised for older packages."""
        params = self._params_without_resampled(nettype)
        model = model_registry.get_model(params)

        # the fallback must have populated params for the downstream consumers
        # (Preprocessor2D reads them from this same object)
        self.assertEqual(params.img_shape_x_resampled, params.img_shape_x)
        self.assertEqual(params.img_shape_y_resampled, params.img_shape_y)

        inp = torch.randn(1, 1, params.N_in_channels, params.img_shape_x, params.img_shape_y)
        out = model.predict(inp, {}).flatten(1, 2)
        self.assertEqual(out.shape, (1, params.N_out_channels, params.img_shape_x, params.img_shape_y))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
