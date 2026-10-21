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
import unittest
from parameterized import parameterized_class

import numpy as np
import torch

from makani.models.noise import (
    toep,
    IsotropicGaussianRandomFieldS2,
    DiffusionNoiseS2,
    DummyNoiseS2,
)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

# -------------------------------------------------------------------------
# Common test dimensions (small enough to be fast on CPU)
# -------------------------------------------------------------------------
IMG_SHAPE = (32, 64)  # nlat, nlon
BATCH_SIZE = 2
NUM_CHANNELS = 3
NUM_TIME_STEPS = 1


# ===========================================================================
# 1. toep()
# ===========================================================================
class TestToep(unittest.TestCase):
    """Tests for the Toeplitz-matrix helper copied from scipy."""

    def test_default_shape(self):
        """toep(c) returns an (n, n) matrix for a length-n vector c."""
        T = toep([1, 2, 3])
        self.assertEqual(T.shape, (3, 3))

    def test_asymmetric_shape(self):
        """toep(c, r) returns a (len(c), len(r)) matrix."""
        T = toep([1, 2, 3], [1, 4, 5, 6])
        self.assertEqual(T.shape, (3, 4))

    def test_first_column_equals_c(self):
        """First column of toep(c) equals c."""
        c = [10, 20, 30]
        T = toep(c)
        np.testing.assert_array_equal(T[:, 0], c)

    def test_first_row_matches_r(self):
        """First row of toep(c, r) is [c[0], r[1], r[2], ...]."""
        c = [1, 2, 3]
        r = [1, 9, 8, 7]
        T = toep(c, r)
        np.testing.assert_array_equal(T[0, :], r)

    def test_symmetric_when_real_no_r(self):
        """Without an explicit r, a real-valued c produces a symmetric matrix."""
        T = toep([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(T, T.T)

    def test_scalar(self):
        """A single-element c returns a 1×1 matrix with that value."""
        T = toep([7])
        self.assertEqual(T.shape, (1, 1))
        self.assertEqual(T[0, 0], 7)


# ===========================================================================
# 2. DummyNoiseS2
# ===========================================================================
@parameterized_class(("device",), _devices)
class TestDummyNoiseS2(unittest.TestCase):
    """DummyNoiseS2 contract: finite output with the correct shape."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

        self.B = BATCH_SIZE
        self.T = NUM_TIME_STEPS
        self.C = NUM_CHANNELS
        self.noise = DummyNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=self.T,
        ).to(self.device)

    def test_is_stateful(self):
        self.assertFalse(self.noise.is_stateful())

    def test_initial_state_shape(self):
        expected = (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1])
        self.assertEqual(tuple(self.noise.state.shape), expected)

    def test_initial_state_finite(self):
        self.assertFalse(torch.isnan(self.noise.state).any())
        self.assertFalse(torch.isinf(self.noise.state).any())

    def test_forward_shape(self):
        out = self.noise.forward()
        expected = (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1])
        self.assertEqual(tuple(out.shape), expected)

    def test_forward_finite(self):
        out = self.noise.forward()
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_update_preserves_shape(self):
        self.noise.update()
        expected = (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1])
        self.assertEqual(tuple(self.noise.state.shape), expected)

    def test_update_preserves_finiteness(self):
        self.noise.update()
        self.assertFalse(torch.isnan(self.noise.state).any())
        self.assertFalse(torch.isinf(self.noise.state).any())

    def test_update_new_batch_size(self):
        """update(batch_size=N) resizes the state and keeps it finite."""
        new_B = 5
        self.noise.update(batch_size=new_B)
        self.assertEqual(self.noise.state.shape[0], new_B)
        self.assertFalse(torch.isnan(self.noise.state).any())
        self.assertFalse(torch.isinf(self.noise.state).any())

    def test_forward_after_update_finite(self):
        self.noise.update()
        out = self.noise.forward()
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_forward_always_zero(self):
        """DummyNoiseS2.forward() must always return an all-zero tensor."""
        out = self.noise.forward()
        self.assertTrue(torch.all(out == 0), "DummyNoiseS2 forward output should always be all zeros")

    def test_update_internal_state_flag(self):
        """forward(update_internal_state=True) exercises the flag path and still returns zeros."""
        out = self.noise.forward(update_internal_state=True)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertTrue(torch.all(out == 0), "DummyNoiseS2 must return zeros even with update_internal_state=True")

    def test_unknown_mode_raises(self):
        """Constructing with an unknown mode raises ValueError."""
        with self.assertRaises(ValueError):
            DummyNoiseS2(
                img_shape=IMG_SHAPE,
                batch_size=self.B,
                num_channels=self.C,
                num_time_steps=self.T,
                mode="invalid_mode",
            )


# ===========================================================================
# 2b. DummyNoiseS2 — constant_random mode
# ===========================================================================
@parameterized_class(("device",), _devices)
class TestDummyNoiseS2ConstantRandom(unittest.TestCase):
    """DummyNoiseS2(mode='constant_random'): fixed non-zero noise redrawn on each update()."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

        self.B = BATCH_SIZE
        self.T = NUM_TIME_STEPS
        self.C = NUM_CHANNELS
        self.noise = DummyNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=self.T,
            mode="constant_random",
            seed=333,
        ).to(self.device)

    def test_is_stateful(self):
        self.assertFalse(self.noise.is_stateful())

    def test_forward_shape(self):
        self.noise.update()
        out = self.noise.forward()
        self.assertEqual(tuple(out.shape), (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1]))

    def test_forward_finite(self):
        self.noise.update()
        out = self.noise.forward()
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_initial_state_is_zero_before_first_update(self):
        """State buffer is initialised to zeros before the first update()."""
        self.assertTrue(torch.all(self.noise.state == 0.0))

    def test_forward_nonzero_after_update(self):
        """After update(), constant_random output must be non-zero."""
        self.noise.update()
        out = self.noise.forward()
        self.assertFalse(torch.all(out == 0), "constant_random mode should produce a non-zero output after update()")

    def test_forward_stable_between_updates(self):
        """Two consecutive forward() calls without an intermediate update() return the same tensor."""
        self.noise.update()
        out1 = self.noise.forward().clone()
        out2 = self.noise.forward()
        self.assertTrue(torch.equal(out1, out2), "forward() should return the same fixed tensor between update() calls")

    def test_update_changes_state(self):
        """A second update() draws a new random state."""
        self.noise.update()
        state1 = self.noise.state.clone()
        self.noise.update()
        self.assertFalse(
            torch.equal(state1, self.noise.state), "A second update() should produce a different random state"
        )

    def test_reproducibility_same_seed(self):
        """Two instances with the same seed produce identical output after the first update()."""

        def _make(seed):
            return DummyNoiseS2(
                img_shape=IMG_SHAPE,
                batch_size=self.B,
                num_channels=self.C,
                num_time_steps=self.T,
                mode="constant_random",
                seed=seed,
            ).to(self.device)

        n1, n2 = _make(42), _make(42)
        n1.update()
        n2.update()
        self.assertTrue(torch.equal(n1.forward(), n2.forward()))

    def test_different_seeds_differ(self):
        """Two instances with different seeds produce different outputs."""

        def _make(seed):
            return DummyNoiseS2(
                img_shape=IMG_SHAPE,
                batch_size=self.B,
                num_channels=self.C,
                num_time_steps=self.T,
                mode="constant_random",
                seed=seed,
            ).to(self.device)

        n1, n2 = _make(1), _make(2)
        n1.update()
        n2.update()
        self.assertFalse(torch.equal(n1.forward(), n2.forward()))

    def test_update_internal_state_flag(self):
        """forward(update_internal_state=True) advances the state after returning the current one."""
        self.noise.update()
        state_before = self.noise.state.clone()
        self.noise.forward(update_internal_state=True)
        self.assertFalse(
            torch.equal(state_before, self.noise.state), "State should change after forward(update_internal_state=True)"
        )


# ===========================================================================
# 3. IsotropicGaussianRandomFieldS2
# ===========================================================================
@parameterized_class(("device",), _devices)
class TestIsotropicGRF(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

        self.B = BATCH_SIZE
        self.T = NUM_TIME_STEPS
        self.C = NUM_CHANNELS
        self.sigma = 2.0

    def _make_noise(self, seed=333, reflect=False, sigma=None, batch_size=None):
        if sigma is None:
            sigma = self.sigma
        if batch_size is None:
            batch_size = self.B
        return IsotropicGaussianRandomFieldS2(
            img_shape=IMG_SHAPE,
            batch_size=batch_size,
            num_channels=self.C,
            num_time_steps=self.T,
            sigma=sigma,
            seed=seed,
            reflect=reflect,
        ).to(self.device)

    def test_is_stateful(self):
        self.assertFalse(self._make_noise().is_stateful())

    def test_initial_state_shape(self):
        noise = self._make_noise()
        # state: (B, T, C, lmax_local, mmax_local, 2)
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)  # real + imag packed

    def test_state_shape_after_update(self):
        noise = self._make_noise()
        noise.update()
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)

    def test_forward_shape(self):
        noise = self._make_noise()
        noise.update()
        out = noise.forward()
        self.assertEqual(tuple(out.shape), (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1]))

    def test_forward_finite(self):
        noise = self._make_noise()
        noise.update()
        out = noise.forward()
        self.assertFalse(torch.isnan(out).any(), "NaN in forward output")
        self.assertFalse(torch.isinf(out).any(), "Inf in forward output")

    def test_forward_no_grad(self):
        """forward() inside torch.no_grad() does not produce grad-requiring tensors."""
        noise = self._make_noise()
        noise.update()
        with torch.no_grad():
            out = noise.forward()
        self.assertFalse(out.requires_grad)

    def test_reproducibility_same_seed(self):
        """Two instances with the same constructor seed produce identical output."""
        n1 = self._make_noise(seed=42)
        n2 = self._make_noise(seed=42)
        n1.update()
        n2.update()
        self.assertTrue(torch.equal(n1.forward(), n2.forward()))

    def test_different_seeds_differ(self):
        """Two instances with different seeds produce different outputs."""
        n1 = self._make_noise(seed=1)
        n2 = self._make_noise(seed=2)
        n1.update()
        n2.update()
        self.assertFalse(torch.equal(n1.forward(), n2.forward()))

    def test_rng_state_save_restore(self):
        """Saving and restoring the RNG state reproduces the same output."""
        noise = self._make_noise()
        cpu_state, gpu_state = noise.get_rng_state()
        noise.update()
        out1 = noise.forward().clone()
        # Restore and replay
        noise.set_rng_state(cpu_state, gpu_state)
        noise.update()
        out2 = noise.forward()
        self.assertTrue(torch.equal(out1, out2))

    def test_successive_updates_change_state(self):
        """Successive update() calls produce different states."""
        noise = self._make_noise()
        noise.update()
        state1 = noise.state.clone()
        noise.update()
        self.assertFalse(torch.equal(state1, noise.state))

    def test_reflect_negates_state(self):
        """reflect=True produces the negation of reflect=False (same seed)."""
        n_normal = self._make_noise(seed=42, reflect=False)
        n_reflect = self._make_noise(seed=42, reflect=True)
        n_normal.update()
        n_reflect.update()
        self.assertTrue(compare_tensors("reflect negates state", n_normal.state, -n_reflect.state))

    # -----------------------------------------------------------------------
    # Statistical tests — require a large batch for reliable estimates
    # -----------------------------------------------------------------------
    def test_spatial_variance_matches_sigma(self):
        """Empirical variance of the GRF output ≈ sigma² over a large batch.

        By construction, sum_l (2l+1)*sigma_l²/(4π) = sigma², so the expected
        spatial variance at any point equals sigma².
        """
        B_stat = 500
        sigma = 2.0
        noise = self._make_noise(seed=333, sigma=sigma, batch_size=B_stat)
        noise.update()
        out = noise.forward()  # (B_stat, T, C, H, W)

        empirical_var = out.var().item()
        self.assertAlmostEqual(
            empirical_var, sigma**2, delta=0.5, msg=f"Expected variance ≈ {sigma**2:.1f}, got {empirical_var:.4f}"
        )

    def test_zero_mean(self):
        """Empirical mean of the GRF output ≈ 0 over a large batch."""
        B_stat = 500
        noise = self._make_noise(seed=333, batch_size=B_stat)
        noise.update()
        out = noise.forward()  # (B_stat, T, C, H, W)

        empirical_mean = out.mean().item()
        self.assertAlmostEqual(empirical_mean, 0.0, delta=0.2, msg=f"Expected mean ≈ 0, got {empirical_mean:.4f}")

    # -----------------------------------------------------------------------
    # reset() — covers BaseNoiseS2 lines 92-99
    # -----------------------------------------------------------------------
    def test_reset_zeros_state(self):
        """reset() zeroes the spectral state regardless of prior content."""
        noise = self._make_noise()
        noise.update()  # fills state with non-zero values
        noise.reset()
        self.assertTrue(torch.all(noise.state == 0.0), "reset() should zero out the state")

    def test_reset_with_new_batch_size(self):
        """reset(batch_size=N) reallocates the state tensor to the new batch size."""
        noise = self._make_noise()
        new_B = 7
        noise.reset(batch_size=new_B)
        self.assertEqual(noise.state.shape[0], new_B)
        self.assertTrue(torch.all(noise.state == 0.0), "reset(batch_size=N) should produce an all-zero state")

    # -----------------------------------------------------------------------
    # learnable=True — covers noise.py line 210
    # -----------------------------------------------------------------------
    def test_learnable_sigma_l_is_parameter(self):
        """With learnable=True, sigma_l is registered as an nn.Parameter."""
        noise = IsotropicGaussianRandomFieldS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=self.T,
            sigma=self.sigma,
            seed=333,
            learnable=True,
        ).to(self.device)
        self.assertIsInstance(
            noise.sigma_l, torch.nn.Parameter, "sigma_l should be an nn.Parameter when learnable=True"
        )

    def test_learnable_gradient_flows_through_sigma_l(self):
        """Gradient of output sum flows back to sigma_l when learnable=True."""
        noise = IsotropicGaussianRandomFieldS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=self.T,
            sigma=self.sigma,
            seed=333,
            learnable=True,
        ).to(self.device)
        noise.update()
        out = noise.forward()  # sigma_l is multiplied in forward(), so grad flows
        out.sum().backward()
        self.assertIsNotNone(noise.sigma_l.grad, "sigma_l should have a gradient")
        self.assertFalse(torch.isnan(noise.sigma_l.grad).any(), "NaN in sigma_l.grad")
        self.assertFalse(torch.isinf(noise.sigma_l.grad).any(), "Inf in sigma_l.grad")

    # -----------------------------------------------------------------------
    # Spectral decay — covers the alpha path and is a meaningful sanity check
    # -----------------------------------------------------------------------
    def test_spectral_decay_alpha_smoother_than_white(self):
        """Higher alpha decays high angular modes and produces a spatially smoother output.

        Smoothness proxy: mean squared longitudinal finite differences.
        A smoother field (alpha=4) must have strictly smaller roughness than white noise (alpha=0).
        """
        B_stat = 100

        def _sample(alpha):
            n = IsotropicGaussianRandomFieldS2(
                img_shape=IMG_SHAPE,
                batch_size=B_stat,
                num_channels=1,
                num_time_steps=1,
                sigma=1.0,
                alpha=alpha,
                seed=333,
            ).to(self.device)
            n.update()
            return n.forward()  # (B_stat, 1, 1, H, W)

        out_white = _sample(alpha=0.0)
        out_smooth = _sample(alpha=4.0)

        roughness_white = torch.diff(out_white, dim=-1).pow(2).mean().item()
        roughness_smooth = torch.diff(out_smooth, dim=-1).pow(2).mean().item()

        self.assertLess(
            roughness_smooth,
            roughness_white,
            msg=f"alpha=4 roughness {roughness_smooth:.4f} should be < " f"alpha=0 roughness {roughness_white:.4f}",
        )


# ===========================================================================
# 4. DiffusionNoiseS2
# ===========================================================================
@parameterized_class(("device",), _devices)
class TestDiffusionNoiseS2(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

        self.B = BATCH_SIZE
        self.T = NUM_TIME_STEPS
        self.C = 2  # use 2 channels to exercise per-channel phi/sigma
        self.lambd = 0.5
        self.phi = math.exp(-self.lambd)

    def _make_noise(self, seed=333, reflect=False, lambd=None, batch_size=None, num_channels=None):
        if lambd is None:
            lambd = self.lambd
        if batch_size is None:
            batch_size = self.B
        if num_channels is None:
            num_channels = self.C
        return DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=self.T,
            lambd=lambd,
            seed=seed,
            reflect=reflect,
        ).to(self.device)

    def test_is_stateful(self):
        self.assertTrue(self._make_noise().is_stateful())

    def test_state_written_in_inference_mode_can_be_updated_after_it(self):
        # validation and inference run in inference mode, where the state becomes an inference tensor; the
        # next update outside of it, e.g. in training, replaces it rather than writing into it
        noise = self._make_noise()
        with torch.inference_mode():
            noise.update(replace_state=True, batch_size=self.B + 1)
            noise.update(replace_state=False)

        noise.update(replace_state=False)
        self.assertFalse(noise.state.is_inference())
        noise.reset()
        self.assertFalse(noise.state.is_inference())
        self.assertEqual(noise.state.shape[0], self.B + 1)
        self.assertTrue(torch.all(noise.state == 0))

    def test_initial_state_shape(self):
        noise = self._make_noise()
        # state: (B, T, C, lmax_local, mmax_local, 2)
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)

    def test_state_shape_after_replace(self):
        noise = self._make_noise()
        noise.update(replace_state=True)
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)

    def test_forward_shape(self):
        noise = self._make_noise()
        noise.update(replace_state=True)
        out = noise.forward()
        self.assertEqual(tuple(out.shape), (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1]))

    def test_forward_finite(self):
        noise = self._make_noise()
        noise.update(replace_state=True)
        out = noise.forward()
        self.assertFalse(torch.isnan(out).any(), "NaN in forward output")
        self.assertFalse(torch.isinf(out).any(), "Inf in forward output")

    def test_replace_state_non_trivial(self):
        """update(replace_state=True) samples from the stationary distribution
        and must produce a non-zero state."""
        noise = self._make_noise()
        noise.update(replace_state=True)
        self.assertFalse(torch.all(noise.state == 0), "State is all-zero after replace_state update")

    def test_advance_update_changes_state(self):
        """update(replace_state=False) advances the AR(1) state."""
        noise = self._make_noise()
        noise.update(replace_state=True)
        state_before = noise.state.clone()
        noise.update(replace_state=False)
        self.assertFalse(torch.equal(state_before, noise.state))

    def test_reproducibility_same_seed(self):
        """Two instances with the same seed produce identical stationary samples."""
        n1 = self._make_noise(seed=42)
        n2 = self._make_noise(seed=42)
        n1.update(replace_state=True)
        n2.update(replace_state=True)
        self.assertTrue(torch.equal(n1.state, n2.state))

    def test_different_seeds_differ(self):
        """Different seeds give different stationary samples."""
        n1 = self._make_noise(seed=1)
        n2 = self._make_noise(seed=2)
        n1.update(replace_state=True)
        n2.update(replace_state=True)
        self.assertFalse(torch.equal(n1.state, n2.state))

    def test_rng_state_save_restore(self):
        """Restoring RNG and tensor state reproduces an AR(1) advance exactly."""
        noise = self._make_noise()
        noise.update(replace_state=True)
        tensor_state = noise.get_tensor_state()
        cpu_state, gpu_state = noise.get_rng_state()

        # Advance once, record state
        noise.update(replace_state=False)
        out1 = noise.state.clone()

        # Restore and replay
        noise.set_tensor_state(tensor_state)
        noise.set_rng_state(cpu_state, gpu_state)
        noise.update(replace_state=False)
        self.assertTrue(torch.equal(out1, noise.state))

    def test_reflect_negates_state(self):
        """reflect=True flips the sign of the stationary state (same seed)."""
        n_normal = self._make_noise(seed=42, reflect=False)
        n_reflect = self._make_noise(seed=42, reflect=True)
        n_normal.update(replace_state=True)
        n_reflect.update(replace_state=True)
        self.assertTrue(compare_tensors("reflect negates stationary state", n_normal.state, -n_reflect.state))

    # -----------------------------------------------------------------------
    # Statistical test — requires a large batch for reliable estimates
    # -----------------------------------------------------------------------
    def test_temporal_correlation_matches_phi(self):
        """AR(1) correlation between consecutive spectral states ≈ phi = exp(-lambd).

        For each spectral coefficient x_{l,m} the AR(1) recurrence gives
        Corr(x_t, x_{t+1}) = phi, regardless of sigma_l.  Flattening over
        (batch, l, m, real/imag) yields many i.i.d. pairs, so the Pearson
        correlation of the flattened vectors should converge tightly to phi.
        """
        B_stat = 500
        lambd = 0.5
        phi_expected = math.exp(-lambd)

        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=B_stat,
            num_channels=1,
            num_time_steps=1,
            lambd=lambd,
            seed=333,
        ).to(self.device)

        # Sample from the stationary distribution
        noise.update(replace_state=True)
        state_t = noise.state.clone()  # (B, 1, 1, L, M, 2)

        # Advance one AR(1) step
        noise.update(replace_state=False)
        state_next = noise.state.clone()  # (B, 1, 1, L, M, 2)

        # Pearson correlation over all entries (correlation is phi for every one)
        x = state_t.reshape(-1).float()
        y = state_next.reshape(-1).float()
        corr = torch.corrcoef(torch.stack([x, y]))[0, 1].item()

        self.assertAlmostEqual(
            corr, phi_expected, delta=0.05, msg=f"Expected temporal correlation ≈ {phi_expected:.4f}, got {corr:.4f}"
        )

    # -----------------------------------------------------------------------
    # reset() — BaseNoiseS2 lines 92-99
    # -----------------------------------------------------------------------
    def test_reset_zeros_state(self):
        """reset() zeroes the spectral state regardless of prior content."""
        noise = self._make_noise()
        noise.update(replace_state=True)  # fills state with non-zero values
        noise.reset()
        self.assertTrue(torch.all(noise.state == 0.0), "reset() should zero out the state")

    def test_reset_with_new_batch_size(self):
        """reset(batch_size=N) reallocates the state tensor to the new batch size."""
        noise = self._make_noise()
        new_B = 7
        noise.reset(batch_size=new_B)
        self.assertEqual(noise.state.shape[0], new_B)
        self.assertTrue(torch.all(noise.state == 0.0), "reset(batch_size=N) should produce an all-zero state")

    # -----------------------------------------------------------------------
    # Pattern-4 state handling invariants
    # -----------------------------------------------------------------------
    def test_state_remains_registered_buffer_after_resize(self):
        """After a batch-changing update, `state` is still a registered buffer (not a plain
        attribute) and therefore still moves with `.to(device)`. Guards against a future
        regression to the raw `self.state = ...` rebind pattern.
        """
        noise = self._make_noise()
        new_B = self.B + 3
        noise.update(replace_state=True, batch_size=new_B)

        buffer_names = {name for name, _ in noise.named_buffers()}
        self.assertIn("state", buffer_names, "state must remain a registered buffer after resize")
        self.assertEqual(noise.state.shape[0], new_B)

        if torch.cuda.is_available():
            moved = noise.to(torch.device("cuda"))
            self.assertEqual(moved.state.device.type, "cuda", ".to(cuda) must move the state buffer after a resize")

    def test_ensure_state_is_idempotent_on_same_batch(self):
        """`_ensure_state` with the current batch size is a no-op: the underlying
        buffer is reused (same data_ptr), not re-registered.
        """
        noise = self._make_noise()
        before_ptr = noise.state.data_ptr()
        noise._ensure_state(noise.state.shape[0])
        after_ptr = noise.state.data_ptr()
        self.assertEqual(before_ptr, after_ptr, "_ensure_state on the same batch size must not rebind the buffer")

    def test_set_tensor_state_raises_on_shape_suffix_mismatch(self):
        """`set_tensor_state` must reject any non-batch shape mismatch up-front with a
        clear ValueError, and must NOT mutate `state` before raising.
        """
        noise = self._make_noise()
        original_shape = tuple(noise.state.shape)
        # perturb one of the suffix dims (channel count); batch dim may match
        bad_shape = list(noise.state.shape)
        bad_shape[2] = bad_shape[2] + 1  # wrong C
        bad = torch.zeros(tuple(bad_shape), dtype=noise.state.dtype, device=noise.state.device)
        with self.assertRaises(ValueError):
            noise.set_tensor_state(bad)
        self.assertEqual(
            tuple(noise.state.shape), original_shape, "state must not be mutated when set_tensor_state raises"
        )

    # -----------------------------------------------------------------------
    # Per-channel kT / lambd lists — noise.py lines 305-319
    # -----------------------------------------------------------------------
    def test_per_channel_kT_list(self):
        """kT given as a list of length num_channels constructs and runs without error."""
        kT_list = [0.001, 0.002]  # two distinct spatial correlation lengths
        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=2,
            num_time_steps=1,
            kT=kT_list,
            lambd=self.lambd,
            seed=333,
        ).to(self.device)
        noise.update(replace_state=True)
        out = noise.forward()
        self.assertEqual(tuple(out.shape), (self.B, 1, 2, IMG_SHAPE[0], IMG_SHAPE[1]))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_per_channel_lambd_list(self):
        """lambd given as a list of length num_channels constructs and runs without error."""
        lambd_list = [0.3, 0.7]  # two distinct temporal decay rates
        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=2,
            num_time_steps=1,
            lambd=lambd_list,
            seed=333,
        ).to(self.device)
        noise.update(replace_state=True)
        out = noise.forward()
        self.assertEqual(tuple(out.shape), (self.B, 1, 2, IMG_SHAPE[0], IMG_SHAPE[1]))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    # -----------------------------------------------------------------------
    # learnable=True — noise.py lines 345-354
    # -----------------------------------------------------------------------
    def test_learnable_constructs_phi_and_sigma_l_as_parameters(self):
        """learnable=True registers phi and sigma_l as nn.Parameters."""
        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=1,
            lambd=self.lambd,
            seed=333,
            learnable=True,
        ).to(self.device)
        self.assertIsInstance(noise.phi, torch.nn.Parameter, "phi should be an nn.Parameter when learnable=True")
        self.assertIsInstance(
            noise.sigma_l, torch.nn.Parameter, "sigma_l should be an nn.Parameter when learnable=True"
        )

    def _learnable_rollout(self, noise, n_steps=3, seed=333):
        """The noise of a rollout from a fresh state, as a rollout drives it."""
        noise.set_rng(seed)
        outs = []
        for k in range(n_steps):
            noise.update(replace_state=(k == 0))
            outs.append(noise())
        return torch.stack(outs)

    def _make_learnable(self):
        return DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=1,
            lambd=self.lambd,
            seed=333,
            learnable=True,
        ).to(self.device)

    def test_learnable_gradients_reach_phi_and_sigma_l(self):
        """phi and sigma_l enter the state only in update(), and get their gradients through it."""
        noise = self._make_learnable()
        weight = torch.randn_like(self._learnable_rollout(noise))
        (weight * self._learnable_rollout(noise)).sum().backward()

        for name in ("phi", "sigma_l"):
            with self.subTest(parameter=name):
                grad = getattr(noise, name).grad
                self.assertIsNotNone(grad, f"{name} should get a gradient")
                self.assertGreater(grad.abs().max().item(), 0.0)

    def test_learnable_gradients_match_finite_differences(self):
        """The gradient through the state recurrence, along a random direction, against central differences."""
        noise = self._make_learnable()
        weight = torch.randn_like(self._learnable_rollout(noise))

        def loss():
            return (weight * self._learnable_rollout(noise)).sum()

        loss().backward()
        eps = 1e-3
        for name in ("phi", "sigma_l"):
            with self.subTest(parameter=name):
                param = getattr(noise, name)
                direction = torch.randn_like(param)
                analytic = (param.grad * direction).sum().item()
                with torch.no_grad():
                    param.add_(eps * direction)
                    plus = loss().item()
                    param.sub_(2 * eps * direction)
                    minus = loss().item()
                    param.add_(eps * direction)
                numeric = (plus - minus) / (2 * eps)
                self.assertAlmostEqual(analytic, numeric, delta=1e-2 * max(1.0, abs(numeric)))

    def test_learnable_state_survives_reset_and_evaluation(self):
        """Resetting, or an update without gradients, does not corrupt the backward of a rollout already run."""
        noise = self._make_learnable()
        loss = self._learnable_rollout(noise).square().sum()

        noise.reset()
        with torch.no_grad():
            noise.update(replace_state=True)
            noise.update(replace_state=False)
        loss.backward()

        self.assertIsNotNone(noise.phi.grad)
        self.assertIsNone(noise.state.grad_fn, "the state should be detached after an update without gradients")

    def test_non_learnable_state_stays_a_plain_buffer(self):
        noise = self._make_noise()
        self._learnable_rollout(noise)
        self.assertIsNone(noise.state.grad_fn)
        self.assertIn("state", noise._buffers)

    # -----------------------------------------------------------------------
    # num_time_steps > 1 — noise.py lines 364-376 (init) and 405-417 (update)
    # -----------------------------------------------------------------------
    def test_num_time_steps_gt1_has_discount_buffer(self):
        """num_time_steps>1 creates a 'discount' buffer of shape (C, T, T)."""
        T = 3
        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=T,
            lambd=self.lambd,
            seed=333,
        ).to(self.device)
        self.assertTrue(hasattr(noise, "discount"), "discount buffer should exist when num_time_steps>1")
        self.assertEqual(tuple(noise.discount.shape), (self.C, T, T))

    def test_num_time_steps_gt1_state_shape(self):
        """With num_time_steps=3 the state time dimension is 3."""
        T = 3
        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=T,
            lambd=self.lambd,
            seed=333,
        ).to(self.device)
        self.assertEqual(noise.state.shape[1], T, f"Expected state.shape[1]=={T}, got {noise.state.shape[1]}")

    def test_num_time_steps_gt1_advance_rolls_history(self):
        """update(replace_state=False) shifts time history: old[1:] becomes new[:-1]."""
        T = 3
        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=T,
            lambd=self.lambd,
            seed=333,
        ).to(self.device)
        noise.update(replace_state=True)
        state_before = noise.state.clone()

        noise.update(replace_state=False)
        state_after = noise.state.clone()

        # After rolling, state_before[:, 1:, ...] must equal state_after[:, :-1, ...]
        self.assertTrue(
            torch.equal(state_after[:, :-1, ...], state_before[:, 1:, ...]),
            "Time history roll: state_after[:, :-1] should equal state_before[:, 1:]",
        )

    # -----------------------------------------------------------------------
    # update_internal_state=True — noise.py line 445
    # -----------------------------------------------------------------------
    def test_update_internal_state_advances_after_forward(self):
        """forward(update_internal_state=True) advances the AR(1) state after returning."""
        noise = self._make_noise()
        noise.update(replace_state=True)
        state_before = noise.state.clone()
        noise.forward(update_internal_state=True)
        self.assertFalse(
            torch.equal(state_before, noise.state),
            "State should have changed after forward(update_internal_state=True)",
        )


if __name__ == "__main__":
    unittest.main()
