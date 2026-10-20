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

import torch
import torch.nn as nn

# distributed computing stuff
from makani.utils import comm

from makani.models import Preprocessor2D
from makani.models.noise import IsotropicGaussianRandomFieldS2 as IGRF


def expand_time(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    r"""
    Replicate a tensor along a new sample dimension after the batch.

    Used to evaluate several interpolant time samples per batch element: the
    input is repeated so each sample can be paired with a different time.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(B, ...)``.
    n_samples : int
        Number of copies.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, n_samples, ...)``.
    """
    shape = x.shape
    x = x.unsqueeze(1).repeat(1, n_samples, *[1 for _ in shape[1:]])

    return x


def expand_batch(x: torch.Tensor, n_batch: int) -> torch.Tensor:
    r"""
    Replicate a tensor along a new leading batch dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input of any shape.
    n_batch : int
        Number of copies.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_batch, *x.shape)``.
    """
    shape = x.shape
    x = x.unsqueeze(0).repeat(n_batch, 1, *[1 for _ in shape[1:]])

    return x


def expand_batch_space(x: torch.Tensor, n_batch: int, n_lat: int, n_lon: int) -> torch.Tensor:
    r"""
    Broadcast a per-time, per-channel tensor over batch and space.

    Turns a table of scalars indexed by time step and channel into a full field,
    so it can be combined elementwise with gridded data.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(T, C)``.
    n_batch : int
        Batch size to expand to.
    n_lat : int
        Number of latitudes.
    n_lon : int
        Number of longitudes.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_batch, T, C, n_lat, n_lon)``.
    """
    T, C = x.shape
    x = x.reshape(1, T, C, 1, 1).repeat(n_batch, 1, 1, n_lat, n_lon)

    return x


class InterpolationWrapper(nn.Module):
    """
    thin wrapper to make usage of the usual deterministic models feasible.
    """

    def __init__(self, model_handle, n_pred_chans, n_static_channels, n_dynamic_channels):
        super().__init__()

        inp_chans = n_pred_chans * 2 + n_static_channels + n_dynamic_channels + 1
        self.model = model_handle(inp_chans=inp_chans)

    def forward(self, x0, x, xu, static, s):
        """
        expects inputs in the shape
        """
        # S has shape [B*S]
        s = s.reshape(s.shape[0], 1, 1, 1)
        s = torch.tile(s, (1, 1, x.shape[-2], x.shape[-1]))

        # concatenate the inputs
        inputlist = [x0, x]
        if xu is not None:
            inputlist.append(xu)
        if static is not None:
            inputlist.append(static)
        inputlist.append(s)
        x_in = torch.cat(inputlist, dim=-3)

        # fwd pass
        x_out = self.model(x_in)

        return x_out


class TimeAwareInterpolationWrapper(nn.Module):
    """
    thin wrapper to make usage of the usual deterministic models feasible.
    """

    def __init__(self, model_handle, n_pred_chans, n_static_channels, n_dynamic_channels):
        super().__init__()

        inp_chans = n_pred_chans * 2 + n_static_channels + n_dynamic_channels
        self.model = model_handle(inp_chans=inp_chans)

    def forward(self, x0, x, xu, static, s):
        """
        expects inputs in the shape
        """

        # concatenate the inputs
        inputlist = [x0, x]
        if xu is not None:
            inputlist.append(xu)
        if static is not None:
            inputlist.append(static)
        x_in = torch.cat(inputlist, dim=-3)

        # fwd pass
        x_out = self.model(x_in, s.unsqueeze(1))

        return x_out


# TODO: update noise state
class StochasticInterpolantWrapper(nn.Module):
    r"""
    Wrapper module to implement tthe stochastic interpolation framework for forecasting described in [1].
    During training, the module

    [1] Chen et al.; Probabilistic forecasting with stochastic interpolants and Follmer Processes
    """

    def __init__(
        self, params, model_handle, noise_epsilon=1.0, use_foellmer=False, antithetic_sampling=False, seed=333, **kwargs
    ):
        super().__init__()

        if params.n_history != 0:
            raise ValueError(
                f"this model currently does not support history, expected n_history == 0 but got {params.n_history}"
            )
        if params.n_future != 0:
            raise ValueError(
                f"this model currently does not support future steps, expected n_future == 0 but got {params.n_future}"
            )

        # get the preprocessor
        self.preprocessor = Preprocessor2D(params)

        # compute the new number of input/output channels
        n_pred_chans = params.N_in_predicted_channels
        n_static_channels = params.N_static_channels
        n_dynamic_channels = params.N_dynamic_channels

        if n_pred_chans != params.N_out_channels:
            raise ValueError(
                f"expected N_in_predicted_channels ({n_pred_chans}) to match N_out_channels ({params.N_out_channels})"
            )

        # TODO: check if the model is actually deterministic first before wrapping it
        if params.get("time_aware", False):
            self.model = TimeAwareInterpolationWrapper(
                model_handle, n_pred_chans, n_static_channels, n_dynamic_channels
            )
        else:
            self.model = InterpolationWrapper(
                model_handle,
                n_pred_chans,
                n_static_channels=params.N_static_channels,
                n_dynamic_channels=params.N_dynamic_channels,
            )

        seed_off = (
            comm.get_rank("model")
            + comm.get_size("model") * comm.get_rank("batch")
            + comm.get_size("model") * comm.get_size("batch") * comm.get_rank("ensemble")
        )
        self.noise_module = IGRF(
            (params.img_crop_shape_x, params.img_crop_shape_y),
            params.batch_size,
            params.N_in_predicted_channels,
            num_time_steps=1,
            sigma=1.0,
            alpha=0.0,
            grid_type=params.data_grid_type,
            seed=seed + seed_off,
        )

        self.use_foellmer = use_foellmer
        self.antithetic_sampling = antithetic_sampling

        # set rng:
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed + 333 + seed_off)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed + 333 + seed_off)

        self.noise_epsilon = noise_epsilon

        # we take the particular choice made in the paper
        self.alpha_fn = lambda s: 1.0 - s
        self.dalpha_fn = lambda s: -1.0
        self.beta_fn = lambda s: torch.square(s)
        self.dbeta_fn = lambda s: 2.0 * s
        self.sigma_fn = lambda s: self.noise_epsilon * (1.0 - s)
        self.dsigma_fn = lambda s: -1.0 * self.noise_epsilon
        self.gamma_fn = lambda s: torch.sqrt(s) * self.sigma_fn(s)
        # note that in the original paper, the sqrt(s) term was not taken a derivative of
        self.dgamma_fn = lambda s: torch.sqrt(s) * self.dsigma_fn(s)

    def get_internal_rng(self, gpu=True):
        r"""
        Return the noise module's private generator.

        Parameters
        ----------
        gpu : bool, optional
            Return the CUDA generator if ``True`` (the default), otherwise the
            CPU generator.

        Returns
        -------
        torch.Generator
            The generator driving the interpolant's noise.
        """
        if gpu:
            return self.noise_module.rng_gpu
        else:
            return self.noise_module.rng_cpu

    def gsq_fn(self, s: torch.Tensor, foellmer=False) -> torch.Tensor:
        r"""
        Squared diffusion coefficient of the sampling SDE.

        The interpolant admits a family of samplers that share the same marginal
        distributions but differ in how much noise they inject; ``gsq_fn``
        selects one. The default is :math:`g^2(s) = \sigma^2(s)`. The Föllmer
        choice instead returns :math:`q^2(s)`, the diffusion of the Föllmer
        process, which drives the sample toward the target with a different
        noise schedule.

        Parameters
        ----------
        s : torch.Tensor
            Interpolant time in :math:`[0, 1]`.
        foellmer : bool, optional
            Use the Föllmer coefficient rather than :math:`\sigma^2`, by
            default ``False``.

        Returns
        -------
        torch.Tensor
            Squared diffusion coefficient, same shape as ``s``.
        """
        if foellmer:
            term1 = (
                2.0 * torch.square(self.sigma_fn(s)) * torch.where(s > 0, s * self.dbeta_fn(s) / self.beta_fn(s), 2.0)
            )
            term2 = 2.0 * s * self.sigma_fn(s) * self.dsigma_fn(s)
            result = torch.abs(term1 - term2 - torch.square(self.sigma_fn(s)))
        else:
            result = torch.square(self.sigma_fn(s))

        return result

    def dlog_rho(self, x: torch.Tensor, x0: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        r"""
        Score function :math:`\nabla_x \log \rho_s(x)` of the interpolant density.

        Recovered algebraically from the learned drift ``b`` rather than
        estimated separately: for a Gaussian stochastic interpolant the drift
        and the score are related by a closed-form expression in the schedule
        functions, so one network suffices for both. The score is what turns the
        probability-flow ODE into an SDE sampler.

        Parameters
        ----------
        x : torch.Tensor
            Current state along the interpolant path.
        x0 : torch.Tensor
            Base sample the path starts from.
        b : torch.Tensor
            Drift predicted by the model at ``(x, s)``.
        s : torch.Tensor
            Interpolant time, strictly inside :math:`(0, 1)`. The expression is
            singular at both endpoints, which the samplers avoid by never
            evaluating there.

        Returns
        -------
        torch.Tensor
            The score, same shape as ``x``.
        """
        # we never call this at s=0 or s=1, so everything is fine
        As = 1.0 / (s * self.sigma_fn(s) * (self.dbeta_fn(s) * self.sigma_fn(s) - self.beta_fn(s) * self.dsigma_fn(s)))
        cs = x * self.dbeta_fn(s) + (self.beta_fn(s) * self.dalpha_fn(s) - self.dbeta_fn(s) * self.alpha_fn(s)) * x0

        return As * (self.beta_fn(s) * b - cs)

    @torch.compile
    def _interpolant(self, x0: torch.Tensor, x1: torch.Tensor, noise: torch.Tensor, s: torch.Tensor):
        # s has shape [B, S], expand:
        ishape = x0.shape
        sr = s.reshape([s.shape[0], s.shape[1]] + [1 for _ in ishape[2:]])
        return self.alpha_fn(sr) * x0 + self.beta_fn(sr) * x1 + self.gamma_fn(sr) * noise

    @torch.compile
    def _drift(self, x0: torch.Tensor, x1: torch.Tensor, noise: torch.Tensor, s: torch.Tensor):
        # s has shape [B, S], expand:
        ishape = x0.shape
        sr = s.reshape([s.shape[0], s.shape[1]] + [1 for _ in ishape[2:]])
        return self.dalpha_fn(sr) * x0 + self.dbeta_fn(sr) * x1 + self.dgamma_fn(sr) * noise

    def stochastic_path(
        self, x0: torch.Tensor, x1: torch.Tensor, noise: torch.Tensor, s: torch.Tensor, return_derivative=True
    ):
        """returns I and R"""

        # get interpolant
        interp = self._interpolant(x0, x1, noise, s)

        if return_derivative:
            drift = self._drift(x0, x1, noise, s)
            return interp, drift
        else:
            return interp

    # @torch.compile
    def _compute_bhat(
        self, x: torch.Tensor, x0: torch.Tensor, xu: torch.Tensor, static: torch.Tensor, s: torch.Tensor, foellmer=False
    ) -> torch.Tensor:
        b = self.model(x0, x, xu, static, s)
        if foellmer:
            ishape = x0.shape
            sr = s.reshape([s.shape[0]] + [1 for _ in ishape[1:]])
            correction = (
                0.5
                * (self.gsq_fn(sr, foellmer=self.use_foellmer) - torch.square(self.sigma_fn(sr)))
                * self.dlog_rho(x, x0, b, sr)
            )
            b = b + correction

        return b

    def _stack_dynamic_features(self, dynamic_features):
        # the dynamic features of the input step, stacked to [B, C, H, W], or None without any
        dynamic_features = {} if dynamic_features is None else dynamic_features
        expected = set(self.preprocessor.dynamic_features)
        if set(dynamic_features) != expected:
            raise ValueError(f"the model takes the dynamic features {sorted(expected)}, got {sorted(dynamic_features)}")

        inpu = self.preprocessor.stack_dynamic_features(dynamic_features)
        if inpu is not None:
            # remove the time dim, there is no history
            inpu = inpu.squeeze(1)

        return inpu

    def _forward_train(self, inp, tar, inpu, n_samples=1):

        # get input shape [B, C, H, W]
        ishape = list(inp.shape)
        samples_fact = 2 if self.antithetic_sampling else 1

        # expand and flatten the dynamic and static features
        with torch.no_grad():
            if inpu is not None:
                # convert to shape [B*S, C, H, W]
                inpu = expand_time(inpu, n_samples * samples_fact).flatten(0, 1)

            # get static
            static = self.preprocessor.get_static_features()

            if static is not None:
                static = torch.tile(static, (ishape[0], 1, 1, 1))
                # convert to shape [B*S, C, H, W]
                static = expand_time(static, n_samples * samples_fact).flatten(0, 1)

            # make sure the shape is such that we can easily broadcast ([B, S])
            stens = torch.empty([ishape[0], n_samples], dtype=inp.dtype, device=inp.device)
            if inp.is_cuda:
                stens.uniform_(0.0, 1.0, generator=self.rng_gpu)
            else:
                stens.uniform_(0.0, 1.0, generator=self.rng_cpu)

            if self.antithetic_sampling:
                stens = torch.cat([stens, 1.0 - stens], dim=1)

            # noise already has the right dims [B, 1, C, H, W]
            noise = self.noise_module(update_internal_state=True)

            # unsqueeze input and target to [B, S, C, H, W]
            inp = expand_time(inp, n_samples * samples_fact)
            tar = expand_time(tar, n_samples * samples_fact)

            # stochstic path, separately for variants with features and ones without
            inter, drift = self.stochastic_path(inp, tar, noise, stens, return_derivative=True)

            # flatten the input ttensors to be 4d
            inp = inp.flatten(0, 1)
            noise = noise.flatten(0, 1)
            stens = stens.flatten(0, 1)
            inter = inter.flatten(0, 1)
            drift = drift.flatten(0, 1)

        # forward pass uses 4d tensors
        drift_pred = self.model(inp, inter, inpu, static, stens)

        return drift_pred, drift

    def _forward_eval(self, inp, inpu, n_steps=1):

        torch._check(len(inp.shape) == 4, lambda: f"expected a 4D input, got {len(inp.shape)}D")

        with torch.no_grad():
            # get static in shape [B, C, H, W]
            static = self.preprocessor.get_static_features()

            # get sampling points: shape [S]
            stens = torch.linspace(0, 1, n_steps + 1, dtype=inp.dtype, device=inp.device)

            # compute delta s, shape: [S]
            delta_s = stens[1:] - stens[:-1]

            # tile and reshape stens [B, S+1]
            stens = expand_batch(stens, inp.shape[0])

        # do the sampling
        # initialization
        s_0 = stens[:, 0]
        deltas_0 = delta_s[0]

        # deterministic term: the model output should be squeezed
        x = inp + self.model(inp, inp, inpu, static, s_0) * delta_s[0]

        # noise term needs to be adjusted to have dimension [B, C, H, W]
        with torch.no_grad():
            noise = self.noise_module(update_internal_state=True).squeeze(1)

        x = x + self.sigma_fn(s_0.reshape(-1, 1, 1, 1)) * torch.sqrt(deltas_0) * noise

        # trajectory: upper bound is n_steps-1, like in the paper
        for i in range(1, n_steps):

            # get s and deltas
            s_i = stens[:, i]
            deltas_i = delta_s[i]

            # pass to bhat computation, get everything into shape: [B, C, H, W]
            bhat = self._compute_bhat(x, inp, inpu, static, s_i, foellmer=self.use_foellmer)

            # deterministic part
            x = x + bhat * deltas_i

            # noise term
            with torch.no_grad():
                noise = self.noise_module(update_internal_state=True).squeeze(1)

            x = x + torch.sqrt(self.gsq_fn(s_i.reshape(-1, 1, 1, 1), foellmer=self.use_foellmer) * deltas_i) * noise

        return x

    def forward(self, inp, tar=None, dynamic_features=None, n_samples=1, n_steps=1):
        r"""
        Train on interpolant samples, or generate a forecast by integrating the SDE.

        The two modes are genuinely different computations, selected by the
        module's training mode. In training, interpolant times are sampled and
        the model learns to predict the drift at those times, so ``tar`` is
        required and ``n_samples`` controls how many times are drawn per batch
        element. In evaluation, no target exists: the sampler integrates the
        interpolant from the base distribution to the target distribution in
        ``n_steps`` steps, and ``n_samples`` is unused.

        The noise of the sampler is resized to the current input batch and
        redrawn from its stationary distribution on entry, so callers that fold
        an ensemble dimension into the batch can pass a ``(B * E, ...)`` input
        without managing noise state themselves.

        Parameters
        ----------
        inp : torch.Tensor
            Input of shape ``(B, inp_chans, nlat, nlon)``.
        tar : torch.Tensor, optional
            Target of shape ``(B, out_chans, nlat, nlon)``. Required in
            training mode, ignored in evaluation mode.
        dynamic_features : dict, optional
            The dynamic features of the input, name to ``(B, 1, C_f, nlat, nlon)``,
            as in the batch. Required if the model takes dynamic features.
        n_samples : int, optional
            Number of interpolant time samples per batch element during
            training, by default ``1``.
        n_steps : int, optional
            Number of integration steps during evaluation, by default ``1``.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            In training mode, a ``(drift_pred, drift)`` pair -- the model's
            predicted drift and the analytic drift of the interpolant path --
            for the loss to compare. In evaluation mode, the generated forecast
            of shape ``(B, out_chans, nlat, nlon)``.

        Raises
        ------
        ValueError
            If the dynamic features do not match the ones the model takes.
        """
        # Keep the noise sized to the current input batch so callers that fold an
        # ensemble dim into the batch (e.g. stochastic_trainer.validate_one_epoch)
        # can hand us a (B*E, ...) input without any state-resize plumbing of their own.
        # Drawing a fresh stationary state here also means the very first noise read
        # inside _forward_train / _forward_eval sees a non-trivial sample.
        self.noise_module.update(replace_state=True, batch_size=inp.shape[0])
        inpu = self._stack_dynamic_features(dynamic_features)
        if self.training:
            return self._forward_train(inp, tar, inpu, n_samples=n_samples)
        else:
            return self._forward_eval(inp, inpu, n_steps=n_steps)
