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

from typing import Union, Tuple


import torch
import torch.nn as nn

from makani.utils import comm
from makani.utils.grids import GridQuadrature, grid_to_quadrature_rule
from makani.mpu.mappings import copy_to_parallel_region

from makani.models.preprocessor_helpers import get_bias_correction, get_static_features
from makani.utils.dataloaders.batch import INPUT_DYNAMIC_FEATURES, TARGET_DYNAMIC_FEATURES, ZENITH


@torch.compiler.disable
def _run_eager(fn, *args, **kwargs):
    """Run a callable outside torch.compile (forces a graph break).

    The noise field is complex-valued (torch.complex + inverse SHT) and inductor's Triton
    backend cannot codegen complex dtypes (KeyError: 'complex64'). A method-level
    @torch.compiler.disable on the noise module's forward is NOT honored when dynamo inlines
    the nn.Module call (it traces straight into it), whereas a disable on a plain function call
    like this one is respected — so we break at the call site instead.
    """
    return fn(*args, **kwargs)


class Preprocessor2D(nn.Module):
    r"""
    Input/output plumbing that surrounds the model in a rollout.

    Sits between the dataloader and the network and owns everything that is not
    the model itself: history handling, dynamic and static features, noise
    injection, bias correction, and the normalization of the history window.

    Everything here works on tensors with an explicit time axis,
    ``(B, T, C, H, W)``. The time steps are folded into the channels in exactly
    one place, :meth:`pack_model_input`, right before the network, which still
    takes a single flat channel stack. Folding and unfolding in between is what
    the history handling used to get wrong.

    The main responsibilities are:

    * **History.** Inputs carry ``n_history + 1`` time steps;
      :meth:`append_history` slides the window forward by one step during an
      autoregressive rollout.
    * **Dynamic features.** Quantities the model consumes but does not
      predict, such as the solar zenith angle. Which ones the model takes is
      fixed by the configuration (:attr:`dynamic_features`); their values come
      with every batch (:meth:`cache_dynamic_features`) and are appended to the
      input (:meth:`append_dynamic_features`). They are carried over from the
      target to the input at each rollout step, since the model cannot generate
      them itself.
    * **Static features.** Time-invariant fields such as orography and the
      land-sea mask, appended to every packed input.
    * **Noise.** Optional stochastic perturbation of the input, either
      concatenated as extra channels or added to selected channels, for
      ensemble forecasting.
    * **Bias correction.** An optional additive correction applied to the
      model input.

    Parameters
    ----------
    params : ParamsBase
        Configuration object. Relevant keys include the image shape and
        resampled shape, ``n_history``, ``history_normalization_mode``,
        ``input_noise``, and the static-feature and bias-correction settings.

    See Also
    --------
    get_preprocessor : factory that constructs this from a config.
    """

    def __init__(self, params):
        super().__init__()

        # image shape — must be set first; used by quadrature and noise constructors below
        self.img_shape = [params.img_shape_x, params.img_shape_y]
        self.img_shape_resampled = [params.img_shape_x_resampled, params.img_shape_y_resampled]

        self.subsampling_factor = params.get("subsampling_factor", 1)
        self.n_history = params.n_history
        self.history_normalization_mode = params.history_normalization_mode
        if self.history_normalization_mode == "exponential":
            self.history_normalization_decay = params.history_normalization_decay
            # inverse ordering, since first element is oldest
            history_normalization_weights = torch.exp(
                (-self.history_normalization_decay)
                * torch.arange(start=self.n_history, end=-1, step=-1, dtype=torch.float32)
            )
            history_normalization_weights = history_normalization_weights / torch.sum(history_normalization_weights)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        elif self.history_normalization_mode == "mean":
            history_normalization_weights = torch.as_tensor(1.0 / float(self.n_history + 1), dtype=torch.float32)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        else:
            history_normalization_weights = torch.ones(self.n_history + 1, dtype=torch.float32)
        self.register_buffer("history_normalization_weights", history_normalization_weights, persistent=False)
        if self.history_normalization_mode != "none":
            self.quadrature = GridQuadrature(
                grid_to_quadrature_rule(params.model_grid_type),
                img_shape=self.img_shape_resampled,
                crop_shape=None,
                crop_offset=(0, 0),
                normalize=True,
                distributed=True,
            )

        self.history_mean = None
        self.history_std = None
        self.history_diff_mean = None
        self.history_diff_var = None
        self.history_eps = 1e-6

        # the dynamic features the model takes, in the order their channels are appended
        self.dynamic_features = [ZENITH] if params.get("add_zenith", False) else []

        # their values for the current batch, per mode, so that evaluation does not clobber
        # an in-flight training step: {mode: {"input" | "target": {name: (B, T, C_f, H, W)}}}
        self._dynamic_cache = {mode: {"input": {}, "target": {}} for mode in ("train", "eval")}

        # get bias correction
        bias = get_bias_correction(params)

        if bias is not None:
            # register static buffer
            self.register_buffer("bias_correction", bias, persistent=False)

        # process static features
        static_features = get_static_features(params)
        self.do_add_static_features = False
        if static_features is not None:

            # remember that we need static features
            self.do_add_static_features = True

            # register static buffer
            self.register_buffer("static_features", static_features, persistent=False)

        if params.get("input_noise", None) is not None:
            noise_params = params.input_noise
            centered_noise = noise_params.get("centered", False)

            # noise seed: important, this will be passed down as-is
            if not centered_noise:
                self.noise_base_seed = 333 + comm.get_rank("model") + comm.get_size("model") * comm.get_rank("data")
                reflect = False
            else:
                # here, ranks (0,1), (2,3), ... should map to the same eff rank, since they only differ by reflection but should otherwise get the
                # same seed
                ensemble_eff_rank = comm.get_rank("ensemble") // 2
                reflect = comm.get_rank("ensemble") % 2 == 0
                self.noise_base_seed = (
                    333
                    + comm.get_rank("model")
                    + comm.get_size("model") * ensemble_eff_rank
                    + comm.get_size("model") * comm.get_size("ensemble") * comm.get_rank("batch")
                )

            if "type" not in noise_params:
                raise ValueError("Error, please specify an input noise type")

            self.input_noise_mode = noise_params.get("mode", "concatenate")

            if self.input_noise_mode == "concatenate":
                noise_channels = noise_params.get("n_channels", 1)
            elif self.input_noise_mode == "perturb":
                self.perturb_channels = noise_params.get("perturb_channels", params.channel_names)
                self.perturb_channels = [params.channel_names.index(ch) for ch in self.perturb_channels]
                noise_channels = len(self.perturb_channels)
            else:
                raise NotImplementedError(f"Error, input noise mode {self.input_noise_mode} not supported.")

            noise_lmax = noise_params.get("lmax", None)

            if noise_params["type"] == "diffusion":
                from makani.models.noise import DiffusionNoiseS2

                # set the spatio-temporal correlation length
                kT = noise_params.get("kT", 0.5 * (100 / 6370) ** 2)
                lambd = noise_params.get("lambd", params.dt * params.dhours / 6.0)

                self.input_noise = DiffusionNoiseS2(
                    img_shape=self.img_shape_resampled,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                    sigma=noise_params.get("sigma", 1.0),
                    kT=kT,  # use various scales
                    lambd=lambd,  # use suggestion here: tau=6h
                    grid_type=params.model_grid_type,
                    lmax=noise_lmax,
                    seed=self.noise_base_seed,
                    reflect=reflect,
                    learnable=noise_params.get("learnable", False),
                )
            elif noise_params["type"] == "white":
                from makani.models.noise import IsotropicGaussianRandomFieldS2

                self.input_noise = IsotropicGaussianRandomFieldS2(
                    img_shape=self.img_shape_resampled,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                    sigma=noise_params.get("sigma", 1.0),
                    alpha=noise_params.get("alpha", 0.0),
                    grid_type=params.model_grid_type,
                    lmax=noise_lmax,
                    seed=self.noise_base_seed,
                    reflect=reflect,
                    learnable=noise_params.get("learnable", False),
                )
            elif noise_params["type"] == "dummy":
                from makani.models.noise import DummyNoiseS2

                self.input_noise = DummyNoiseS2(
                    img_shape=self.img_shape_resampled,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                )
            else:
                raise NotImplementedError(f'Error, input noise type {noise_params["type"]} not supported.')

    def flatten_history(self, x):
        r"""
        Fold the time dimension into the channel dimension.

        Only for the boundary to code that still takes a flat channel stack, the
        network and the losses; everything in between keeps the time axis.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T, C, H, W)``. Tensors that are already 4D
            are returned unchanged, so this is safe to call unconditionally.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, T * C, H, W)``.
        """
        # flatten input
        if x.dim() == 5:
            b_, t_, c_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, t_ * c_, h_, w_))

        return x

    def expand_history(self, x, nhist):
        r"""
        Split a flattened channel stack back into time and channel dimensions.

        Inverse of :meth:`flatten_history`.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T * C, H, W)``. Tensors that are already 5D
            are returned unchanged.
        nhist : int
            Number of time steps ``T`` to split off. The channel dimension must
            be divisible by this; the check is a ``torch._check`` so it survives
            as a runtime assertion under ``torch.compile`` rather than becoming
            a graph-breaking data-dependent branch.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, nhist, C, H, W)``.
        """
        if x.dim() == 4:
            b_, ct_, h_, w_ = x.shape
            # torch._check (rather than `if ...: raise`) so this stays a runtime
            # assertion under torch.compile instead of becoming a data-dependent
            # branch that breaks the graph.
            torch._check(
                ct_ % nhist == 0,
                lambda: (
                    f"expand_history: channel dim {ct_} is not divisible by nhist={nhist}. "
                    f"The flattened-history input may not match the preprocessor's expected "
                    f"n_history={self.n_history} (so ct_ should be a multiple of n_history+1={nhist})."
                ),
            )
            x = torch.reshape(x, (b_, nhist, ct_ // nhist, h_, w_))
        return x

    def _check_time_axis(self, x, caller):
        torch._check(
            x.dim() == 5,
            lambda: f"{caller}: expected a tensor with an explicit time axis, (B, T, C, H, W), got shape {tuple(x.shape)}",
        )

    def pack_model_input(self, x):
        r"""
        Pack the input into the single channel stack the network takes.

        The one place where the time axis is folded into the channels: the time
        steps are stacked oldest first, and the static features, which have no
        time axis, are appended after them.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, C, H, W)``, with the dynamic features
            already appended.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, T * C + n_static, H, W)``.
        """
        self._check_time_axis(x, "pack_model_input")

        x = self.flatten_history(x)

        if self.do_add_static_features:
            # the same fields for every sample
            static = torch.tile(self.static_features, dims=(x.shape[0], 1, 1, 1))
            x = torch.cat([x, static], dim=1)

        return x

    def append_history(self, x, pred, step, update_state=True):
        r"""
        Slide the history window forward by one autoregressive step.

        Drops the oldest time step from ``x`` and appends the new prediction
        ``pred``, producing the input for the next step of the rollout.

        This also advances the cached dynamic features: quantities such as the
        solar zenith angle are known in advance but not predicted by the model,
        so the target's value for this step has to move into the input window,
        or the next step would be conditioned on stale values. Whether the train
        or eval cache is updated follows the module's training mode.

        Parameters
        ----------
        x : torch.Tensor
            Current input window, shape ``(B, n_history + 1, C, H, W)``.
        pred : torch.Tensor
            Newly predicted step, shape ``(B, 1, C, H, W)``.
        step : int
            Index of the current rollout step, which selects the target step of
            the dynamic features. Steps beyond the cached range leave them
            untouched.
        update_state : bool, optional
            If ``False``, only shift the history and leave the dynamic features
            as they are. By default ``True``.

        Returns
        -------
        torch.Tensor
            The next input window, same shape as ``x``.
        """
        self._check_time_axis(x, "append_history")
        self._check_time_axis(pred, "append_history")

        if update_state:
            cache = self._dynamic_cache[self._mode()]
            for name in self.dynamic_features:
                target = cache["target"].get(name, None)
                if (target is None) or (step >= target.shape[1]):
                    continue
                window = cache["input"][name]
                window.copy_(torch.cat([window[:, 1:], target[:, step : step + 1]], dim=1))

        return torch.cat([x[:, 1:], pred], dim=1)

    def append_dynamic_features(self, x):
        r"""
        Append the dynamic features of the current batch, and the input noise.

        The channels are appended in a fixed order along the channel axis:
        ``[x | dynamic features, in the order of dynamic_features | noise]``.
        In ``perturb`` mode the noise is added to the selected channels instead
        of appended. Which cache is read follows the module's training mode.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, n_history + 1, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Input with the appended channels, ``(B, n_history + 1, C', H, W)``.

        Raises
        ------
        RuntimeError
            If a dynamic feature the model takes was not cached for this batch.
        """
        self._check_time_axis(x, "append_dynamic_features")

        cache = self._dynamic_cache[self._mode()]["input"]
        appended = []
        for name in self.dynamic_features:
            if name not in cache:
                raise RuntimeError(
                    f"append_dynamic_features: the model takes the dynamic feature '{name}', but none is "
                    f"cached for this batch. Call cache_dynamic_features with the batch first."
                )
            feature = cache[name]
            # if these diverge, the cat below would fail with a cryptic message. torch._check
            # keeps this a runtime assertion under torch.compile instead of a graph break.
            torch._check(
                (feature.shape[0] == x.shape[0]) and (feature.shape[1] == x.shape[1]),
                lambda: (
                    f"append_dynamic_features: '{name}' has batch and time {tuple(feature.shape[:2])}, "
                    f"the input {tuple(x.shape[:2])}. Was it cached at a different batch size?"
                ),
            )
            appended.append(feature)

        if hasattr(self, "input_noise"):
            # run the noise module eagerly: it is complex-valued (SHT) and cannot be inductor-
            # compiled; the method-level disable on its forward is not honored through the
            # nn.Module call, so break at the call site. See _run_eager.
            noise = _run_eager(self.input_noise)
            torch._check(
                noise.shape[0] == x.shape[0],
                lambda: (
                    f"append_dynamic_features: batch mismatch between input_noise state "
                    f"({noise.shape[0]}) and input ({x.shape[0]}). "
                    f"Did you call update_internal_state(batch_size=...) at a different "
                    f"batch than the current forward pass?"
                ),
            )
            if self.input_noise_mode == "concatenate":
                appended.append(noise)
            elif self.input_noise_mode == "perturb":
                # fully out-of-place: build a zero noise field and add to all channels
                noise_full = torch.zeros_like(x)
                noise_full[:, :, self.perturb_channels] = noise
                x = x + noise_full

        if not appended:
            return x

        return torch.cat([x] + appended, dim=2)

    def history_compute_stats(self, x):
        r"""
        Compute and cache the normalization statistics for a history window.

        Must be called before :meth:`history_normalize` or
        :meth:`history_denormalize`, and re-called whenever the batch size
        changes. Statistics are area-weighted using the grid quadrature, so
        they are true spherical means rather than uniform grid averages, and
        are broadcast across the spatial model-parallel group so every rank
        normalizes identically.

        The behavior depends on ``history_normalization_mode``:

        * ``"none"`` -- statistics are set to zero mean and unit standard
          deviation, making normalization a no-op.
        * ``"timediff"`` -- statistics of the *differences* between consecutive
          time steps are cached instead, characterizing the tendency rather than
          the state.
        * anything else (e.g. ``"exponential"``) -- a weighted mean and standard
          deviation over the history window, using
          ``history_normalization_weights``.

        Parameters
        ----------
        x : torch.Tensor
            History window of shape ``(B, n_history + 1, C, H, W)``.

        Returns
        -------
        None
            Results are stored on the module as ``history_mean`` /
            ``history_std``, of shape ``(B, 1, C, 1, 1)`` so that they broadcast
            over the time and grid dimensions (or ``history_diff_mean`` /
            ``history_diff_var`` in ``"timediff"`` mode).
        """
        self._check_time_axis(x, "history_compute_stats")

        if self.history_normalization_mode == "none":
            self.history_mean = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32, device=x.device)
            self.history_std = torch.ones((1, 1, 1, 1, 1), dtype=torch.float32, device=x.device)
        elif self.history_normalization_mode == "timediff":
            # time difference mean:
            self.history_diff_mean = torch.mean(self.quadrature(x[:, 1:, ...] - x[:, 0:-1, ...]), dim=(1, 2))

            # time difference std
            self.history_diff_var = torch.mean(
                self.quadrature(torch.square((x[:, 1:, ...] - x[:, 0:-1, ...]) - self.history_diff_mean)), dim=(1, 2)
            )

            # time difference stds
            self.history_diff_mean = copy_to_parallel_region(self.history_diff_mean, "spatial")
            self.history_diff_var = copy_to_parallel_region(self.history_diff_var, "spatial")
        else:
            # quadrature reduces (H, W) -> (B, T, C); the weighted sum over T -> (B, 1, C)
            self.history_mean = torch.sum(self.quadrature(x * self.history_normalization_weights), dim=1, keepdim=True)
            b_, _, c_ = self.history_mean.shape
            self.history_mean = self.history_mean.reshape(b_, 1, c_, 1, 1)

            self.history_std = torch.sum(
                self.quadrature(torch.square(x - self.history_mean) * self.history_normalization_weights),
                dim=1,
                keepdim=True,
            )
            self.history_std = torch.sqrt(self.history_std.reshape(b_, 1, c_, 1, 1))

            # copy to parallel region
            self.history_mean = copy_to_parallel_region(self.history_mean, "spatial")
            self.history_std = copy_to_parallel_region(self.history_std, "spatial")

        return

    def _check_history_stats(self, x, caller: str):
        """
        Controlled-fail validation for history normalization paths.

        - Raises RuntimeError if stats haven't been computed yet (caller forgot
          history_compute_stats before normalize/denormalize).
        - Raises ValueError if the input batch doesn't match the stats batch,
          which in the mean/exponential modes is the only shape dim that isn't a
          broadcast singleton. Common cause: stats were computed on one batch and
          normalize/denormalize is being invoked on a differently-sized input
          without a fresh history_compute_stats call.
        """
        if self.history_mean is None or self.history_std is None:
            raise RuntimeError(
                f"{caller}: history_mean / history_std are None. "
                f"Call history_compute_stats(x) before {caller} (mode='{self.history_normalization_mode}')."
            )
        stats_batch = self.history_mean.shape[0]
        torch._check(
            stats_batch == x.shape[0],
            lambda: (
                f"{caller}: batch mismatch between input ({x.shape[0]}) and cached "
                f"history stats ({stats_batch}). Did you forget to call "
                f"history_compute_stats on the current input before {caller}?"
            ),
        )

    def history_normalize(self, x, target=False):
        r"""
        Normalize a tensor with the cached history statistics.

        A no-op in the ``"none"`` and ``"timediff"`` modes, which do not define
        a state-space normalization.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T, C, H, W)``.
        target : bool, optional
            If ``True``, ``x`` holds only the predicted channels, which lead
            the input channels the statistics were computed on, so only those
            statistics are used. By default ``False``.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as ``x``.

        Raises
        ------
        RuntimeError
            If :meth:`history_compute_stats` has not been called yet.
        """
        if self.history_normalization_mode in ["none", "timediff"]:
            return x

        self._check_time_axis(x, "history_normalize")
        self._check_history_stats(x, caller="history_normalize")

        mean, std = self._history_stats(x, target)

        return (x - mean) / std

    def history_denormalize(self, xn, target=False):
        r"""
        Undo :meth:`history_normalize` using the cached statistics.

        A no-op in the ``"none"`` and ``"timediff"`` modes.

        Parameters
        ----------
        xn : torch.Tensor
            Normalized tensor of shape ``(B, T, C, H, W)``.
        target : bool, optional
            If ``True``, ``xn`` holds only the predicted channels, see
            :meth:`history_normalize`. By default ``False``.

        Returns
        -------
        torch.Tensor
            Denormalized tensor with the same shape as ``xn``.

        Raises
        ------
        RuntimeError
            If :meth:`history_compute_stats` has not been called yet.
        """
        if self.history_normalization_mode in ["none", "timediff"]:
            return xn

        self._check_time_axis(xn, "history_denormalize")
        self._check_history_stats(xn, caller="history_denormalize")

        mean, std = self._history_stats(xn, target)

        return xn * std + mean

    def _history_stats(self, x, target):
        # (B, 1, C, 1, 1), broadcast over time and grid
        if target:
            num_channels = x.shape[2]
            return self.history_mean[:, :, :num_channels], self.history_std[:, :, :num_channels]
        return self.history_mean, self.history_std

    def _mode(self):
        return "train" if self.training else "eval"

    def cache_dynamic_features(self, batch):
        r"""
        Cache the dynamic features of a batch.

        Stores the features the model consumes but does not predict, so that
        :meth:`append_dynamic_features` and :meth:`append_history` can draw on
        them during the rollout. Separate train and eval caches are kept,
        selected by the module's training mode, so that evaluation does not
        clobber the state of an in-flight training step.

        The batch has to carry exactly the features the model takes,
        :attr:`dynamic_features`: a missing one would leave the model without
        an input it was built for, and an extra one would be silently ignored.

        Parameters
        ----------
        batch : dict
            A batch as the loaders emit it, see :mod:`makani.utils.dataloaders.batch`;
            only the dynamic features are read. Without target features, as in
            inference, the target cache is cleared.

        Raises
        ------
        ValueError
            If the features of the batch are not the ones the model takes, or
            a feature has no time axis.
        """
        cache = self._dynamic_cache[self._mode()]

        self._store_features(cache["input"], batch.get(INPUT_DYNAMIC_FEATURES, {}), "input")

        target = batch.get(TARGET_DYNAMIC_FEATURES, None)
        if target is None:
            cache["target"].clear()
        else:
            self._store_features(cache["target"], target, "target")

    def _store_features(self, cache, features, side):
        expected = set(self.dynamic_features)
        if set(features) != expected:
            raise ValueError(
                f"cache_dynamic_features: the model takes the {side} dynamic features {sorted(expected)}, "
                f"the batch carries {sorted(features)}"
            )

        for name, feature in features.items():
            if feature.dim() != 5:
                raise ValueError(
                    f"cache_dynamic_features: {side} feature '{name}' needs a time axis, (B, T, C, H, W), "
                    f"got shape {tuple(feature.shape)}"
                )
            if (side == "input") and (feature.shape[1] != self.n_history + 1):
                raise ValueError(
                    f"cache_dynamic_features: input feature '{name}' has {feature.shape[1]} time steps, "
                    f"the input window n_history + 1 = {self.n_history + 1}"
                )

            # plain tensors rather than registered buffers: per-step scratch from the dataloader,
            # which must not end up in the state dict. Reuse the memory when the shape allows.
            current = cache.get(name, None)
            if (current is not None) and (current.shape == feature.shape):
                current.copy_(feature)
            else:
                cache[name] = feature.clone()

    def get_base_seed(self, default=333):
        r"""
        Return the per-rank base seed of the noise module.

        Parameters
        ----------
        default : int, optional
            Value returned when no input noise is configured, by default ``333``.

        Returns
        -------
        int
            The noise base seed, which encodes the rank's position in the
            ensemble and model-parallel grid.
        """
        if hasattr(self, "input_noise"):
            return self.noise_base_seed
        else:
            return default

    def get_internal_rng(self, gpu=True):
        r"""
        Return the noise module's private generator.

        Exposed so callers can draw additional randomness from the same stream
        as the input noise, keeping everything reproducible under one seed
        instead of mixing in the global RNG.

        Parameters
        ----------
        gpu : bool, optional
            Return the CUDA generator if ``True`` (the default), otherwise the
            CPU generator.

        Returns
        -------
        torch.Generator or None
            ``None`` if no input noise is configured.
        """
        if hasattr(self, "input_noise"):
            if gpu:
                return self.input_noise.rng_gpu
            else:
                return self.input_noise.rng_cpu
        else:
            return None

    def set_rng(self, reset=True, seed=333):
        r"""
        Re-seed the noise module, optionally clearing its state.

        A no-op if no input noise is configured.

        Parameters
        ----------
        reset : bool, optional
            Also zero the noise state, by default ``True``. Leave the state in
            place only if you intend to continue an existing trajectory under a
            new seed.
        seed : int, optional
            New seed, by default ``333``.
        """
        if hasattr(self, "input_noise"):
            self.input_noise.set_rng(seed)
            if reset:
                self.input_noise.reset()
        return

    def get_internal_state(self, tensor=False):
        r"""
        Capture the noise module's state for checkpointing.

        Parameters
        ----------
        tensor : bool, optional
            If ``True``, return the spectral state tensor. If ``False`` (the
            default), return the RNG state instead. A full resume generally
            needs both.

        Returns
        -------
        torch.Tensor or tuple or None
            The state tensor, or a ``(cpu_state, gpu_state)`` RNG tuple. Yields
            ``None`` / ``(None, None)`` when no input noise is configured.

        See Also
        --------
        set_internal_state : restores what this returns.
        """
        if hasattr(self, "input_noise"):
            if tensor:
                state = self.input_noise.get_tensor_state()
            else:
                state = self.input_noise.get_rng_state()
        else:
            if tensor:
                state = None
            else:
                state = (None, None)

        return state

    def set_internal_state(self, state: Union[Tuple, torch.Tensor]):
        r"""
        Restore the noise module's state.

        A no-op if no input noise is configured or if ``state`` is ``None``.

        Parameters
        ----------
        state : tuple or torch.Tensor
            A tensor is treated as the spectral state; a tuple is treated as an
            ``(cpu_state, gpu_state)`` RNG pair. Which one it is is inferred
            from the type, so the two forms returned by
            :meth:`get_internal_state` can both be passed back here.
        """
        if hasattr(self, "input_noise") and (state is not None):
            if isinstance(state, torch.Tensor):
                self.input_noise.set_tensor_state(state)
            else:
                self.input_noise.set_rng_state(*state)

        return

    def update_internal_state(self, replace_state=False, batch_size=None):
        r"""Advance the stochastic noise state by one step.

        A no-op if no input noise is configured.

        Parameters
        ----------
        replace_state : bool, optional
            If ``True``, draw a fresh state from the stationary distribution
            rather than taking one autoregressive step. Required when changing
            the batch size of stateful noise. By default ``False``.
        batch_size : int, optional
            Resize the noise state to this batch size. ``None`` (the default)
            leaves it unchanged.

        Raises
        ------
        RuntimeError
            If a resize of stateful noise is requested while continuing an
            autoregressive sequence; see below.

        Notes
        -----
        ``batch_size`` resizes the state to the given batch; None leaves it at its
        current size. Resizing is guarded: the state carries an ``n_history + 1``
        time axis holding the autoregressive noise history, and reallocating it
        zeroes that history. Doing so while continuing an AR sequence
        (``replace_state`` falsy) would silently restart the noise from zero and
        yield a spin-up transient instead of the intended stationary distribution,
        so it is rejected.

        Only stateful noise (``DiffusionNoiseS2``) is guarded: white and dummy
        noise redraw from scratch every step, so for them a resize destroys
        nothing.

        This matches what every caller already does: the ensemble trainer and the
        inferencer resize only when priming an episode, always with
        ``replace_state=True``, and then roll forward at a fixed batch.
        """
        if hasattr(self, "input_noise"):
            current_batch = self.input_noise.state.shape[0]
            if (
                (batch_size is not None)
                and (not replace_state)
                and (current_batch != batch_size)
                and self.input_noise.is_stateful()
            ):
                raise RuntimeError(
                    f"update_internal_state: refusing to resize the stochastic noise state from "
                    f"batch {current_batch} to {batch_size} while continuing an autoregressive "
                    f"noise sequence (replace_state={replace_state}). The state carries an "
                    f"n_history+1 time history which resizing would zero, silently restarting "
                    f"the noise from zero. Pass replace_state=True to draw a fresh state at the "
                    f"new batch size, or keep the batch size fixed for the whole rollout."
                )
            self.input_noise.update(replace_state=replace_state, batch_size=batch_size)
        return

    def get_static_features(self):
        r"""
        Return a copy of the static feature tensor.

        Returns
        -------
        torch.Tensor or None
            Static features of shape ``(1, n_static, H, W)``, or ``None`` if
            none are configured. Cloned so callers cannot mutate the buffer.
        """
        if self.do_add_static_features:
            return self.static_features.clone()
        else:
            return None

    def get_dynamic_features(self):
        r"""
        Return copies of the cached dynamic features.

        Which cache is read follows the module's training mode.

        Returns
        -------
        inp : dict
            Input features by name, each ``(B, n_history + 1, C_f, H, W)``.
        tar : dict
            Target features by name, empty if the batch had no target.
        """
        cache = self._dynamic_cache[self._mode()]
        inp = {name: feature.clone() for name, feature in cache["input"].items()}
        tar = {name: feature.clone() for name, feature in cache["target"].items()}

        return inp, tar

    def stack_dynamic_features(self, features):
        r"""
        Stack dynamic features along the channel axis, in the order the model takes them.

        For models that pack their input themselves rather than through
        :meth:`append_dynamic_features`.

        Parameters
        ----------
        features : dict
            Features by name, as :meth:`get_dynamic_features` returns them.

        Returns
        -------
        torch.Tensor or None
            ``(B, T, sum C_f, H, W)``, or ``None`` if the model takes no dynamic features.
        """
        if not self.dynamic_features:
            return None

        return torch.cat([features[name] for name in self.dynamic_features], dim=2)

    def correct_bias(self, inp: torch.Tensor):
        r"""
        Subtract the configured bias correction from the input.

        A no-op if no bias correction is configured.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor; the correction must broadcast against it.

        Returns
        -------
        torch.Tensor
            The corrected tensor, same shape as ``inp``.
        """
        if hasattr(self, "bias_correction"):
            inp = inp - self.bias_correction
        return inp


def get_preprocessor(params):
    r"""
    Construct the preprocessor for a given configuration.

    Indirection point for callers, so that selecting a different preprocessor
    implementation stays a change in one place rather than at every call site.
    Currently always returns :class:`Preprocessor2D`.

    Parameters
    ----------
    params : ParamsBase
        Configuration object, forwarded to the preprocessor.

    Returns
    -------
    Preprocessor2D
        The configured preprocessor.
    """
    return Preprocessor2D(params)
