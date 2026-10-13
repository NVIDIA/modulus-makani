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
    the model itself: history handling, static and unpredicted features, noise
    injection, bias correction, and the normalization of the history window.
    Keeping this in one module is what lets the network see a single flat
    channel stack while the training loop still reasons in terms of time steps
    and variable groups.

    The main responsibilities are:

    * **History.** Inputs may carry ``n_history + 1`` time steps. These are
      flattened into the channel dimension for the model
      (:meth:`flatten_history`) and expanded back when needed
      (:meth:`expand_history`); :meth:`append_history` slides the window forward
      by one step during autoregressive rollout.
    * **Unpredicted features.** Quantities the model consumes but does not
      predict, such as the solar zenith angle, are cached per step and appended
      to the input. They must be carried over from the target to the input at
      each rollout step, since the model cannot generate them itself.
    * **Static features.** Time-invariant fields such as orography and the
      land-sea mask, appended to every input.
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

        # unpredicted input channels:
        self.unpredicted_inp_train = None
        self.unpredicted_tar_train = None
        self.unpredicted_inp_eval = None
        self.unpredicted_tar_eval = None

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

        The network takes a single channel stack, so a history window has to be
        flattened before it is passed in.

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

    def add_static_features(self, x):
        r"""
        Append the time-invariant feature channels to the input.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, C + n_static, H, W)``, or ``x`` unchanged if
            no static features are configured.
        """
        if self.do_add_static_features:
            # we need to replicate the grid for each batch:
            static = torch.tile(self.static_features, dims=(x.shape[0], 1, 1, 1))
            x = torch.cat([x, static], dim=1)

        return x

    def remove_static_features(self, x):
        r"""
        Strip the static feature channels appended by :meth:`add_static_features`.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, C + n_static, H, W)``, with the static
            features occupying the trailing channels.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, C, H, W)``, or ``x`` unchanged if no static
            features are configured.
        """
        # only remove if something was added in the first place
        if self.do_add_static_features:
            nfeat = self.static_features.shape[1]
            x = x[:, : x.shape[1] - nfeat, :, :]
        return x

    def append_history(self, x1, x2, step, update_state=True):
        r"""
        Slide the history window forward by one autoregressive step.

        Drops the oldest time step from ``x1`` and appends the new prediction
        ``x2``, producing the input for the next step of the rollout.

        This also advances the cached unpredicted features: quantities such as
        the solar zenith angle are known in advance but not predicted by the
        model, so the target's copy for this step has to be moved into the input
        buffer or the next step would be conditioned on stale values. Whether
        the train or eval buffers are updated follows the module's training
        mode.

        Parameters
        ----------
        x1 : torch.Tensor
            Current input history, shape ``(B, (n_history + 1) * C, H, W)``.
        x2 : torch.Tensor
            Newly predicted step, shape ``(B, C, H, W)``.
        step : int
            Index of the current rollout step, used to select the corresponding
            unpredicted features. Steps beyond the cached range leave the
            unpredicted features untouched.
        update_state : bool, optional
            If ``False``, skip advancing the unpredicted feature buffers and
            only shift the history. By default ``True``.

        Returns
        -------
        torch.Tensor
            Updated history of the same shape as ``x1``. With
            ``n_history == 0`` this is just ``x2``.
        """

        # update the unpredicted input
        if update_state:
            if self.training:
                if (self.unpredicted_tar_train is not None) and (step < self.unpredicted_tar_train.shape[1]):
                    utar = self.unpredicted_tar_train[:, step : (step + 1), :, :, :]
                    if self.n_history == 0:
                        self.unpredicted_inp_train.copy_(utar)
                    else:
                        self.unpredicted_inp_train.copy_(
                            torch.cat([self.unpredicted_inp_train[:, 1:, :, :, :], utar], dim=1)
                        )
            else:
                if (self.unpredicted_tar_eval is not None) and (step < self.unpredicted_tar_eval.shape[1]):
                    utar = self.unpredicted_tar_eval[:, step : (step + 1), :, :, :]
                    if self.n_history == 0:
                        self.unpredicted_inp_eval.copy_(utar)
                    else:
                        self.unpredicted_inp_eval.copy_(
                            torch.cat([self.unpredicted_inp_eval[:, 1:, :, :, :], utar], dim=1)
                        )

        if self.n_history > 0:
            # this is more complicated
            x1 = self.expand_history(x1, nhist=self.n_history + 1)
            x2 = self.expand_history(x2, nhist=1)

            # append
            res = torch.cat([x1[:, 1:, :, :, :], x2], dim=1)

            # flatten again
            res = self.flatten_history(res)
        else:
            res = x2

        return res

    def _append_channels(self, x, xc):

        # x-dimension
        xdim = x.dim()

        # expand history
        x = self.expand_history(x, self.n_history + 1)

        # the cached features, if any; the noise below is appended either way
        appended = []
        if xc is not None:
            # Batch alignment between input and cached unpredicted features.
            # If these diverge, the `cat` below would fail with a cryptic message —
            # fail up-front with context about the likely cause. torch._check keeps
            # this a runtime assertion under torch.compile instead of a graph break.
            torch._check(
                x.shape[0] == xc.shape[0],
                lambda: (
                    f"_append_channels: batch mismatch between input ({x.shape[0]}) and "
                    f"cached unpredicted features ({xc.shape[0]}). "
                    f"Did you cache xz/yz at a different batch size than the current forward?"
                ),
            )
            appended.append(self.expand_history(xc, self.n_history + 1))

        # this routine also adds noise every time a channel gets appended
        if hasattr(self, "input_noise"):
            # run the noise module eagerly: it is complex-valued (SHT) and cannot be inductor-
            # compiled; the method-level disable on its forward is not honored through the
            # nn.Module call, so break at the call site. See _run_eager.
            n = _run_eager(self.input_noise)
            torch._check(
                n.shape[0] == x.shape[0],
                lambda: (
                    f"_append_channels: batch mismatch between input_noise state "
                    f"({n.shape[0]}) and input ({x.shape[0]}). "
                    f"Did you call update_internal_state(batch_size=...) at a different "
                    f"batch than the current forward pass?"
                ),
            )
            if self.input_noise_mode == "concatenate":
                appended.append(n)
            elif self.input_noise_mode == "perturb":
                # fully out-of-place: build a zero noise field and add to all channels
                noise_full = torch.zeros_like(x)
                noise_full[:, :, self.perturb_channels] = n
                x = x + noise_full

        # concatenate
        xo = torch.cat([x] + appended, dim=2)

        # flatten if requested
        if xdim == 4:
            xo = self.flatten_history(xo)

        return xo

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
            History window of shape ``(B, (n_history + 1) * C, H, W)`` or
            ``(B, T, C, H, W)``.

        Returns
        -------
        None
            Results are stored on the module as ``history_mean`` /
            ``history_std`` (or ``history_diff_mean`` / ``history_diff_var``
            in ``"timediff"`` mode).
        """
        if self.history_normalization_mode == "none":
            self.history_mean = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=x.device)
            self.history_std = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=x.device)
        elif self.history_normalization_mode == "timediff":
            # reshaping
            xdim = x.dim()
            if xdim == 4:
                b_, c_, h_, w_ = x.shape
                xr = torch.reshape(x, (b_, (self.n_history + 1), c_ // (self.n_history + 1), h_, w_))
            else:
                xshape = x.shape
                xr = x

            # time difference mean:
            self.history_diff_mean = torch.mean(self.quadrature(xr[:, 1:, ...] - xr[:, 0:-1, ...]), dim=(1, 2))

            # time difference std
            self.history_diff_var = torch.mean(
                self.quadrature(torch.square((xr[:, 1:, ...] - xr[:, 0:-1, ...]) - self.history_diff_mean)), dim=(1, 2)
            )

            # time difference stds
            self.history_diff_mean = copy_to_parallel_region(self.history_diff_mean, "spatial")
            self.history_diff_var = copy_to_parallel_region(self.history_diff_var, "spatial")
        else:
            xdim = x.dim()
            if xdim == 4:
                b_, c_, h_, w_ = x.shape
                xr = torch.reshape(x, (b_, (self.n_history + 1), c_ // (self.n_history + 1), h_, w_))
            else:
                xshape = x.shape
                xr = x

            # mean
            # quadrature reduces (H, W) → (B, T, C); weighted sum over T with keepdim → (B, 1, C)
            self.history_mean = torch.sum(self.quadrature(xr * self.history_normalization_weights), dim=1, keepdim=True)
            # reshape to (B, 1, C, 1, 1) so it broadcasts with xr (B, T, C, H, W)
            b_, _, c_ = self.history_mean.shape
            self.history_mean = self.history_mean.reshape(b_, 1, c_, 1, 1)

            # compute std: (B, T, C, H, W) - (B, 1, C, 1, 1) broadcasts correctly
            self.history_std = torch.sum(
                self.quadrature(torch.square(xr - self.history_mean) * self.history_normalization_weights),
                dim=1,
                keepdim=True,
            )
            self.history_std = torch.sqrt(self.history_std.reshape(b_, 1, c_, 1, 1))

            # squeeze T dim → (B, C, 1, 1); spatial singletons broadcast in history_normalize
            self.history_mean = self.history_mean.reshape(b_, c_, 1, 1)
            self.history_std = self.history_std.reshape(b_, c_, 1, 1)

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
            Tensor of shape ``(B, C, H, W)`` or ``(B, T, C, H, W)``. The
            original layout is restored on return.
        target : bool, optional
            If ``True``, treat ``x`` as a single-step target and use only the
            leading channels of the statistics. If ``False`` (the default),
            treat it as a full history window and tile the statistics over the
            ``n_history + 1`` steps.

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

        self._check_history_stats(x, caller="history_normalize")

        xdim = x.dim()
        if xdim == 5:
            xshape = x.shape
            x = self.flatten_history(x)

        # normalize
        if target:
            # strip off the unpredicted channels
            xn = (x - self.history_mean[:, : x.shape[1], :, :]) / self.history_std[:, : x.shape[1], :, :]
        else:
            # tile to include history
            hm = torch.tile(self.history_mean, (1, self.n_history + 1, 1, 1))
            hs = torch.tile(self.history_std, (1, self.n_history + 1, 1, 1))
            xn = (x - hm) / hs

        if xdim == 5:
            xn = torch.reshape(xn, xshape)

        return xn

    def history_denormalize(self, xn, target=False):
        r"""
        Undo :meth:`history_normalize` using the cached statistics.

        A no-op in the ``"none"`` and ``"timediff"`` modes.

        Parameters
        ----------
        xn : torch.Tensor
            Normalized tensor of shape ``(B, C, H, W)`` or ``(B, T, C, H, W)``.
        target : bool, optional
            If ``True``, treat ``xn`` as a single-step target and use only the
            leading channels of the statistics; otherwise tile them over the
            history window. By default ``False``.

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

        self._check_history_stats(xn, caller="history_denormalize")

        xndim = xn.dim()
        if xndim == 5:
            xnshape = xn.shape
            xn = self.flatten_history(xn)

        # de-normalize
        if target:
            # strip off the unpredicted channels
            x = xn * self.history_std[:, : xn.shape[1], :, :] + self.history_mean[:, : xn.shape[1], :, :]
        else:
            # tile to include history
            hm = torch.tile(self.history_mean, (1, self.n_history + 1, 1, 1))
            hs = torch.tile(self.history_std, (1, self.n_history + 1, 1, 1))
            x = xn * hs + hm

        if xndim == 5:
            x = torch.reshape(x, xnshape)

        return x

    def _ensure_cached(self, name: str, tensor):
        """
        Centralized rebind for the cached unpredicted-feature attributes.

        - tensor is None: the cached attribute is cleared to None.
        - current is None or has a different shape: store a fresh clone.
        - shapes match: in-place ``copy_`` to reuse the existing memory.

        These are plain attributes (not registered buffers): they are per-step scratch
        populated from dataloader output on-device, and should not appear in ``state_dict``.
        """
        current = getattr(self, name)
        if tensor is None:
            setattr(self, name, None)
            return
        if (current is not None) and (current.shape == tensor.shape):
            current.copy_(tensor)
        else:
            setattr(self, name, tensor.clone())

    def cache_unpredicted_features(self, x, y, xz=None, yz=None):
        r"""
        Cache the unpredicted input and target features for this sample.

        Stores the auxiliary channels the model consumes but does not predict,
        so that :meth:`append_history` and :meth:`append_unpredicted_features`
        can draw on them during rollout. Separate train and eval caches are
        kept, selected by the module's training mode, so that evaluation does
        not clobber the state of an in-flight training step.

        Parameters
        ----------
        x : torch.Tensor
            Model input. Passed through unchanged.
        y : torch.Tensor
            Model target. Passed through unchanged.
        xz : torch.Tensor, optional
            Unpredicted input features, shape ``(B, T, C_z, H, W)``. ``None``
            clears the cache.
        yz : torch.Tensor, optional
            Unpredicted target features, shape ``(B, steps, C_z, H, W)``.
            ``None`` clears the cache.

        Returns
        -------
        x : torch.Tensor
            The input, unchanged.
        y : torch.Tensor
            The target, unchanged.
        """
        if self.training:
            self._ensure_cached("unpredicted_inp_train", xz)
            self._ensure_cached("unpredicted_tar_train", yz)
        else:
            self._ensure_cached("unpredicted_inp_eval", xz)
            self._ensure_cached("unpredicted_tar_eval", yz)

        return x, y

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

    def append_unpredicted_features(self, inp, target=False):
        r"""
        Append the cached unpredicted features to a tensor.

        Also injects the input noise when configured, since noise is applied at
        the same point in the pipeline. The train or eval cache is selected by
        the module's training mode; if the relevant cache is empty, only the
        noise is applied, and without noise the input is returned unchanged.

        Parameters
        ----------
        inp : torch.Tensor
            Tensor of shape ``(B, C, H, W)`` or ``(B, T, C, H, W)``. The
            original layout is restored on return.
        target : bool, optional
            Append the target-side unpredicted features rather than the
            input-side ones, by default ``False``.

        Returns
        -------
        torch.Tensor
            Tensor with the unpredicted feature channels appended.
        """
        if self.training:
            cache = self.unpredicted_tar_train if target else self.unpredicted_inp_train
        else:
            cache = self.unpredicted_tar_eval if target else self.unpredicted_inp_eval

        # noise is injected here as well, so an empty cache only means nothing to do without it
        if (cache is None) and not hasattr(self, "input_noise"):
            return inp

        return self._append_channels(inp, cache)

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

    def get_unpredicted_features(self):
        r"""
        Return copies of the cached unpredicted input and target features.

        Which cache is read follows the module's training mode.

        Returns
        -------
        inpu : torch.Tensor or None
            Cached unpredicted input features, or ``None`` if unset.
        taru : torch.Tensor or None
            Cached unpredicted target features, or ``None`` if unset.
        """
        if self.training:
            if self.unpredicted_inp_train is not None:
                inpu = self.unpredicted_inp_train.clone()
            else:
                inpu = None
            if self.unpredicted_tar_train is not None:
                taru = self.unpredicted_tar_train.clone()
            else:
                taru = None
        else:
            if self.unpredicted_inp_eval is not None:
                inpu = self.unpredicted_inp_eval.clone()
            else:
                inpu = None
            if self.unpredicted_tar_eval is not None:
                taru = self.unpredicted_tar_eval.clone()
            else:
                taru = None

        return inpu, taru

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
