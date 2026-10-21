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

import math
import numpy as np
import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import torch
import torch.nn as nn
from torch import amp

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils import comm
from torch_harmonics.distributed import split_tensor_along_dim


class BaseNoiseS2(nn.Module):
    r"""
    Abstract base class for random fields on the sphere :math:`S^2`.

    Noise for an ensemble weather model has to be spatially correlated: white
    noise on a lat-lon grid is neither isotropic (grid cells shrink toward the
    poles) nor physically plausible as a perturbation. All subclasses therefore
    generate noise in *spectral* space and transform it to the grid with an
    inverse SHT, which makes the correlation structure an explicit choice of
    angular power spectrum and keeps the field isotropic by construction.

    This base class owns the machinery every variant needs: the inverse SHT
    (distributed when spatial model parallelism is active), private RNGs, and a
    ``state`` buffer holding the current spectral coefficients. Subclasses
    define the spectrum and decide whether they are stateful (the field evolves
    from step to step, as in :class:`DiffusionNoiseS2`) or stateless (each draw
    is independent, as in :class:`IsotropicGaussianRandomFieldS2`).

    The state buffer is non-persistent: it is a per-run scratch tensor and is
    deliberately kept out of checkpoints, so restoring one does not resurrect a
    stale noise realization.

    Parameters
    ----------
    img_shape : (int, int)
        Output grid as ``(nlat, nlon)``.
    batch_size : int
        Initial batch size of the state buffer. Automatically resized when a
        different batch size is passed to ``update`` or ``reset``.
    num_channels : int
        Number of noise channels.
    num_time_steps : int
        Number of time steps held in the state.
    grid_type : str, optional
        Grid the noise is generated on, by default ``"equiangular"``.
    lmax : int, optional
        Spectral truncation. Defaults to the maximum supported by the grid;
        lowering it produces a smoother field.
    seed : int, optional
        Seed for the private CPU and CUDA generators, by default ``333``.
    reflect : bool, optional
        If ``True``, negate every draw. Used for antithetic ensemble pairing:
        two members sharing a seed but differing in this flag produce exactly
        opposite perturbations, which cancels first-order sampling error in the
        ensemble mean. By default ``False``.
    **kwargs
        Ignored; present so noise configs can pass extra keys.

    See Also
    --------
    build_noise : factory constructing the right subclass from a config dict.
    """

    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps,
        grid_type="equiangular",
        lmax=None,
        seed=333,
        reflect=False,
        **kwargs,
    ):
        super().__init__()

        # Number of latitudinal modes.
        self.nlat, self.nlon = img_shape
        self.num_channels = num_channels
        self.num_time_steps = num_time_steps
        self.reflect = reflect

        # Inverse SHT
        if comm.get_size("spatial") > 1:
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
            self.isht = thd.DistributedInverseRealSHT(self.nlat, self.nlon, lmax=lmax, mmax=lmax, grid=grid_type)
            self.lmax_local = self.isht.l_shapes[comm.get_rank("h")]
            self.mmax_local = self.isht.m_shapes[comm.get_rank("w")]
            self.nlat_local = self.isht.lat_shapes[comm.get_rank("h")]
            self.nlon_local = self.isht.lon_shapes[comm.get_rank("w")]
        else:
            self.isht = th.InverseRealSHT(self.nlat, self.nlon, lmax=lmax, mmax=lmax, grid=grid_type)
            self.lmax_local = self.isht.lmax
            self.mmax_local = self.isht.mmax
            self.nlat_local = self.nlat
            self.nlon_local = self.nlon

        self.lmax = self.isht.lmax
        self.mmax = self.isht.mmax

        # generator objects:
        self.set_rng(seed=seed)

        # allocate the state buffer via the centralized helper; subclasses customize the
        # per-batch shape by overriding _state_shape_suffix.
        self._ensure_state(batch_size, device=torch.device("cpu"), dtype=torch.float32)

    @property
    def _state_shape_suffix(self):
        """
        Shape of the state buffer beyond the batch dim. Subclasses override this to
        customize the layout (e.g. DummyNoiseS2 stores state in spatial, not spectral, form).
        """
        return (self.num_time_steps, self.num_channels, self.lmax_local, self.mmax_local, 2)

    def _ensure_state(self, batch_size, device=None, dtype=None):
        """
        Single source of truth for (re)allocating ``self.state``.

        This is the only method that (re-)registers the state buffer. Calling it with
        the same shape as the current state is a no-op; calling it with a different
        batch size re-registers the buffer so buffer semantics (``.to(device)``,
        ``state_dict`` membership via ``_buffers``, etc.) are preserved rather than
        relying on the ``__setattr__`` hook.
        """
        if device is None:
            device = self.state.device if ("state" in self._buffers) else torch.device("cpu")
        if dtype is None:
            dtype = self.state.dtype if ("state" in self._buffers) else torch.float32

        target_shape = (batch_size,) + tuple(self._state_shape_suffix)
        if ("state" not in self._buffers) or (tuple(self.state.shape) != target_shape):
            self.register_buffer(
                "state",
                torch.zeros(target_shape, dtype=dtype, device=device),
                persistent=False,
            )

    def _set_state(self, state):
        """
        Replace the state by a new tensor, never writing into the current one.

        The current state may still be in the autograd graph, as for learnable stateful noise
        during a rollout, see :meth:`DiffusionNoiseS2.update`, or be an inference tensor, if it
        was written in inference mode, as in validation or inference. Writing into it in place
        would corrupt the backward of the steps that used it, or fail outside inference mode.
        """
        self._buffers["state"] = state

    def is_stateful(self):
        r"""
        Whether the noise carries state across calls.

        Callers use this to decide whether the noise module must be
        checkpointed, may be resized mid-sequence, or kept in sync across
        ensemble members. Must be overridden by subclasses.

        Returns
        -------
        bool
            ``True`` if successive draws depend on the previous state.

        Raises
        ------
        NotImplementedError
            Always, unless overridden.
        """
        raise NotImplementedError("is_stateful method not implemented for this noise class")

    def extra_repr(self):
        r"""
        Extra fields shown in the module's ``repr``.

        Returns
        -------
        str
            Grid shape, channel and time step counts, spectral truncation and
            reflection flag.
        """
        return (
            f"img_shape=({self.nlat}, {self.nlon}), "
            f"num_channels={self.num_channels}, num_time_steps={self.num_time_steps}, "
            f"lmax={self.lmax}, reflect={self.reflect}"
        )

    def set_rng(self, seed=333):
        r"""
        Re-seed the module's private CPU and CUDA generators.

        The noise stream is deliberately independent of the global RNG, so
        seeding it explicitly is the only way to make a run reproducible or to
        decorrelate ensemble members from one another.

        Parameters
        ----------
        seed : int, optional
            Seed applied to both generators, by default ``333``. The CUDA
            generator is only created when CUDA is available.
        """
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed)

    def reset(self, batch_size=None):
        r"""
        Zero the internal state, optionally resizing it.

        Not needed between rollouts driven by a
        :class:`~makani.utils.dataloaders.rollout.Rollout`, which replaces the
        state at its first step.

        Parameters
        ----------
        batch_size : int, optional
            New batch size. If given and different from the current one, the
            state is reallocated at that size.
        """
        if batch_size is not None:
            self._ensure_state(batch_size)
        with torch.no_grad():
            self._set_state(torch.zeros_like(self.state))

    def update(self, replace_state=False, batch_size=None):
        r"""
        Draw a fresh set of spectral coefficients into the state.

        The base implementation is memoryless: it overwrites the state with new
        standard normal coefficients. Stateful subclasses override this to
        propagate the previous state forward in time instead.

        Parameters
        ----------
        replace_state : bool, optional
            Accepted for interface compatibility with stateful subclasses,
            where it selects whether the new sample replaces the state or is
            appended to it. Ignored here, since the base class always replaces.
        batch_size : int, optional
            If given, resize the state buffer to this batch size before drawing.
        """

        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            newstate = torch.empty_like(self.state)
            if self.state.is_cuda:
                newstate.normal_(mean=0.0, std=1.0, generator=self.rng_gpu)
            else:
                newstate.normal_(mean=0.0, std=1.0, generator=self.rng_cpu)

            if self.reflect:
                newstate = -newstate

        self._set_state(newstate)

        return

    def set_rng_state(self, cpu_state, gpu_state):
        r"""
        Restore the generators from previously captured states.

        Together with :meth:`get_rng_state` this makes a run resumable: without
        it, restarting from a checkpoint would continue with a different noise
        stream than an uninterrupted run would have produced.

        Parameters
        ----------
        cpu_state : torch.ByteTensor or None
            State for the CPU generator, as returned by :meth:`get_rng_state`.
            ``None`` leaves the CPU generator untouched.
        gpu_state : torch.ByteTensor or None
            State for the CUDA generator. Ignored if CUDA is unavailable or if
            ``None``.
        """
        if cpu_state is not None:
            self.rng_cpu.set_state(cpu_state)
        if torch.cuda.is_available() and (gpu_state is not None):
            self.rng_gpu.set_state(gpu_state)

        return

    def get_rng_state(self):
        r"""
        Capture the current generator states for checkpointing.

        Returns
        -------
        cpu_state : torch.ByteTensor
            State of the CPU generator.
        gpu_state : torch.ByteTensor or None
            State of the CUDA generator, or ``None`` if CUDA is unavailable.

        See Also
        --------
        set_rng_state : restores what this returns.
        """
        cpu_state = self.rng_cpu.get_state()
        gpu_state = None
        if torch.cuda.is_available():
            gpu_state = self.rng_gpu.get_state()

        return cpu_state, gpu_state

    def get_tensor_state(self):
        r"""
        Return a detached copy of the current spectral state.

        Returns
        -------
        torch.Tensor
            Copy of the state buffer, shape
            ``(batch, *self._state_shape_suffix)``. A copy rather than a view,
            so the caller can hold on to it across further ``update`` calls.
        """
        return self.state.detach().clone()

    def set_tensor_state(self, newstate):
        r"""
        Replace the spectral state, taking on the batch size of the new one.

        The state becomes a detached copy of ``newstate``, on the device and
        with the dtype of the current one, see :meth:`_set_state`: the caller's
        tensor is never written to, nor kept.

        Parameters
        ----------
        newstate : torch.Tensor
            Replacement state. Everything beyond the batch dimension must match
            the module's expected layout; only the batch size may differ.

        Raises
        ------
        ValueError
            If the shape beyond the batch dimension does not match, so that a
            state of the wrong layout is never installed.
        """
        # Validate the state layout (everything beyond the batch dim) matches this
        # noise module's expected suffix BEFORE touching `self.state`. Only the batch
        # dim may differ, the state takes the batch of the new one. Any other
        # difference raises `ValueError` up-front with a useful message instead of
        # silently installing a state of the wrong layout.
        expected_suffix = tuple(self._state_shape_suffix)
        actual_suffix = tuple(newstate.shape[1:]) if newstate.dim() >= 1 else tuple(newstate.shape)
        if actual_suffix != expected_suffix:
            raise ValueError(
                f"set_tensor_state: shape mismatch beyond batch dim. "
                f"Expected suffix {expected_suffix}, got {actual_suffix} "
                f"(full newstate.shape={tuple(newstate.shape)}, "
                f"current state.shape={tuple(self.state.shape)})."
            )
        # a copy on the device and in the dtype of the state, so that the caller can keep using its tensor
        with torch.no_grad():
            self._set_state(newstate.detach().to(device=self.state.device, dtype=self.state.dtype).clone())
        return


class IsotropicGaussianRandomFieldS2(BaseNoiseS2):
    r"""
    Isotropic Gaussian random field on the unit sphere. Stateless.

    Draws standard normal spherical harmonic coefficients, scales them by a
    per-degree standard deviation, and transforms to the grid. Because the
    scaling depends only on the degree :math:`l` and not the order :math:`m`,
    the resulting field is statistically isotropic -- no direction or location
    on the sphere is special, which a field generated directly on a lat-lon
    grid would not satisfy.

    The angular power spectrum follows a power law,

    .. math::

        \sigma_l = \sigma \sqrt{\frac{(2l+1)^{-\alpha}}{Z}},
        \qquad
        Z = \sum_l \frac{(2l+1)\,(2l+1)^{-\alpha}}{4\pi}

    where the normalization :math:`Z` fixes the pointwise variance of the field
    at :math:`\sigma^2` regardless of :math:`\alpha` or the truncation, so
    changing the correlation length does not silently change the amplitude.
    :math:`\alpha = 0` gives white noise; larger :math:`\alpha` damps high
    degrees and yields a smoother, longer-correlated field.

    Each call to ``update`` draws an independent realization, so the noise is
    stateless and :meth:`is_stateful` returns ``False``.

    Parameters
    ----------
    img_shape : (int, int)
        Output grid as ``(nlat, nlon)``.
    batch_size : int
        Initial batch size of the state buffer.
    num_channels : int
        Number of noise channels.
    num_time_steps : int, optional
        Number of time steps held in the state, by default ``1``.
    sigma : float, default is 1.0
        Scale parameter corresponding to the diagonal entry of the covariance
        kernel, i.e. the pointwise standard deviation of the field.
    alpha : float, default is 0.0
        Decay factor in the angular power spectrum. White noise corresponds to
        ``alpha = 0.0``.
    grid_type : string, default is "equiangular"
        Grid type. Currently supports ``"equiangular"`` and ``"legendre-gauss"``.
    lmax : int, optional
        Spectral truncation. Defaults to the maximum supported by the grid;
        lowering it produces a smoother field.
    seed : int, optional
        Seed for the private generators, by default ``333``.
    reflect : bool, optional
        Negate every draw, for antithetic ensemble pairing. By default ``False``.
    learnable : bool, default is False
        Parameter which enables learnable Gaussian noise: the per-degree
        standard deviations become trainable instead of fixed, letting the
        model adapt the correlation structure during training.
    **kwargs
        Ignored; present so noise configs can pass extra keys.

    References
    ----------
    [1] Lang, A.; Schwab C.; Isotropic Gaussian random fields on the sphere:
    regularity, fast simulation and stochastic partial differential equations;
    The Annals of Applied Probability; 2015, Vol. 25, No. 6, 3047-3094;
    DOI: 10.1214/14-AAP1067

    See Also
    --------
    DiffusionNoiseS2 : stateful noise with temporal correlation.
    """

    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps=1,
        sigma=1.0,
        alpha=0.0,
        grid_type="equiangular",
        lmax=None,
        seed=333,
        reflect=False,
        learnable=False,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=num_time_steps,
            grid_type=grid_type,
            lmax=lmax,
            seed=seed,
            reflect=reflect,
        )

        # stash config for extra_repr
        self.sigma = sigma
        self.alpha = alpha
        self.learnable = learnable

        if not isinstance(alpha, float):
            alpha = float(alpha)

        # Compute ls, angular power spectrum and sigma_l:
        ls = torch.arange(self.lmax).reshape(-1, 1)
        ms = torch.arange(self.mmax)
        power_spectrum = torch.pow(2 * ls + 1, -alpha)
        norm_factor = torch.sum((2 * ls + 1) * power_spectrum / 4.0 / math.pi)
        sigma_l = sigma * torch.sqrt(power_spectrum / norm_factor)
        sigma_l = torch.where(ms <= ls, sigma_l, 0.0)

        # the new shape is B, T, C, L, M
        sigma_l = sigma_l.reshape((1, 1, 1, self.lmax, self.mmax)).to(dtype=torch.float32)

        # split tensor
        if comm.get_size("h") > 1:
            sigma_l = split_tensor_along_dim(sigma_l, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]

        # split tensor
        if comm.get_size("w") > 1:
            sigma_l = split_tensor_along_dim(sigma_l, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]

        # register buffer
        if learnable:
            self.register_parameter("sigma_l", nn.Parameter(sigma_l))
            self.sigma_l.sharded_dims_mp = [None, None, None, "h", "w"]
        else:
            self.register_buffer("sigma_l", sigma_l, persistent=False)

    @override
    def is_stateful(self):
        r"""
        Whether the noise carries state across calls.

        Returns
        -------
        bool
            Always ``False``: every draw is an independent realization.
        """
        return False

    def extra_repr(self):
        r"""
        Extra fields shown in the module's ``repr``.

        Returns
        -------
        str
            The base class fields plus ``sigma``, ``alpha`` and ``learnable``.
        """
        return super().extra_repr() + f", sigma={self.sigma}, alpha={self.alpha}, learnable={self.learnable}"

    # run eager: the noise field is complex-valued (torch.complex + inverse SHT), and
    # inductor's Triton backend has no mapping for complex dtypes (KeyError: 'complex64'
    # in signature_of). Disabling compilation graph-breaks cleanly so the complex ops
    # execute in eager where they are supported.
    @torch.compiler.disable
    @override
    def forward(self, update_internal_state=False):
        r"""
        Scale the stored coefficients by the power spectrum and transform to the grid.

        Parameters
        ----------
        update_internal_state : bool, optional
            If ``True``, draw a fresh set of coefficients *after* producing the
            output, so the next call returns an independent realization. By
            default ``False``, which returns the same field until ``update`` is
            called explicitly.

        Returns
        -------
        torch.Tensor
            Real-valued noise of shape
            ``(batch, num_time_steps, num_channels, nlat_local, nlon_local)``.
        """

        # combine channels and time:
        # torch.view_as_complex on a registered buffer hits a torch.compile/Inductor
        # bug (set_() size mismatch when itemsize changes float32→complex64); construct
        # the complex tensor explicitly instead.
        _s = self.state / math.sqrt(2)
        cstate = torch.complex(_s[..., 0], _s[..., 1]) * self.sigma_l
        batch_size = cstate.shape[0]

        # flatten history
        cstate = cstate.reshape(batch_size, self.num_time_steps * self.num_channels, self.lmax_local, self.mmax_local)

        # transform
        with amp.autocast(device_type=cstate.device.type, enabled=False):
            eta = self.isht(cstate)

        # expand history
        eta = eta.reshape(batch_size, self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local)

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return eta


def toep(c, r=None):
    r"""
    Construct a Toeplitz matrix from its first column and row.

    Every diagonal of the result is constant: ``T[i, j]`` depends only on
    ``i - j``. Used here to build the discount matrix of powers of :math:`\phi`
    that correlates a freshly drawn noise history, so that resampling the whole
    history reproduces the temporal correlation of the process instead of
    giving independent steps.

    Vendored from SciPy to avoid a runtime dependency on it.

    Parameters
    ----------
    c : array_like
        First column of the matrix. Flattened if not already 1D.
    r : array_like, optional
        First row of the matrix. ``r[0]`` is ignored -- ``c[0]`` sets the
        diagonal. Defaults to ``conj(c)``, which gives a Hermitian matrix.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(c), len(r))``.

    References
    ----------
    Taken from scipy:
    https://github.com/scipy/scipy/blob/v1.13.0/scipy/linalg/_special_matrices.py#L17-L77
    """

    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1-D array containing a reversed c followed by r[1:] that could be
    # strided to give us toeplitz matrix.
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]

    return np.lib.stride_tricks.as_strided(vals[len(c) - 1 :], shape=out_shp, strides=(-n, n)).copy()


class DiffusionNoiseS2(BaseNoiseS2):
    r"""
    Temporally correlated random field from a diffusion process on the sphere. Stateful.

    Perturbations in an ensemble forecast should not be redrawn independently at
    every step -- that produces noise the model damps out immediately rather
    than a coherent perturbation that grows. This module instead evolves the
    spectral coefficients as an Ornstein-Uhlenbeck process, so each step is
    correlated with the last:

    .. math::

        \eta_{t+1} = \phi\, \eta_t + \sqrt{1 - \phi^2}\; \sigma_l\, \xi_t,
        \qquad \phi = e^{-\lambda}

    with :math:`\xi_t` standard normal. The prefactor
    :math:`\sqrt{1-\phi^2}` keeps the process stationary: the marginal variance
    stays at :math:`\sigma^2` for every :math:`t` instead of drifting as the
    correlation length changes. :math:`\lambda = \Delta t / \tau` sets how fast
    the field decorrelates in time, while ``kT`` sets the spatial correlation
    length through the angular power spectrum.

    Both ``kT`` and ``lambd`` may be given per channel, so different variables
    can carry perturbations at different scales.

    Parameters
    ----------
    img_shape : (int, int)
        Output grid as ``(nlat, nlon)``.
    batch_size : int
        Initial batch size of the state buffer.
    num_channels : int
        Number of noise channels.
    num_time_steps : int, optional
        Number of time steps held in the state, by default ``1``.
    sigma : float, default is 1
        Stationary standard deviation.
    kT : float or List, default is 0.5 * (500 km / 6370 km)^2 = 0.00308057
        Spatial correlation length. If this is a list it has to match
        ``num_channels``.
    lambd : float or List, default is 1.0
        Temporal correlation length, should be set to ``(t / tau)``. If this is
        a list it has to match ``num_channels``.
    grid_type : string, default is "equiangular"
        Grid type. Currently supports ``"equiangular"`` and ``"legendre-gauss"``.
    lmax : int, optional
        Spectral truncation. Defaults to the maximum supported by the grid;
        lowering it produces a smoother field.
    seed : int, optional
        Seed for the private generators, by default ``333``.
    reflect : bool, optional
        Negate every draw, for antithetic ensemble pairing. By default ``False``.
    learnable : bool, default is False
        Parameter which enables learnable Diffusion noise: the per-degree
        standard deviations become trainable.
    **kwargs
        Ignored; present so noise configs can pass extra keys.

    References
    ----------
    Palmer, T. et al.; Stochastic parametrization and model uncertainty; ECMWF
    Technical Memorandum 598, 2009, appendix 8.1.
    https://www.ecmwf.int/sites/default/files/elibrary/2009/11577-stochastic-parametrization-and-model-uncertainty.pdf

    See Also
    --------
    IsotropicGaussianRandomFieldS2 : stateless noise with no temporal correlation.
    """

    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps=1,
        sigma=1.0,
        kT=0.5 * (500.0 / 6370.0) ** 2,
        lambd=1.0,
        grid_type="equiangular",
        lmax=None,
        seed=333,
        reflect=False,
        learnable=False,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=num_time_steps,
            grid_type=grid_type,
            lmax=lmax,
            seed=seed,
            reflect=reflect,
        )

        # stash config for extra_repr (store originals before processing into tensors below)
        self.sigma = sigma
        self.kT = kT
        self.lambd = lambd
        self.learnable = learnable

        # Compute l:
        ls = torch.arange(self.lmax)

        # make sure kT is a torch.Tensor
        if isinstance(kT, list):
            kT = torch.as_tensor(kT)
            if len(kT.shape) != 1:
                raise ValueError(f"expected kT to be a 1D tensor, got shape {tuple(kT.shape)}")
            if kT.shape[0] != num_channels:
                raise ValueError(f"expected kT to have {num_channels} entries (one per channel), got {kT.shape[0]}")
        else:
            kT = torch.as_tensor([kT]).repeat(num_channels)
        kT = kT.reshape(self.num_channels, 1)

        # same for lambd
        if isinstance(lambd, list):
            lambd = torch.as_tensor(lambd)
            if len(lambd.shape) != 1:
                raise ValueError(f"expected lambd to be a 1D tensor, got shape {tuple(lambd.shape)}")
            if lambd.shape[0] != num_channels:
                raise ValueError(
                    f"expected lambd to have {num_channels} entries (one per channel), got {lambd.shape[0]}"
                )
        else:
            lambd = torch.as_tensor([lambd]).repeat(num_channels)
        lambd = lambd.reshape(self.num_channels, 1)

        # f-tensor:
        ektllp1 = torch.exp(-kT * ls * (ls + 1))
        F0norm = torch.sum((2 * ls[1:] + 1) * ektllp1[..., 1:], dim=-1, keepdim=True)
        # create a discount vector in time:
        phi = torch.exp(-lambd)
        F0 = sigma * torch.sqrt(0.5 * (1 - phi**2) / F0norm)
        sigma_l = F0 * torch.exp(-0.5 * kT * ls * (ls + 1))
        # we multiply by 4 pi to get the correct variance. Check ECMWF docs and their Spherical Harmonic normalization
        sigma_l = math.sqrt(4 * math.pi) * sigma_l

        # the new shape is C, L, M
        phi = phi.reshape((self.num_channels, 1, 1)).to(dtype=torch.float32)
        # the new shape is B, T, C, L, M
        sigma_l = sigma_l.reshape((1, 1, self.num_channels, self.lmax, 1)).to(dtype=torch.float32)

        # split tensor
        if comm.get_size("h") > 1:
            sigma_l = split_tensor_along_dim(sigma_l, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]

        # unsqueeze complex dim
        phi = phi.unsqueeze(-1)
        sigma_l = sigma_l.unsqueeze(-1)

        # register buffer
        if learnable:
            self.phi = nn.Parameter(phi)
            self.phi.is_shared_mp = ["matmul", "h", "w"]
            self.phi.sharded_dims_mp = [None, None, None]
            self.sigma_l = nn.Parameter(sigma_l)
            self.sigma_l.is_shared_mp = ["matmul", "w"]
            self.sigma_l.sharded_dims_mp = [None, None, None, "h", None, None]
        else:
            self.register_buffer("phi", phi, persistent=False)
            self.register_buffer("sigma_l", sigma_l, persistent=False)

        # state buffer is already allocated by BaseNoiseS2.__init__ via _ensure_state

        # if num_time_steps > 1, we need the toeplitz matrix for the discounts:
        #            [    1,     0,   0, 0]
        # discount = [  phi,     1,   0, 0]
        #            [phi^2,   phi,   1, 0]
        #            [phi^3, phi^2, phi, 1]
        if self.num_time_steps > 1:
            if learnable:
                raise NotImplementedError("num_time_steps>1 learnable diffusion noise not supported")

            discount = []
            phi_flat = self.phi.reshape(-1)
            for phi_tmp in phi_flat.tolist():
                phivec = np.power(phi_tmp, np.arange(0, self.num_time_steps))
                disc = torch.as_tensor(toep(phivec, np.zeros(self.num_time_steps)))
                disc = disc.to(dtype=torch.float32)
                discount.append(disc)
            discount = torch.stack(discount, dim=0)
            self.register_buffer("discount", discount, persistent=False)

    @override
    def is_stateful(self):
        r"""
        Whether the noise carries state across calls.

        Returns
        -------
        bool
            Always ``True``: each step is correlated with the previous one, so
            the state must be checkpointed, and replaced or restored at the
            start of a rollout.
        """
        return True

    def extra_repr(self):
        r"""
        Extra fields shown in the module's ``repr``.

        Returns
        -------
        str
            The base class fields plus ``sigma``, ``kT``, ``lambd`` and
            ``learnable``.
        """
        return (
            super().extra_repr() + f", sigma={self.sigma}, kT={self.kT}, lambd={self.lambd}, learnable={self.learnable}"
        )

    @override
    def update(self, replace_state=False, batch_size=None):
        r"""
        Advance the noise process by one step, or resample the whole history.

        Parameters
        ----------
        replace_state : bool, optional
            If ``False`` (the default), take one autoregressive step: the
            existing state is damped by :math:`\phi` and a fresh innovation is
            added, so the new field stays correlated with the old one. When
            ``num_time_steps > 1`` the oldest step is dropped and the new one
            appended.

            If ``True``, discard the state and draw a fresh history from the
            *stationary* distribution. The first time step is scaled by
            :math:`1/\sqrt{1-\phi^2}` and, for ``num_time_steps > 1``, the
            history is correlated by a Toeplitz discount matrix of powers of
            :math:`\phi`. This matters when starting a rollout: drawing
            independent steps instead would begin from a field with the wrong
            variance and no temporal structure, and the process would need many
            steps to spin up.
        batch_size : int, optional
            If given, resize the state buffer to this batch size before drawing.
        """
        if batch_size is not None:
            self._ensure_state(batch_size)

        # the innovations are drawn without gradients
        with torch.no_grad():
            # draw either the full T-step history (replace) or a single step (AR)
            if replace_state:
                eta = torch.empty_like(self.state)
            else:
                B = self.state.shape[0]
                eta = torch.empty(
                    (B, 1, self.num_channels, self.lmax_local, self.mmax_local, 2),
                    dtype=self.state.dtype,
                    device=self.state.device,
                )
            if self.state.is_cuda:
                eta.normal_(mean=0.0, std=1.0, generator=self.rng_gpu)
            else:
                eta.normal_(mean=0.0, std=1.0, generator=self.rng_cpu)

        # the recurrence keeps its graph if the noise is learnable: phi and sigma_l enter the
        # state only here, and get their gradients through it in the steps that use the state
        with torch.set_grad_enabled(self.learnable and torch.is_grad_enabled()):
            with amp.autocast(device_type=self.state.device.type, enabled=False):
                # multiply by sigma
                eta_l = self.sigma_l * eta

                # reflect if required:
                if self.reflect:
                    eta_l = -eta_l

                if not replace_state:
                    # update previous state
                    if self.num_time_steps > 1:
                        last_state = self.state[:, -1, ...].unsqueeze(1)
                        newstep = self.phi * last_state + eta_l
                        newstate = torch.cat([self.state[:, 1:, ...], newstep], dim=1)
                    else:
                        newstate = self.phi * self.state + eta_l
                else:
                    # the very first element in the time history requires a different weighting to sample the
                    # stationary distribution; out of place, so that the graph stays intact
                    newstate = torch.cat([eta_l[:, :1] / torch.sqrt(1.0 - self.phi**2), eta_l[:, 1:]], dim=1)
                    # get the right history by multiplying with the discount matrix
                    if self.num_time_steps > 1:
                        newstate = torch.einsum("ctr,brclmu->btclmu", self.discount, newstate).contiguous()

        # if the noise is learnable, the state is the tensor of the recurrence, so that the graph reaches phi
        # and sigma_l. A rollout has to replace the state at its first step, or restore a detached snapshot
        # (Rollout(noise_state=...)): continuing the live state of a previous rollout would chain into the
        # graph of that rollout, which is gone after its backward
        self._set_state(newstate)

        return

    # run eager: complex-valued (torch.complex + inverse SHT); inductor's Triton backend
    # has no mapping for complex dtypes (KeyError: 'complex64'). See the note on
    # IsotropicGaussianRandomFieldS2.forward.
    @torch.compiler.disable
    @override
    def forward(self, update_internal_state=False):
        r"""
        Transform the current spectral state to a noise field on the grid.

        Parameters
        ----------
        update_internal_state : bool, optional
            If ``True``, advance the process by one step *after* producing the
            output, so the next call returns the following step of the
            trajectory. By default ``False``, leaving the caller to drive the
            process via :meth:`update`.

        Returns
        -------
        torch.Tensor
            Real-valued noise of shape
            ``(batch, num_time_steps, num_channels, nlat_local, nlon_local)``.
        """

        # combine channels and time:
        # see IsotropicGaussianRandomFieldS2.forward for why we avoid view_as_complex
        cstate = torch.complex(self.state[..., 0], self.state[..., 1])
        batch_size = cstate.shape[0]

        # flatten history
        cstate = cstate.reshape(batch_size, self.num_time_steps * self.num_channels, self.lmax_local, self.mmax_local)

        # transform
        with amp.autocast(device_type=cstate.device.type, enabled=False):
            eta = self.isht(cstate)

        # expand history
        eta = eta.reshape(batch_size, self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local)

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return eta


class DummyNoiseS2(BaseNoiseS2):
    r"""
    Dummy noise module for testing and debugging. This noise is stateless.

    The module always emits a tensor with the correct output shape
    ``(B, T, C, H, W)`` but carries no stochastic signal beyond what the chosen
    mode specifies. Unlike the real noise classes it stores its state in
    *spatial* rather than spectral layout and never runs an SHT, so it is cheap
    enough to use in tests that only care about shapes and control flow.

    Supported modes:

    ``constant_zero`` (default)
        Always emits an all-zero tensor. Useful for verifying shape consistency
        of the noise pipeline without introducing any stochastic signal. In
        particular, when the preprocessor is configured in ``"perturb"`` mode
        the noise is *added* to the model input channels. Returning zeros
        guarantees that the input is not modified, so integration tests can
        check shapes and control-flow correctness without having to account for
        random perturbations.

    ``constant_random``
        Draws a Gaussian tensor once per :meth:`update` call and holds it fixed
        until the next one. Useful for verifying that the model handles a
        non-zero, reproducible noise pattern correctly without the overhead of
        the spherical harmonic transform used by the real noise classes.

    Parameters
    ----------
    img_shape : (int, int)
        Number of latitudinal and longitudinal modes.
    batch_size : int
        Batch size for the noise.
    num_channels : int
        Number of channels for the noise.
    num_time_steps : int, optional
        Number of time steps, by default ``1``.
    mode : str, default 'constant_zero'
        Output mode. One of ``'constant_zero'`` or ``'constant_random'``.
    seed : int, default 333
        Random seed used in ``'constant_random'`` mode; ignored otherwise.
    **kwargs
        Ignored; present so noise configs can pass extra keys.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the two supported values.
    """

    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps=1,
        mode="constant_zero",
        seed=333,
        **kwargs,
    ):

        if mode not in ("constant_zero", "constant_random"):
            raise ValueError(f"DummyNoiseS2: unknown mode '{mode}'. " f"Expected 'constant_zero' or 'constant_random'.")

        self.mode = mode

        # BaseNoiseS2.__init__ sets up nlat/nlon, comm splits, rng_cpu/rng_gpu, and
        # allocates the state buffer via _ensure_state. The shape is picked up through
        # our override of _state_shape_suffix below, which gives a spatial (H, W)
        # layout instead of the spectral (L, M, 2) default.
        super().__init__(
            img_shape=img_shape,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=num_time_steps,
            seed=seed,
        )

    @property
    def _state_shape_suffix(self):
        # spatial (H, W) rather than spectral (L, M, 2)
        return (self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local)

    @override
    def is_stateful(self):
        r"""
        Whether the noise carries state across calls.

        Returns
        -------
        bool
            Always ``False``: the stored tensor is held fixed between updates
            rather than evolved.
        """
        return False

    def extra_repr(self):
        r"""
        Extra fields shown in the module's ``repr``.

        Returns
        -------
        str
            The base class fields plus ``mode``.
        """
        return super().extra_repr() + f", mode={self.mode}"

    @override
    def update(self, replace_state=False, batch_size=None):
        r"""
        Refresh the stored tensor according to the selected mode.

        Parameters
        ----------
        replace_state : bool, optional
            Accepted for interface compatibility with the stateful noise
            classes; ignored here, since the state is always overwritten.
        batch_size : int, optional
            If given, resize the state buffer to this batch size first.
        """
        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            newstate = torch.empty_like(self.state)

            if self.mode == "constant_zero":
                newstate.zero_()
            else:  # constant_random
                if self.state.is_cuda:
                    newstate.normal_(mean=0.0, std=1.0, generator=self.rng_gpu)
                else:
                    newstate.normal_(mean=0.0, std=1.0, generator=self.rng_cpu)

        self._set_state(newstate)

        return

    @override
    def forward(self, update_internal_state=False):
        r"""
        Return the stored tensor, no transform involved.

        Parameters
        ----------
        update_internal_state : bool, optional
            If ``True``, refresh the stored tensor *after* returning the current
            one, by default ``False``.

        Returns
        -------
        torch.Tensor
            Tensor of shape
            ``(batch, num_time_steps, num_channels, nlat_local, nlon_local)``:
            all zeros, or a fixed Gaussian draw, depending on ``mode``. This is
            the live state buffer, not a copy.
        """

        state = self.state

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return state


@torch.compiler.disable
def run_eager(fn, *args, **kwargs):
    """Run a callable outside torch.compile (forces a graph break).

    The noise field is complex-valued (torch.complex + inverse SHT) and inductor's Triton
    backend cannot codegen complex dtypes (KeyError: 'complex64'). A method-level
    @torch.compiler.disable on the noise module's forward is NOT honored when dynamo inlines
    the nn.Module call (it traces straight into it), whereas a disable on a plain function call
    like this one is respected — so we break at the call site instead.
    """
    return fn(*args, **kwargs)


def noise_seed_reflect(centered: bool, seed_offset: int = 0):
    """Derive the per-rank base seed and reflection flag for a noise source.

    Mirrors the seeding used for the input noise so that every ensemble member gets an
    independent realization while model-parallel ranks stay consistent. ``seed_offset``
    lets independent noise sources (e.g. input noise vs. stochastic physics) draw from
    decorrelated streams while keeping the same per-member structure.

    centered=False: each ensemble member is fully independent.
    centered=True: antithetic pairing -- ranks (0,1), (2,3), ... share a seed and differ
    only by a sign flip (variance reduction for the ensemble estimator).
    """
    if not centered:
        seed = 333 + seed_offset + comm.get_rank("model") + comm.get_size("model") * comm.get_rank("data")
        reflect = False
    else:
        ensemble_eff_rank = comm.get_rank("ensemble") // 2
        reflect = comm.get_rank("ensemble") % 2 == 0
        seed = (
            333
            + seed_offset
            + comm.get_rank("model")
            + comm.get_size("model") * ensemble_eff_rank
            + comm.get_size("model") * comm.get_size("ensemble") * comm.get_rank("batch")
        )
    return seed, reflect


def build_noise(
    noise_params, *, img_shape, batch_size, num_channels, num_time_steps, grid_type, seed, reflect, default_lambd=1.0
):
    """Factory that constructs a noise module from a config dict.

    Centralizes the type dispatch so both the input-noise path and the stochastic-physics
    module share a single construction routine. ``num_channels`` is passed explicitly (the
    input-noise "perturb" mode resolves it from the perturbed channel list, so it is not read
    from ``noise_params`` here). ``default_lambd`` supplies the temporal correlation default
    (typically dt*dhours/6h) since it depends on the dataset cadence.
    """
    ntype = noise_params.get("type", None)
    if ntype is None:
        raise ValueError("Error, please specify a noise type")

    lmax = noise_params.get("lmax", None)

    if ntype == "diffusion":
        return DiffusionNoiseS2(
            img_shape=img_shape,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=num_time_steps,
            sigma=noise_params.get("sigma", 1.0),
            kT=noise_params.get("kT", 0.5 * (100 / 6370) ** 2),
            lambd=noise_params.get("lambd", default_lambd),
            grid_type=grid_type,
            lmax=lmax,
            seed=seed,
            reflect=reflect,
            learnable=noise_params.get("learnable", False),
        )
    elif ntype == "white":
        return IsotropicGaussianRandomFieldS2(
            img_shape=img_shape,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=num_time_steps,
            sigma=noise_params.get("sigma", 1.0),
            alpha=noise_params.get("alpha", 0.0),
            grid_type=grid_type,
            lmax=lmax,
            seed=seed,
            reflect=reflect,
            learnable=noise_params.get("learnable", False),
        )
    elif ntype == "dummy":
        return DummyNoiseS2(
            img_shape=img_shape,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=num_time_steps,
        )
    else:
        raise NotImplementedError(f"Error, noise type {ntype} not supported.")
