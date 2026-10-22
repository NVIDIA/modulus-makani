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

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# zeroes the imaginary parts of the DC and Nyquist bins before the inverse, which cuFFT does not do
from torch_harmonics.fft import irfft


class RealFFT1(nn.Module):
    r"""
    Forward real-valued FFT along the last dimension, with mode truncation.

    Wraps :func:`torch.fft.rfft` behind the same module interface the SHT
    classes use, so a spectral layer can swap a planar transform for a spherical
    one without changing its call sites. The transform is applied to the last
    dimension and the output is truncated to ``mmax`` modes.

    Parameters
    ----------
    nlon : int
        Number of grid points along the transformed dimension.
    lmax : int, optional
        Upper bound on the retained mode count, clamped to ``nlon // 2 + 1``.
        Defaults to the full ``nlon // 2 + 1``.
    mmax : int, optional
        Number of modes actually kept, clamped to ``lmax``. Defaults to ``lmax``.

    See Also
    --------
    InverseRealFFT1 : the corresponding inverse transform.
    """

    def __init__(self, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft

        self.nlon = nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.lmax)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Transform to spectral space and truncate.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input of shape ``(..., nlon)``.
        norm : str, optional
            Normalization mode forwarded to :func:`torch.fft.rfft`, by default
            ``"ortho"`` (unitary, so forward and inverse are adjoint).

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(..., mmax)``.
        """
        y = self.fft_handle(x, n=self.nlon, dim=-1, norm=norm)

        # mode truncation
        y = y[..., : self.mmax].contiguous()

        return y


class InverseRealFFT1(nn.Module):
    r"""
    Inverse real-valued FFT along the last dimension.

    Counterpart to :class:`RealFFT1`. Truncated inputs are zero-padded back up
    to ``nlon // 2 + 1`` modes implicitly by :func:`torch.fft.irfft`, so a
    signal round-tripped through ``RealFFT1`` and this module is band-limited
    rather than exactly recovered whenever ``mmax`` truncates.

    Parameters
    ----------
    nlon : int
        Number of grid points along the reconstructed dimension.
    lmax : int, optional
        Upper bound on the mode count, clamped to ``nlon // 2 + 1``. Defaults to
        the full ``nlon // 2 + 1``.
    mmax : int, optional
        Number of input modes expected, clamped to ``lmax``. Defaults to ``lmax``.

    See Also
    --------
    RealFFT1 : the corresponding forward transform.
    """

    def __init__(self, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.ifft_handle = irfft

        self.nlon = nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.lmax)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Transform back to grid space.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(..., m)`` with ``m <= nlon // 2 + 1``.
            Missing high modes are zero-padded.
        norm : str, optional
            Normalization mode forwarded to :func:`torch.fft.irfft`, by default
            ``"ortho"``. Must match the mode used in the forward transform.

        Returns
        -------
        torch.Tensor
            Real-valued signal of shape ``(..., nlon)``.
        """
        # implicit padding
        y = self.ifft_handle(x, n=self.nlon, dim=-1, norm=norm)

        return y


class RealFFT2(nn.Module):
    r"""
    Forward real-valued 2D FFT with mode truncation, mirroring the SHT interface.

    Drop-in planar replacement for a forward spherical harmonic transform: the
    last two dimensions are treated as latitude and longitude, and the result is
    truncated to ``lmax`` latitudinal and ``mmax`` longitudinal modes.

    Truncation along the latitude axis is two-sided. The FFT along a
    non-halved axis places positive frequencies at the front and negative
    frequencies at the back, so keeping the ``lmax`` lowest frequencies means
    concatenating the leading ``ceil(lmax/2)`` and trailing ``floor(lmax/2)``
    entries; a plain ``[:lmax]`` slice would discard all negative frequencies
    instead. The longitude axis needs no such care because ``rfft2`` already
    returns only non-negative frequencies there. Truncation is skipped entirely
    when ``lmax`` and ``mmax`` are at their maxima.

    Parameters
    ----------
    nlat : int
        Number of grid points along the second-to-last (latitude) dimension.
    nlon : int
        Number of grid points along the last (longitude) dimension.
    lmax : int, optional
        Retained latitudinal modes, clamped to ``nlat``. Defaults to ``nlat``.
    mmax : int, optional
        Retained longitudinal modes, clamped to ``nlon // 2 + 1``. Defaults to
        ``nlon // 2 + 1``.

    See Also
    --------
    InverseRealFFT2 : the corresponding inverse transform.
    """

    def __init__(self, nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Transform to spectral space and truncate both spectral axes.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input of shape ``(..., nlat, nlon)``.
        norm : str, optional
            Normalization mode forwarded to :func:`torch.fft.rfft2`, by default
            ``"ortho"``.

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(..., lmax, mmax)``.
        """
        y = self.fft_handle(x, s=(self.nlat, self.nlon), dim=(-2, -1), norm=norm)

        if self.truncate:
            y = torch.cat((y[..., : self.lmax_high, : self.mmax], y[..., -self.lmax_low :, : self.mmax]), dim=-2)

        return y


class InverseRealFFT2(nn.Module):
    r"""
    Inverse real-valued 2D FFT, mirroring the inverse SHT interface.

    Counterpart to :class:`RealFFT2`. Truncated coefficients are zero-padded
    back to the full grid before the inverse transform. The padding is inserted
    *between* the retained positive and negative latitudinal frequencies (not
    appended at the end), which is where the missing high frequencies belong in
    FFT layout.

    Parameters
    ----------
    nlat : int
        Number of grid points along the second-to-last (latitude) dimension.
    nlon : int
        Number of grid points along the last (longitude) dimension.
    lmax : int, optional
        Latitudinal modes expected on input, clamped to ``nlat``. Defaults to ``nlat``.
    mmax : int, optional
        Longitudinal modes expected on input, clamped to ``nlon // 2 + 1``.
        Defaults to ``nlon // 2 + 1``.

    See Also
    --------
    RealFFT2 : the corresponding forward transform.
    """

    def __init__(self, nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here: the inverse in latitude first, then the real inverse in longitude, which
        # needs the m=0 column to be Hermitian. Truncation to an even lmax breaks that, and backends
        # differ in how they treat the remaining imaginary parts
        self.ifft_handle = irfft

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Zero-pad the truncated spectrum and transform back to grid space.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(..., lmax, m)`` with ``m >= mmax``;
            anything beyond ``mmax`` is dropped.
        norm : str, optional
            Normalization mode forwarded to the inverse FFTs, by default
            ``"ortho"``. Must match the mode used in the forward transform.

        Returns
        -------
        torch.Tensor
            Real-valued field of shape ``(..., nlat, nlon)``.
        """
        # truncation is implicit but better do it manually
        xt = x[..., : self.mmax]

        if self.truncate:
            # pad
            xth = xt[..., : self.lmax_high, :]
            xtl = xt[..., -self.lmax_low :, :]
            xthp = F.pad(xth, (0, 0, 0, self.nlat - self.lmax))
            xt = torch.cat([xthp, xtl], dim=-2)

        out = torch.fft.ifft(xt, n=self.nlat, dim=-2, norm=norm)
        out = self.ifft_handle(out, n=self.nlon, dim=-1, norm=norm)

        return out


class RealFFT3(nn.Module):
    r"""
    Forward real-valued 3D FFT with truncation on all three spectral axes.

    Three-dimensional analogue of :class:`RealFFT2`, used by models that treat a
    vertical (or temporal) axis spectrally alongside the two horizontal ones.
    The transform runs over the last three dimensions ``(d, h, w)``. As in the
    2D case, truncation along ``d`` and ``h`` is two-sided (leading and trailing
    frequency blocks are concatenated) while ``w`` is a simple head slice,
    because ``rfftn`` halves only the last axis. Normalization is fixed to
    ``"ortho"``.

    Parameters
    ----------
    nd : int
        Number of grid points along the third-to-last dimension.
    nh : int
        Number of grid points along the second-to-last dimension.
    nw : int
        Number of grid points along the last dimension.
    ldmax : int, optional
        Retained modes along ``d``, clamped to ``nd``. Defaults to ``nd``.
    lhmax : int, optional
        Retained modes along ``h``, clamped to ``nh``. Defaults to ``nh``.
    lwmax : int, optional
        Retained modes along ``w``, clamped to ``nw // 2 + 1``. Defaults to
        ``nw // 2 + 1``.

    See Also
    --------
    InverseRealFFT3 : the corresponding inverse transform.
    """

    def __init__(self, nd, nh, nw, ldmax=None, lhmax=None, lwmax=None):
        super().__init__()

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)

        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):
        r"""
        Transform to spectral space and truncate all three spectral axes.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input of shape ``(..., nd, nh, nw)``.

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(..., ldmax, lhmax, lwmax)``.
        """
        x = torch.fft.rfftn(x, s=(self.nd, self.nh, self.nw), dim=(-3, -2, -1), norm="ortho")

        # truncate in w
        x = x[..., : self.lwmax]

        # truncate in h
        x = torch.cat([x[..., : self.lhmax_high, :], x[..., -self.lhmax_low :, :]], dim=-2)

        # truncate in d
        x = torch.cat([x[..., : self.ldmax_high, :, :], x[..., -self.ldmax_low :, :, :]], dim=-3)

        return x


class InverseRealFFT3(nn.Module):
    r"""
    Inverse real-valued 3D FFT, counterpart to :class:`RealFFT3`.

    Zero-pads the ``d`` and ``h`` spectral axes back to full length (inserting
    the padding between the retained positive and negative frequency blocks)
    and lets the real inverse in ``w`` pad the halved ``w`` axis implicitly.
    Padding is skipped on any axis that was not truncated. Normalization is
    fixed to ``"ortho"``.

    Parameters
    ----------
    nd : int
        Number of grid points along the third-to-last dimension.
    nh : int
        Number of grid points along the second-to-last dimension.
    nw : int
        Number of grid points along the last dimension.
    ldmax : int, optional
        Modes expected along ``d``, clamped to ``nd``. Defaults to ``nd``.
    lhmax : int, optional
        Modes expected along ``h``, clamped to ``nh``. Defaults to ``nh``.
    lwmax : int, optional
        Modes expected along ``w``, clamped to ``nw // 2 + 1``. Defaults to
        ``nw // 2 + 1``.

    See Also
    --------
    RealFFT3 : the corresponding forward transform.
    """

    def __init__(self, nd, nh, nw, ldmax=None, lhmax=None, lwmax=None):
        super().__init__()

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)

        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):
        r"""
        Zero-pad the truncated spectrum and transform back to grid space.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(..., ldmax, lhmax, lwmax)``.

        Returns
        -------
        torch.Tensor
            Real-valued field of shape ``(..., nd, nh, nw)``.
        """

        # pad in d
        if self.ldmax < self.nd:
            # pad
            xh = x[..., : self.ldmax_high, :, :]
            xl = x[..., -self.ldmax_low :, :, :]
            xhp = F.pad(xh, (0, 0, 0, 0, 0, self.nd - self.ldmax))
            x = torch.cat([xhp, xl], dim=-3)

        # pad in h
        if self.lhmax < self.nh:
            # pad
            xh = x[..., : self.lhmax_high, :]
            xl = x[..., -self.lhmax_low :, :]
            xhp = F.pad(xh, (0, 0, 0, self.nh - self.lhmax))
            x = torch.cat([xhp, xl], dim=-2)

        # the complex inverse in d and h first, then the real inverse in w, as in InverseRealFFT2
        x = torch.fft.ifft2(x, s=(self.nd, self.nh), dim=(-3, -2), norm="ortho")
        x = irfft(x, n=self.nw, dim=-1, norm="ortho")

        return x
