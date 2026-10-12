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


def soft_clamp(x: torch.Tensor, offset=0.0) -> torch.Tensor:
    r"""
    Smooth one-sided clamp to non-negative values.

    ``0`` for ``x + offset <= 0``, ``(x + offset)^2`` up to ``1/2`` and
    ``x + offset - 1/4`` beyond, which is continuously differentiable and
    becomes the identity far from the clamp. Used to keep water channels
    non-negative at the model output: with ``offset = mean / std`` the clamp
    sits at the physical zero of a z-scored channel. The caller subtracts
    ``offset`` again to stay in normalized units.

    Parameters
    ----------
    x : torch.Tensor
        Values to clamp.
    offset : float or torch.Tensor, optional
        Shift applied before clamping, by default ``0.0``.

    Returns
    -------
    torch.Tensor
        Clamped values, shifted by ``offset``.
    """
    x = x + offset
    y = torch.where(x > 0.0, x**2, torch.zeros_like(x))
    y = torch.where(x >= 0.5, x - 0.25, y)
    return y
from torch import nn


class ComplexReLU(nn.Module):
    r"""
    Complex-valued variants of the ReLU activation function.

    A complex number has no total order, so "rectification" admits several
    inequivalent generalizations. This module implements four of them, selected
    via ``mode``, for a complex input :math:`z = x + i y`:

    * ``"real"`` -- rectify the real part only, leave the imaginary part
      untouched: :math:`\mathrm{ReLU}(x) + i y`.
    * ``"cartesian"`` -- rectify real and imaginary parts independently:
      :math:`\mathrm{ReLU}(x) + i\,\mathrm{ReLU}(y)`.
    * ``"modulus"`` -- rectify the magnitude while preserving the phase,

      .. math::

          z \mapsto \begin{cases}
              (|z| + b)\, \frac{z}{|z|} & \text{if } |z| + b > 0 \\
              0 & \text{otherwise}
          \end{cases}

      where :math:`b` is a learnable bias. This keeps the activation
      equivariant to global phase rotations.
    * ``"halfplane"`` -- pass :math:`z` through unchanged if its phase falls in
      the quarter-plane :math:`[b, b + \pi/2)` and scale it by
      ``negative_slope`` otherwise; here the learnable bias :math:`b` is an
      angle rather than a magnitude offset.

    Parameters
    ----------
    negative_slope : float, optional
        Slope applied to the rejected part of the input, as in
        :class:`torch.nn.LeakyReLU`. ``0.0`` (the default) gives a hard
        rectifier. Used by all modes except ``"modulus"``.
    mode : str, optional
        One of ``"real"`` (default), ``"cartesian"``, ``"modulus"`` or
        ``"halfplane"``. See above.
    bias_shape : tuple of int, optional
        Shape of the learnable bias for the ``"modulus"`` and ``"halfplane"``
        modes. Defaults to a single shared scalar. Ignored by the other modes,
        which use a fixed bias of ``0``.
    scale : float, optional
        Value the learnable bias is initialized to, by default ``1.0``.

    Raises
    ------
    NotImplementedError
        If ``mode`` is not one of the four supported values. Raised on the
        first forward pass rather than at construction time.
    """

    def __init__(self, negative_slope=0.0, mode="real", bias_shape=None, scale=1.0):
        super().__init__()

        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(scale * torch.ones(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(scale * torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the complex rectifier elementwise.

        Parameters
        ----------
        z : torch.Tensor
            Complex-valued input tensor of arbitrary shape. If ``bias_shape``
            was given, it must broadcast against ``z``.

        Returns
        -------
        torch.Tensor
            Complex tensor of the same shape and dtype as ``z``.
        """
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)

        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = torch.where(zabs + self.bias > 0, (zabs + self.bias) * z / zabs, 0.0)
            # out = self.act(zabs - self.bias) * torch.exp(1.j * z.angle())

        elif self.mode == "halfplane":
            # bias is an angle parameter in this case
            modified_angle = torch.angle(z) - self.bias
            condition = torch.logical_and((0.0 <= modified_angle), (modified_angle < torch.pi / 2.0))
            out = torch.where(condition, z, self.negative_slope * z)

        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)

        else:
            raise NotImplementedError

        return out


class ComplexActivation(nn.Module):
    r"""
    Lift an arbitrary real-valued activation to the complex plane.

    Generalizes :class:`ComplexReLU` to any pointwise nonlinearity :math:`\sigma`
    supplied by the caller. For a complex input :math:`z = x + i y`:

    * ``"cartesian"`` -- apply :math:`\sigma` to the real and imaginary parts
      independently: :math:`\sigma(x) + i\,\sigma(y)`.
    * ``"modulus"`` -- apply :math:`\sigma` to the magnitude and re-attach the
      original phase,

      .. math::

          z \mapsto \sigma(|z| + b)\, e^{i \arg z}

      with a learnable bias :math:`b`. Phase-equivariant.
    * anything else -- identity; :math:`z` is returned unchanged and
      ``activation`` is never called.

    Parameters
    ----------
    activation : torch.nn.Module or callable
        Real-valued pointwise nonlinearity. In ``"cartesian"`` mode it is
        applied to a real view of shape ``(..., 2)``, so it must act elementwise.
    mode : str, optional
        One of ``"cartesian"`` (default), ``"modulus"``, or any other string for
        the identity.
    bias_shape : tuple of int, optional
        Shape of the learnable bias in ``"modulus"`` mode; defaults to a single
        shared scalar. In the other modes a zero bias is registered as a
        non-persistent buffer and is not trained.
    """

    def __init__(self, activation, mode="cartesian", bias_shape=None):
        super().__init__()

        # store parameters
        self.mode = mode
        if self.mode == "modulus":
            if bias_shape is not None:
                self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(torch.zeros((1), dtype=torch.float32))
        else:
            bias = torch.zeros((1), dtype=torch.float32)
            self.register_buffer("bias", bias, persistent=False)

        # real valued activation
        self.act = activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the lifted activation elementwise.

        Parameters
        ----------
        z : torch.Tensor
            Complex-valued input tensor of arbitrary shape. If ``bias_shape``
            was given, it must broadcast against ``z``.

        Returns
        -------
        torch.Tensor
            Complex tensor of the same shape and dtype as ``z``.
        """
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)
        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = self.act(zabs + self.bias) * torch.exp(1.0j * z.angle())
        else:
            # identity
            out = z

        return out


class MagnitudePreservingSiLU(nn.Module):
    r"""
    SiLU rescaled to preserve the magnitude of unit-variance activations.

    Plain SiLU shrinks the second moment of its input, so stacking many of them
    makes activation magnitudes drift layer over layer. This variant divides by
    a constant :math:`c` chosen so the output variance is again unity for
    standard normal input:

    .. math::

        \mathrm{SiLU}_{mp}(x) = \frac{1}{c}\, x\, \sigma(x),
        \qquad c = \sqrt{\mathbb{E}_{x \sim \mathcal{N}(0,1)}\bigl[(x \sigma(x))^2\bigr]}
        \approx 0.596

    Keeping magnitudes fixed removes the need for the normalization layers that
    would otherwise be doing this job, which matters for diffusion-style
    backbones trained without them.

    Parameters
    ----------
    normalization_factor : float, optional
        The constant :math:`c` above, by default ``0.596`` (the value for
        standard normal input). Pass a different value if the expected input
        distribution is not unit-variance.

    References
    ----------
    Inspired by https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/
    """

    def __init__(self, normalization_factor=0.596):
        super().__init__()

        # store normalization factor
        self.norminv = 1.0 / normalization_factor
        self.silu = torch.nn.SiLU(inplace=False)

    def forward(self, x):
        r"""
        Apply the rescaled SiLU elementwise.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape and dtype as ``x``.
        """
        return self.norminv * self.silu(x)
