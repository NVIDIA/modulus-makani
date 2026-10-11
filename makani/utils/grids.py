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

import torch.amp as amp

import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch_harmonics.quadrature import legendre_gauss_weights, clenshaw_curtiss_weights, precompute_latitudes


from makani.utils import comm
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim
from makani.mpu.mappings import reduce_from_parallel_region


_SERIAL_TRANSFORMS = {
    "sht": th.RealSHT,
    "isht": th.InverseRealSHT,
    "vsht": th.RealVectorSHT,
    "ivsht": th.InverseRealVectorSHT,
}

_DISTRIBUTED_TRANSFORMS = {
    "sht": thd.DistributedRealSHT,
    "isht": thd.DistributedInverseRealSHT,
    "vsht": thd.DistributedRealVectorSHT,
    "ivsht": thd.DistributedInverseRealVectorSHT,
}


def get_spectral_transform(kind, nlat, nlon, grid_type, lmax=None, mmax=None, distributed=False):
    r"""
    Construct a (vector) spherical harmonic transform from a grid name and resolution.

    The one place in makani that calls torch-harmonics transform constructors.
    It accepts both constructor conventions: the ``(nlat, nlon, grid=...)``
    signature of torch-harmonics 0.9.x and the grid-descriptor signature of
    1.0, which is used whenever :func:`torch_harmonics.as_grid` exists.
    Callers that use the distributed transforms must have initialized
    ``torch_harmonics.distributed`` themselves.

    Parameters
    ----------
    kind : str
        ``"sht"``, ``"isht"``, ``"vsht"`` or ``"ivsht"``.
    nlat, nlon : int
        Grid resolution.
    grid_type : str
        Grid name, e.g. ``"equiangular"`` or ``"legendre-gauss"``.
    lmax, mmax : int, optional
        Cutoffs (exclusive). ``None`` leaves the choice to torch-harmonics.
    distributed : bool, optional
        Use the spatially distributed transform, by default ``False``.

    Returns
    -------
    torch.nn.Module
        The transform.

    Raises
    ------
    ValueError
        If an explicitly requested cutoff is not the one the transform ends up
        with, which would mean torch-harmonics silently truncated it.
    """
    handle = (_DISTRIBUTED_TRANSFORMS if distributed else _SERIAL_TRANSFORMS)[kind]

    if hasattr(th, "as_grid"):
        transform = handle(th.as_grid(grid_type, nlat=nlat, nlon=nlon), lmax=lmax, mmax=mmax)
    else:
        transform = handle(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid_type)

    if ((lmax is not None) and (transform.lmax != lmax)) or ((mmax is not None) and (transform.mmax != mmax)):
        raise ValueError(
            f"{kind} on a {nlat}x{nlon} {grid_type} grid was built with (lmax, mmax)=({transform.lmax}, {transform.mmax}) "
            f"instead of the requested ({lmax}, {mmax})"
        )

    return transform


def grid_to_quadrature_rule(grid_type):

    grid_to_quad_dict = {
        "euclidean": "uniform",
        "equiangular": "naive",
        "legendre-gauss": "legendre-gauss",
        "clenshaw-curtiss": "clenshaw-curtiss",
        "weatherbench2": "weatherbench2",
    }

    if grid_type not in grid_to_quad_dict.keys():
        raise NotImplementedError(f"Grid type {grid_type} does not have a quadrature rule")
    else:
        return grid_to_quad_dict[grid_type]


def compute_spherical_bandlimit(img_shape, grid_type):

    if grid_type == "equiangular":
        lmax = (img_shape[0] - 1) // 2
        mmax = img_shape[1] // 2
        return min(lmax, mmax)
    elif grid_type == "legendre-gauss":
        lmax = img_shape[0] - 1
        mmax = img_shape[1] // 2
        return min(lmax, mmax)
    else:
        raise NotImplementedError(f"Unknown type {grid_type} not implemented")


class GridConverter(torch.nn.Module):
    def __init__(self, src_grid, dst_grid, lat_rad, lon_rad):
        super().__init__()
        self.src = src_grid
        self.dst = dst_grid
        self.src_lat = lat_rad
        self.src_lon = lon_rad

        if self.src != self.dst:
            if self.dst == "legendre-gauss":
                cost_lg, _ = legendre_gauss_weights(lat_rad.shape[0], -1, 1)
                tq = torch.arccos(cost_lg) - torch.pi / 2.0
                self.dst_lat = tq.to(lat_rad.device)
                self.dst_lon = lon_rad

                # compute indices
                permutation = torch.arange(lat_rad.shape[0] - 1, -1, -1).to(torch.long).to(lat_rad.device)
                jj = torch.searchsorted(lat_rad, self.dst_lat, sorter=permutation) - 1
                self.indices = jj[permutation]

                # compute weights
                self.interp_weights = (
                    (self.dst_lat - lat_rad[self.indices]) / torch.diff(lat_rad)[self.indices]
                ).reshape(-1, 1)
            else:
                raise NotImplementedError(f"Error, destination grid type {self.dst} not implemented.")
        else:
            self.dst_lat = self.src_lat
            self.dst_lon = self.src_lon

    def get_src_coords(self):
        return self.src_lat, self.src_lon

    def get_dst_coords(self):
        return self.dst_lat, self.dst_lon

    def forward(self, data):
        if self.src == self.dst:
            return data
        else:
            return torch.lerp(
                data[..., self.indices, :], data[..., self.indices + 1, :], self.interp_weights.to(dtype=data.dtype)
            )


class GridQuadrature(torch.nn.Module):
    def __init__(
        self, quadrature_rule, img_shape, crop_shape=None, crop_offset=(0, 0), normalize=False, distributed=False
    ):
        super().__init__()

        self.distributed = comm.is_distributed("spatial") and distributed
        crop_shape = img_shape if crop_shape is None else crop_shape

        if quadrature_rule == "naive":
            jacobian = torch.clamp(torch.sin(torch.linspace(0, torch.pi, img_shape[0])), min=0.0)
            dtheta = torch.pi / img_shape[0]
            dlambda = 2 * torch.pi / img_shape[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
            # numerical precision can be an issue here, make sure it sums to 4pi:
            quad_weight = quad_weight * (4.0 * torch.pi) / torch.sum(quad_weight)
        elif quadrature_rule == "clenshaw-curtiss":
            cost, weights = clenshaw_curtiss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == "legendre-gauss":
            cost, weights = legendre_gauss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == "weatherbench2":
            # compute the 'integrated area weight' that weatherbench2 uses
            lats = torch.linspace(0, torch.pi, img_shape[0])
            cell_bounds = torch.cat([torch.as_tensor([0]), (lats[:-1] + lats[1:]) / 2, torch.as_tensor([torch.pi])])
            jacobian = torch.cos(cell_bounds[:-1]) - torch.cos(cell_bounds[1:])
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * jacobian.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == "uniform":
            quad_weight = torch.ones((img_shape[0],)).unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
            # normalize to 4pi
            quad_weight = 4.0 * torch.pi * quad_weight / torch.sum(quad_weight)
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # if distributed, make sure to split correctly across ranks:
        # in case of model parallelism, we need to make sure that we use the correct shapes per rank
        # for h
        if self.distributed and (comm.get_size("h") > 1):
            shapes_h = compute_split_shapes(crop_shape[0], comm.get_size("h"))
            local_shape_h = shapes_h[comm.get_rank("h")]
            local_offset_h = crop_offset[0] + sum(shapes_h[: comm.get_rank("h")])
        else:
            local_shape_h = crop_shape[0]
            local_offset_h = crop_offset[0]

        # for w
        if self.distributed and (comm.get_size("w") > 1):
            shapes_w = compute_split_shapes(crop_shape[1], comm.get_size("w"))
            local_shape_w = shapes_w[comm.get_rank("w")]
            local_offset_w = crop_offset[1] + sum(shapes_w[: comm.get_rank("w")])
        else:
            local_shape_w = crop_shape[1]
            local_offset_w = crop_offset[1]

        # crop globally if requested
        if crop_shape is not None:
            quad_weight = quad_weight[
                local_offset_h : local_offset_h + local_shape_h, local_offset_w : local_offset_w + local_shape_w
            ]

        # make it contiguous
        quad_weight = quad_weight.contiguous()

        # reshape
        H, W = quad_weight.shape
        quad_weight = quad_weight.reshape(1, 1, H, W)

        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # integrate over last two axes only:
        quad = torch.sum(x * self.quad_weight.to(x.dtype), dim=(-2, -1))
        if self.distributed and (comm.get_size("spatial") > 1):
            quad = reduce_from_parallel_region(quad.contiguous(), "spatial")

        return quad


class BandLimitMask(torch.nn.Module):
    def __init__(self, img_shape, grid_type, lmax=None, type="sht"):
        super().__init__()
        self.img_shape = img_shape
        self.grid_type = grid_type
        self.lmax = lmax if lmax is not None else compute_spherical_bandlimit(img_shape, grid_type)
        self.type = type

        if self.type == "sht":
            # SHT for the computation of SH coefficients
            if comm.get_size("spatial") > 1:
                from torch_harmonics.distributed import DistributedRealSHT, DistributedInverseRealSHT
                import torch_harmonics.distributed as thd

                if not thd.is_initialized():
                    polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                    azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                    thd.init(polar_group, azimuth_group)
                self.forward_transform = DistributedRealSHT(*img_shape, lmax=lmax, mmax=lmax, grid=grid_type).float()
                self.inverse_transform = DistributedInverseRealSHT(
                    *img_shape, lmax=lmax, mmax=lmax, grid=grid_type
                ).float()
            else:
                from torch_harmonics import RealSHT, InverseRealSHT

                self.forward_transform = RealSHT(*img_shape, lmax=lmax, mmax=lmax, grid=grid_type).float()
                self.inverse_transform = InverseRealSHT(*img_shape, lmax=lmax, mmax=lmax, grid=grid_type).float()

        elif self.type == "fft":

            # get the cutoff frequency in m for each latitude
            lats, _ = precompute_latitudes(self.img_shape[0], grid=self.grid_type)
            # get the grid spacing at the equator
            delta_equator = 2 * torch.pi / (self.lmax - 1)
            mlim = torch.ceil(2 * torch.pi * torch.sin(lats) / delta_equator).reshape(self.img_shape[0], 1)
            ms = torch.arange(self.lmax).reshape(1, -1)
            mask = ms <= mlim
            mask = split_tensor_along_dim(mask, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
            mask = split_tensor_along_dim(mask, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
            self.register_buffer("mask", mask, persistent=False)

            if comm.get_size("spatial") > 1:
                from makani.mpu.fft import DistributedRealFFT1, DistributedInverseRealFFT1

                self.forward_transform = DistributedRealFFT1(img_shape[1], lmax=lmax, mmax=lmax).float()
                self.inverse_transform = DistributedInverseRealFFT1(img_shape[1], lmax=lmax, mmax=lmax).float()
            else:
                from makani.models.common.fft import RealFFT1, InverseRealFFT1

                self.forward_transform = RealFFT1(img_shape[1], lmax=lmax, mmax=lmax).float()
                self.inverse_transform = InverseRealFFT1(img_shape[1], lmax=lmax, mmax=lmax).float()
        else:
            raise ValueError(f"Unknown truncation type {self.type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with amp.autocast(device_type="cuda", enabled=False):
            dtype = x.dtype
            x = x.float()

            x = self.forward_transform(x)

            if hasattr(self, "mask"):
                x = torch.where(self.mask, x, torch.zeros_like(x))

            x = self.inverse_transform(x)

            x = x.to(dtype=dtype)

        return x
