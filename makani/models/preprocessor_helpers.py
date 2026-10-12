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

import os
from functools import partial
import numpy as np
import torch

from makani.utils.invariants import check_legacy_invariant_options, parse_invariant_features, read_invariants


def get_bias_correction(params):
    r"""
    Load and shard the bias correction field, if one is configured.

    The returned tensor is already restricted to this rank's local domain and
    subsampled to match the model grid, so the caller can subtract it directly
    without further slicing.

    Parameters
    ----------
    params : ParamsBase
        Configuration. ``bias_correction`` gives the path to the correction
        file; the ``img_local_*`` keys and ``subsampling_factor`` determine the
        local shard.

    Returns
    -------
    torch.Tensor or None
        Bias correction of shape ``(1, C, H_local, W_local)``, or ``None`` if
        no ``bias_correction`` path is configured.

    Raises
    ------
    IOError
        If ``bias_correction`` is set but does not point at an existing file.
    """
    if params.get("bias_correction", None) is not None:
        from makani.utils.auxiliary_fields import get_bias_correction

        # set up sharding parameters
        start_x = params.get("img_local_offset_x", 0)
        end_x = min(start_x + params.get("img_local_shape_x", params.img_shape_x), params.img_shape_x)
        start_y = params.get("img_local_offset_y", 0)
        end_y = min(start_y + params.get("img_local_shape_y", params.img_shape_y), params.img_shape_y)

        bc_path = params.get("bias_correction", None)
        if not os.path.isfile(bc_path):
            raise IOError(f"Specify a valid bias correction path, got {bc_path}")

        bias = torch.as_tensor(get_bias_correction(bc_path, params.get("out_channels")), dtype=torch.float32)

        # shard the bias correction
        subsampling_factor = params.get("subsampling_factor", 1)
        bias = bias[:, :, start_x:end_x:subsampling_factor, start_y:end_y:subsampling_factor]

    else:
        bias = None

    return bias


def _synthesize_invariant(params, path):
    """Whether to stand in for an invariant field instead of reading it from disk.

    Only when running on synthetic data *and* the file is not there: a synthetic-data run with
    the real invariants mounted keeps using them, and a run on real data still fails loudly on a
    missing path, since a model trained without its orography is a silently different model.
    """
    return bool(params.get("enable_synthetic_data", False)) and ((path is None) or (not os.path.isfile(path)))


def _synthetic_invariant(params, num_channels: int, one_hot: bool = False) -> torch.Tensor:
    """A stand-in invariant field of the right shape, for synthetic-data runs.

    The channel count is passed in from the configuration, the same count ``get_auxiliary_channels``
    commits the model input to.

    The values are the normalized form directly -- zeros for a continuous field, a one-hot vector
    on the first class for a categorical one -- so the caller skips normalization, which for a
    constant field would divide by a zero standard deviation.
    """
    field = torch.zeros((1, num_channels, params.img_shape_x, params.img_shape_y), dtype=torch.float32)

    if one_hot:
        field[:, 0, :, :] = 1.0

    return field


def _read_invariant_features(params, path, features, normalizer) -> torch.Tensor:
    """Read and encode the configured invariants, see :mod:`makani.utils.invariants`.

    Returns the full grid as ``(1, n_channels, H, W)``; sharding is up to the caller.
    """
    if (path is None) or (not os.path.isfile(path)):
        raise IOError(f"Specify a valid invariants path, got {path}")

    fields, lat, lon = read_invariants(path, [f.channel for f in features])

    if fields.shape[1:] != (params.img_shape_x, params.img_shape_y):
        raise ValueError(
            f"Invariants in {path} are on a {fields.shape[1]}x{fields.shape[2]} grid, "
            f"but the data is {params.img_shape_x}x{params.img_shape_y}."
        )
    # a latitude flip would otherwise go unnoticed
    if hasattr(params, "lat") and not np.allclose(lat, np.asarray(params.lat)):
        raise ValueError(f"Latitudes of the invariants in {path} differ from those of the data.")

    encoded = []
    for feature, field in zip(features, torch.as_tensor(fields)):
        field = field.reshape(1, 1, *field.shape)

        if feature.encoding == "normalize":
            eps = 1e-6
            if normalizer is not None:
                field = normalizer(field, eps=eps)
            else:
                field = (field - torch.mean(field)) / (torch.std(field) + eps)

        elif feature.encoding == "onehot":
            classes = torch.floor(field) if feature.rounding == "floor" else torch.round(field)
            classes = classes.to(torch.long)
            if classes.min() < 0 or classes.max() >= feature.num_classes:
                raise ValueError(
                    f"Invariant '{feature.channel}' has classes in [{classes.min()}, {classes.max()}], "
                    f"outside of the configured [0, {feature.num_classes})."
                )
            # one hot encode and move channels to front
            field = torch.nn.functional.one_hot(classes[0, 0], num_classes=feature.num_classes)
            field = torch.permute(field, (2, 0, 1)).unsqueeze(0).to(torch.float32)

        encoded.append(field)

    return torch.cat(encoded, dim=1)


def get_static_features(params):
    r"""
    Assemble the time-invariant feature channels for the model input.

    Collects whichever static fields the configuration asks for -- grid
    coordinates, the invariants selected from the invariants file, Copernicus embeddings --
    into a single tensor, sharded to this rank's local domain and subsampled to
    the model grid. These give the network the geographic context that the
    prognostic variables alone do not carry.

    Parameters
    ----------
    params : ParamsBase
        Configuration. ``add_grid``, ``invariants`` (read from
        ``invariants_path``, see :mod:`makani.utils.invariants`) and
        ``add_copernicus_emb`` select which features are included, in that
        order; ``normalize_static_features`` enables area-weighted
        normalization; the ``img_local_*`` / ``img_crop_*`` keys and
        ``subsampling_factor`` determine the shard.

    Returns
    -------
    torch.Tensor or None
        Static features of shape ``(1, n_static, H_local, W_local)``, or
        ``None`` if no static features are requested.
    """

    # a checkpoint of the per-file invariants would otherwise silently lose them
    check_legacy_invariant_options(params)

    # set up normalizer
    normalize_static_features = params.get("normalize_static_features", False)
    normalizer = None
    if normalize_static_features:
        from makani.utils.grid_types import DEFAULT_GRID_TYPE
        from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

        # params restored from a checkpoint written before the grid type became
        # required may not carry one; a fresh run always does
        quadrature_rule = grid_to_quadrature_rule(params.get("data_grid_type", DEFAULT_GRID_TYPE))
        crop_shape = [
            params.get("img_crop_shape_x", params.img_shape_x),
            params.get("img_crop_shape_y", params.img_shape_y),
        ]
        crop_offset = [params.get("img_crop_offset_x", 0), params.get("img_crop_offset_y", 0)]

        quadrature = GridQuadrature(
            quadrature_rule,
            img_shape=params.img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=True,
            distributed=False,
        )

        def normalize(tensor, eps=0.0):
            mean = quadrature(tensor).reshape(1, -1, 1, 1)
            std = torch.sqrt(quadrature(torch.square(tensor - mean)).reshape(1, -1, 1, 1))
            tensor = (tensor - mean) / (std + eps)
            return tensor

        normalizer = partial(normalize, eps=0.0)

    # set up sharding parameters
    start_x = params.get("img_local_offset_x", 0)
    end_x = min(start_x + params.get("img_local_shape_x", params.img_shape_x), params.img_shape_x)
    start_y = params.get("img_local_offset_y", 0)
    end_y = min(start_y + params.get("img_local_shape_y", params.img_shape_y), params.img_shape_y)
    subsampling_factor = params.get("subsampling_factor", 1)

    static_features = None
    if params.get("add_grid", False):
        with torch.no_grad():
            if hasattr(params, "lat") and hasattr(params, "lon"):
                from makani.utils.grids import GridConverter

                lat = torch.as_tensor(params.lat).to(torch.float32)
                lon = torch.as_tensor(params.lon).to(torch.float32)

                # convert grid if required
                gconv = GridConverter(
                    params.data_grid_type, params.model_grid_type, torch.deg2rad(lat), torch.deg2rad(lon)
                )
                tx, ty = gconv.get_dst_coords()
                tx = tx.to(torch.float32)
                ty = ty.to(torch.float32)
            else:
                tx = torch.linspace(0, 1, params.img_shape_x + 1, dtype=torch.float32)[0:-1]
                ty = torch.linspace(0, 1, params.img_shape_y + 1, dtype=torch.float32)[0:-1]

            x_grid, y_grid = torch.meshgrid(tx, ty, indexing="ij")
            x_grid, y_grid = x_grid.unsqueeze(0).unsqueeze(0), y_grid.unsqueeze(0).unsqueeze(0)
            grid = torch.cat([x_grid, y_grid], dim=1)

            # transform if requested
            gridtype = params.get("gridtype", "sinusoidal")
            if gridtype == "sinusoidal":
                num_freq = params.get("grid_num_frequencies", 1)

                add_cos = params.get("add_cos_to_grid", True)
                singrid = None
                for freq in range(1, num_freq + 1):
                    if singrid is None:
                        if add_cos:
                            singrid = [torch.sin(grid), torch.cos(grid)]
                        else:
                            singrid = [torch.sin(grid)]
                    else:
                        if add_cos:
                            singrid = singrid + [torch.sin(freq * grid), torch.cos(freq * grid)]
                        else:
                            singrid = singrid + [torch.sin(freq * grid)]

                static_features = torch.cat(singrid, dim=-3)
            else:
                static_features = grid

            # normalize if requested
            if normalizer is not None:
                static_features = normalizer(static_features)

            # shard spatially
            static_features = static_features[:, :, start_x:end_x:subsampling_factor, start_y:end_y:subsampling_factor]

    invariant_features = parse_invariant_features(params.get("invariants", None))
    if invariant_features:
        invariants_path = params.get("invariants_path", None)

        with torch.no_grad():
            if _synthesize_invariant(params, invariants_path):
                inv = torch.cat(
                    [
                        _synthetic_invariant(params, num_channels=f.num_channels, one_hot=(f.encoding == "onehot"))
                        for f in invariant_features
                    ],
                    dim=1,
                )
            else:
                inv = _read_invariant_features(params, invariants_path, invariant_features, normalizer)

            # shard
            inv = inv[:, :, start_x:end_x:subsampling_factor, start_y:end_y:subsampling_factor]

            if static_features is None:
                static_features = inv
            else:
                static_features = torch.cat([static_features, inv], dim=1)

    if params.get("add_copernicus_emb", False):
        copernicus_emb_path = params.get("copernicus_emb_path", None)

        with torch.no_grad():
            if _synthesize_invariant(params, copernicus_emb_path):
                # eight embedding channels, per get_auxiliary_channels
                emb = _synthetic_invariant(params, num_channels=8)
            else:
                from makani.utils.auxiliary_fields import get_copernicus_emb

                if not os.path.isfile(copernicus_emb_path):
                    raise IOError(f"Specify a valid copernicus embedding path, got {copernicus_emb_path}")

                emb = get_copernicus_emb(copernicus_emb_path)

                # one hot encode and move channels to front:
                emb = torch.permute(emb, (2, 0, 1))
                emb = torch.reshape(emb, (1, emb.shape[0], emb.shape[1], emb.shape[2]))

                # no normalization since the data is already in the right format
                eps = 1e-6
                if normalizer is not None:
                    emb = normalizer(emb, eps=eps)
                else:
                    emb = (emb - torch.mean(emb)) / (torch.std(emb) + eps)

            # shard
            emb = emb[:, :, start_x:end_x:subsampling_factor, start_y:end_y:subsampling_factor]

            if static_features is None:
                static_features = emb
            else:
                static_features = torch.cat([static_features, emb], dim=1)

    return static_features
