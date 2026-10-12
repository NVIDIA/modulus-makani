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

import re
from collections import OrderedDict


from makani.utils.invariants import check_legacy_invariant_options, parse_invariant_features

# Auxiliary channels are the inputs the model consumes but does not predict. They are
# named with a prefix that says how they are laid out in the input: dynamic ones are
# appended at every history step, static ones once after the whole history.
DYNAMIC_AUX_PREFIX = "xd_"
STATIC_AUX_PREFIX = "xs_"


def is_dynamic_aux_channel(channel_name):
    return channel_name.startswith(DYNAMIC_AUX_PREFIX)


def is_static_aux_channel(channel_name):
    return channel_name.startswith(STATIC_AUX_PREFIX)


def get_auxiliary_channels(
    add_zenith=False,
    n_noise_chan=0,
    add_grid=False,
    gridtype="sinusoidal",
    grid_num_frequencies=1,
    add_cos_to_grid=True,
    invariants=None,
    add_copernicus_emb=False,
    **kwargs,
):
    """
    Return the names of the auxiliary channels, in the order in which they are appended to the input.

    This is the single account of the auxiliary channels: the driver derives the channel counts
    from it, and it has to match what the preprocessor appends. The keyword names are those of the
    run parameters, so that the parameters can be passed as ``**params.to_dict()``.
    """
    check_legacy_invariant_options(kwargs)

    channel_names = []

    # dynamic, in the order the preprocessor appends them at every step
    if add_zenith:
        channel_names.append(f"{DYNAMIC_AUX_PREFIX}zen")

    if n_noise_chan > 0:
        for c in range(n_noise_chan):
            channel_names.append(f"{DYNAMIC_AUX_PREFIX}noise{c}")

    # static, in the order of get_static_features
    if add_grid:
        if gridtype == "sinusoidal":
            for f in range(1, grid_num_frequencies + 1):
                channel_names += [f"{STATIC_AUX_PREFIX}sinlat{f}", f"{STATIC_AUX_PREFIX}sinlon{f}"]
                if add_cos_to_grid:
                    channel_names += [f"{STATIC_AUX_PREFIX}coslat{f}", f"{STATIC_AUX_PREFIX}coslon{f}"]
        else:
            channel_names += [f"{STATIC_AUX_PREFIX}lat", f"{STATIC_AUX_PREFIX}lon"]

    for feature in parse_invariant_features(invariants):
        if feature.encoding == "onehot":
            channel_names += [f"{STATIC_AUX_PREFIX}{feature.channel}{k}" for k in range(feature.num_classes)]
        else:
            channel_names.append(f"{STATIC_AUX_PREFIX}{feature.channel}")

    if add_copernicus_emb:
        channel_names += [f"{STATIC_AUX_PREFIX}cop{i}" for i in range(8)]

    return channel_names


def get_water_channels(channel_names):
    """
    Helper routine to extract water channels from channel names
    """

    water_chans = []
    for c, ch in enumerate(channel_names):
        if ch[0] in {"q", "r"} or ch == "tcwv":
            water_chans.append(c)

    return water_chans


def get_wind_channels(channel_names):
    """
    Helper routine to extract water channels from channel names
    """

    wind_chans = []
    for c, ch in enumerate(channel_names):
        if ch[0] == "u" and ("v" + ch[1:]) in channel_names:
            vc = channel_names.index("v" + ch[1:])
            wind_chans = wind_chans + [c, vc]

    return wind_chans


def split_channel_name(channel_name: str):
    """Split a makani channel name into its variable prefix and pressure level.

    This is the single definition of how channel names are classified. Every
    reader that has to decide whether a channel is atmospheric or a surface
    field goes through here, so that the classification cannot drift apart
    between data sources.

    Parameters
    ----------
    channel_name : str
        Channel name such as ``"z500"`` or ``"u10m"``.

    Returns
    -------
    prefix : str
        Variable prefix, e.g. ``"z"``. Equals ``channel_name`` for surface
        channels. Prefixes longer than three characters are supported
        (``"clwc500"`` -> ``("clwc", 500)``): the pattern only requires that the
        digits are preceded by letters, the prefix itself is everything before
        them.
    level : int or None
        Pressure level in hPa, or ``None`` for surface channels.

    Notes
    -----
    ``"d2"`` (2 metre dewpoint) is the one surface name that would otherwise
    parse as variable ``"d"`` on 2 hPa, so it is excluded explicitly.
    """
    match = re.search(r"[0-9]{1,4}$", channel_name)
    if (re.search(r"[a-z]{1,3}[0-9]{1,4}$", channel_name) is not None) and (channel_name != "d2"):
        return channel_name[: match.start()], int(match.group())
    return channel_name, None


def get_channel_groups(channel_names, aux_channel_names=[]):
    """
    Helper routine to extract indices of atmospheric, surface and auxiliary variables and group them into their respective groups.
    The resulting numbering does NOT respect history.
    """

    atmo_groups = OrderedDict()
    atmo_chans = []
    surf_chans = []
    dyn_aux_chans = []
    stat_aux_chans = []

    # parse channel names and group variables by pressure level/surface variables
    for idx, chn in enumerate(channel_names):
        # check if pattern matches an atmospheric variable
        _, pressure_level = split_channel_name(chn)
        if pressure_level is not None:
            if pressure_level not in atmo_groups.keys():
                atmo_groups[pressure_level] = []
            atmo_groups[pressure_level].append(idx)
        else:
            surf_chans.append(idx)

    # check the correctness of the groups (they should all come in the same order and same number of vars)
    n_atmo_chans = None
    for plvl, idx in atmo_groups.items():
        if n_atmo_chans is None:
            n_atmo_chans = len(idx)
        else:
            if n_atmo_chans != len(idx):
                raise ValueError(
                    f"expected all atmospheric pressure level groups to have the same number of channels ({n_atmo_chans}), but got {len(idx)}"
                )

        atmo_chans += idx

    # append the auxiliary variable to the surface channels
    for idx, chn in enumerate(aux_channel_names):
        if is_static_aux_channel(chn):
            stat_aux_chans.append(idx + len(channel_names))
        elif is_dynamic_aux_channel(chn):
            dyn_aux_chans.append(idx + len(channel_names))
        else:
            raise ValueError(
                f"Auxiliary channel '{chn}' is neither dynamic ('{DYNAMIC_AUX_PREFIX}') nor static ('{STATIC_AUX_PREFIX}')."
            )

    return atmo_chans, surf_chans, dyn_aux_chans, stat_aux_chans, atmo_groups.keys()
