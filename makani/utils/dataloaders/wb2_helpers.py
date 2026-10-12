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

"""Helpers for reading WeatherBench2 style zarr stores.

WeatherBench2 republishes ERA5 (and model output) as cloud hosted zarr, with one
array per variable and a shared ``level`` coordinate. Two things differ from
every other source makani reads, and this module exists to absorb both:

* **Names are spelled out in full.** Where makani and ERA5 say ``z``, ``t2m``,
  ``tp``, WB2 says ``geopotential``, ``2m_temperature``,
  ``total_precipitation_6hr``. The tables below are the dictionary between the
  two vocabularies, in both directions -- makani -> WB2 for reading a store or
  writing one, WB2 -> makani for interpreting one.
* **Levels are an axis, not part of the name.** A makani channel list names each
  level separately (``z500``, ``z850``), while a WB2 store has a single
  ``geopotential`` array indexed by position along ``level``. Translating a
  channel therefore yields a *(variable, level index)* pair, not just a name,
  and the index depends on the level coordinate of the particular store. That is
  what :func:`build_wb2_channel_map` produces.

Note the ``total_precipitation_6hr`` entry: the accumulation window is part of
the WB2 variable name, so a store fixes it and the reader cannot choose it. That
is recorded as ``accumulation="window"``, to distinguish it from sources that
hand out running totals or instantaneous rates.

Reach for :func:`surface_wb2_name` / :func:`atmospheric_wb2_name` rather than
indexing the tables, so their layout stays an implementation detail here.
"""

from typing import Dict, NamedTuple, Optional

# channel name classification lives in one place for all data sources
from makani.utils.features import get_channel_groups, split_channel_name


# ---------------------------------------------------------------------------
# ERA5 short name <-> WeatherBench2 long name mappings
# ---------------------------------------------------------------------------


class Wb2Variable(NamedTuple):
    """A WeatherBench2 variable backing a makani channel.

    Attributes
    ----------
    name : str
        WB2 long name, i.e. the variable name in the zarr store.
    kind : str
        ``"pl"``, ``"sfc"``, ``"accum"`` or ``"invariant"``.
    units : str, optional
        Units the store is expected to carry.
    accumulation : str
        ``"none"`` for instantaneous fields, ``"window"`` for totals already
        accumulated over a fixed window that is baked into the variable name.
    """

    name: str
    kind: str
    units: Optional[str] = None
    accumulation: str = "none"


surface_variables: Dict[str, Wb2Variable] = {
    "u10m": Wb2Variable("10m_u_component_of_wind", "sfc", units="m s-1"),
    "v10m": Wb2Variable("10m_v_component_of_wind", "sfc", units="m s-1"),
    "t2m": Wb2Variable("2m_temperature", "sfc", units="K"),
    "d2": Wb2Variable("2m_dewpoint_temperature", "sfc", units="K"),
    "u100m": Wb2Variable("100m_u_component_of_wind", "sfc", units="m s-1"),
    "v100m": Wb2Variable("100m_v_component_of_wind", "sfc", units="m s-1"),
    # the accumulation window is part of the WB2 name rather than something the
    # reader chooses, hence "window" rather than "since_start" or "rate"
    "tp": Wb2Variable("total_precipitation_6hr", "accum", units="m", accumulation="window"),
    "sp": Wb2Variable("surface_pressure", "sfc", units="Pa"),
    "msl": Wb2Variable("mean_sea_level_pressure", "sfc", units="Pa"),
    "tcwv": Wb2Variable("total_column_water_vapour", "sfc", units="kg m-2"),
    "sst": Wb2Variable("sea_surface_temperature", "sfc", units="K"),
}

atmospheric_variables: Dict[str, Wb2Variable] = {
    "z": Wb2Variable("geopotential", "pl", units="m2 s-2"),
    "u": Wb2Variable("u_component_of_wind", "pl", units="m s-1"),
    "v": Wb2Variable("v_component_of_wind", "pl", units="m s-1"),
    "t": Wb2Variable("temperature", "pl", units="K"),
    "r": Wb2Variable("relative_humidity", "pl", units="%"),
    "q": Wb2Variable("specific_humidity", "pl", units="kg kg-1"),
}

# time invariant fields, keyed by ECMWF short name like their NCAR counterparts.
# WB2 stores them without a time axis, ARCO-ERA5 repeats them along it.
invariant_variables: Dict[str, Wb2Variable] = {
    "z": Wb2Variable("geopotential_at_surface", "invariant", units="m2 s-2"),
    "lsm": Wb2Variable("land_sea_mask", "invariant"),
    "slt": Wb2Variable("soil_type", "invariant"),
    "cl": Wb2Variable("lake_cover", "invariant"),
    "dl": Wb2Variable("lake_depth", "invariant", units="m"),
    "cvl": Wb2Variable("low_vegetation_cover", "invariant"),
    "cvh": Wb2Variable("high_vegetation_cover", "invariant"),
    "tvl": Wb2Variable("type_of_low_vegetation", "invariant"),
    "tvh": Wb2Variable("type_of_high_vegetation", "invariant"),
    "sdor": Wb2Variable("standard_deviation_of_orography", "invariant", units="m"),
    "isor": Wb2Variable("anisotropy_of_sub_gridscale_orography", "invariant"),
    "anor": Wb2Variable("angle_of_sub_gridscale_orography", "invariant", units="radians"),
    "slor": Wb2Variable("slope_of_sub_gridscale_orography", "invariant"),
    "sdfor": Wb2Variable("standard_deviation_of_filtered_subgrid_orography", "invariant", units="m"),
}

# reverse lookups
surface_variables_inv = {v.name: k for k, v in surface_variables.items()}
atmospheric_variables_inv = {v.name: k for k, v in atmospheric_variables.items()}


# ---------------------------------------------------------------------------
# Channel name helpers
# ---------------------------------------------------------------------------


def split_convert_channel_names(makani_channel_names):
    """Split ERA5/makani channel names into surface and atmospheric groups with WB2 names.

    Returns
    -------
    atmospheric_channel_names : list[str]
        ERA5 short prefixes for atmospheric variables (e.g. ["z", "u"]).
    atmospheric_channel_names_wb2 : list[str]
        Corresponding WB2 long names (e.g. ["geopotential", "u_component_of_wind"]).
    surface_channel_names : list[str]
        ERA5 short names for surface variables (e.g. ["t2m", "u10m"]).
    surface_channel_names_wb2 : list[str]
        Corresponding WB2 long names.
    atmospheric_levels : list[int]
        Sorted list of distinct pressure levels found in the channel list.
    """
    atmospheric_channel_indices, surface_channel_indices, _, _, atmospheric_levels = get_channel_groups(
        makani_channel_names
    )

    atmospheric_channel_names = sorted(
        list(set(split_channel_name(makani_channel_names[k])[0] for k in atmospheric_channel_indices))
    )
    atmospheric_channel_names_wb2 = [atmospheric_wb2_name(c) for c in atmospheric_channel_names]

    surface_channel_names = sorted([makani_channel_names[k] for k in surface_channel_indices])
    surface_channel_names_wb2 = [surface_wb2_name(c) for c in surface_channel_names]

    atmospheric_levels = sorted(list(atmospheric_levels))

    return (
        atmospheric_channel_names,
        atmospheric_channel_names_wb2,
        surface_channel_names,
        surface_channel_names_wb2,
        atmospheric_levels,
    )


def surface_wb2_name(channel_name):
    """Return the WB2 long name of a makani surface channel.

    ``"t2m"`` -> ``"2m_temperature"``. Callers should go through this rather
    than indexing :data:`surface_variables` directly, so that the table layout stays
    an implementation detail of this module and an unknown name produces a
    readable error instead of a bare ``KeyError``.

    Raises
    ------
    ValueError
        If the channel has no WB2 counterpart.
    """
    try:
        return surface_variables[channel_name].name
    except KeyError:
        raise ValueError(f"Unknown surface variable '{channel_name}'. Known names: {list(surface_variables)}") from None


def atmospheric_wb2_name(prefix):
    """Return the WB2 long name of a makani atmospheric variable prefix.

    ``"z"`` -> ``"geopotential"``. The prefix is the channel name without its
    pressure level, as produced by
    :func:`makani.utils.features.split_channel_name`.

    Raises
    ------
    ValueError
        If the prefix has no WB2 counterpart.
    """
    try:
        return atmospheric_variables[prefix].name
    except KeyError:
        raise ValueError(
            f"Unknown atmospheric variable prefix '{prefix}'. Known prefixes: {list(atmospheric_variables)}"
        ) from None


def invariant_wb2_name(channel_name):
    """Return the WB2 long name of a time invariant field.

    ``"lsm"`` -> ``"land_sea_mask"``.

    Raises
    ------
    ValueError
        If the field has no WB2 counterpart.
    """
    try:
        return invariant_variables[channel_name].name
    except KeyError:
        raise ValueError(
            f"Unknown invariant variable '{channel_name}'. Known names: {list(invariant_variables)}"
        ) from None


def build_wb2_channel_map(channel_names, level_values=None):
    """Build a per-channel WB2 conversion table for online zarr reading.

    This is the online equivalent of what the offline conversion scripts do:
    it maps each ERA5/makani channel name to the zarr variable name and, for
    atmospheric variables, the integer index into the store's ``level``
    coordinate array.

    Parameters
    ----------
    channel_names : list[str]
        ERA5/makani channel names in channel-index order (e.g. ["u10m", "z500"]).
    level_values : array-like of int, optional
        Ordered pressure levels available in the zarr store
        (e.g. [50, 100, 150, ..., 1000]).  Required when the channel list
        contains any atmospheric variable.

    Returns
    -------
    list of (zarr_variable_name: str, level_array_idx: int | None)
        One entry per channel.  ``level_array_idx`` is the integer position
        in the store's ``level`` coordinate array, *not* the pressure value.
    """
    level_to_idx = {}
    if level_values is not None:
        level_to_idx = {int(lv): i for i, lv in enumerate(level_values)}

    channel_map = []
    for ch_name in channel_names:
        prefix, pressure = split_channel_name(ch_name)
        if pressure is not None:
            zarr_name = atmospheric_wb2_name(prefix)
            if pressure not in level_to_idx:
                raise ValueError(
                    f"Pressure level {pressure} hPa (channel '{ch_name}') not found in zarr store. "
                    f"Available levels: {sorted(level_to_idx)}"
                )
            channel_map.append((zarr_name, level_to_idx[pressure]))
        else:
            channel_map.append((surface_wb2_name(ch_name), None))

    return channel_map


# ---------------------------------------------------------------------------
# GCS storage helper (useful for online reads from Google Cloud Storage)
# ---------------------------------------------------------------------------


def gcs_storage_options():
    """Return gcsfs storage options, falling back to anonymous access if no ADC found."""
    try:
        import google.auth

        google.auth.default()
        return {}
    except Exception:
        return {"token": "anon"}
