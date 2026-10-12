# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers for reading NSF NCAR ERA5 (RDA dataset d633000) from S3.

The bucket ``s3://nsf-ncar-era5`` hosts ERA5 on its native 0.25 degree
lat/lon grid, which is already identical to the grid makani expects
(latitude 90 -> -90, longitude 0 -> 359.75), stored as netCDF4/HDF5.

Object keys follow::

    <stream>/<YYYYMM>/<stream>.<param>_<short_name>.ll025<grid>.<start>_<end>.nc

with one file per variable per *day* for pressure levels, per *month* for
surface analysis, and per *half month* for the accumulated forecast fields.

Two things make this source awkward, and they are what most of this module is:

* **Object keys have to be constructed, not discovered.** Listing a bucket this
  large per read is not viable, so the key of every file is derived from the
  variable descriptor and the date. The three stream layouts each need their own
  rule, see :func:`analysis_pl_key`, :func:`analysis_sfc_key` and
  :func:`accumulation_key`.
* **Precipitation has to be reassembled.** d633000 ships no total precipitation
  and no ready-made accumulation window: ``tp`` is ``lsp + cp``, summed over an
  hour range that may straddle two forecast runs, since a run reaches forecast
  hour 12 while runs start 12 hours apart. :func:`resolve_accumulation_segments`
  does that arithmetic.

Channel grouping itself is shared with the other readers and lives in
:mod:`makani.utils.dataloaders.channel_helpers`.
"""

import calendar
import datetime as dt
from typing import Dict, List, NamedTuple

# channel grouping and name classification are shared across the data source readers
from makani.utils.dataloaders.channel_helpers import ChannelGroup, build_channel_groups


NCAR_ERA5_BUCKET = "nsf-ncar-era5"

# netCDF "hours since" reference used throughout d633000
NCAR_EPOCH = dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)

# the accumulated forecast stream is initialized twice a day, each run covering
# forecast hours 1..12, so the two runs tile every hour of the day exactly once.
# d633000 stores these already de-accumulated, one value per forecast hour.
ACCUM_INIT_HOURS = (6, 18)
ACCUM_MAX_FORECAST_HOUR = 12


class NcarVariable(NamedTuple):
    """Locator for a single variable within the NCAR ERA5 bucket.

    Attributes
    ----------
    stream : str
        Top level prefix, e.g. ``"e5.oper.an.pl"``.
    param : str
        ECMWF parameter table and code, e.g. ``"128_129"``.
    short_name : str
        ECMWF short name as it appears in the object key, e.g. ``"z"``.
    grid : str
        Grid suffix in the object key: ``"sc"`` for scalars, ``"uv"`` for the
        wind components on pressure levels.
    h5_name : str
        Name of the netCDF variable inside the file, e.g. ``"Z"``. These are
        upper case and irregular: surface fields whose short name starts with a
        digit are prefixed with ``VAR_``.
    """

    stream: str
    param: str
    short_name: str
    grid: str
    h5_name: str


# ---------------------------------------------------------------------------
# ERA5 short name -> NCAR variable mappings
# ---------------------------------------------------------------------------

surface_variables: Dict[str, NcarVariable] = {
    "u10m": NcarVariable("e5.oper.an.sfc", "128_165", "10u", "sc", "VAR_10U"),
    "v10m": NcarVariable("e5.oper.an.sfc", "128_166", "10v", "sc", "VAR_10V"),
    "u100m": NcarVariable("e5.oper.an.sfc", "228_246", "100u", "sc", "VAR_100U"),
    "v100m": NcarVariable("e5.oper.an.sfc", "228_247", "100v", "sc", "VAR_100V"),
    "t2m": NcarVariable("e5.oper.an.sfc", "128_167", "2t", "sc", "VAR_2T"),
    "d2": NcarVariable("e5.oper.an.sfc", "128_168", "2d", "sc", "VAR_2D"),
    "sp": NcarVariable("e5.oper.an.sfc", "128_134", "sp", "sc", "SP"),
    "msl": NcarVariable("e5.oper.an.sfc", "128_151", "msl", "sc", "MSL"),
    "tcwv": NcarVariable("e5.oper.an.sfc", "128_137", "tcwv", "sc", "TCWV"),
    "sst": NcarVariable("e5.oper.an.sfc", "128_034", "sstk", "sc", "SSTK"),
}

atmospheric_variables: Dict[str, NcarVariable] = {
    "z": NcarVariable("e5.oper.an.pl", "128_129", "z", "sc", "Z"),
    "t": NcarVariable("e5.oper.an.pl", "128_130", "t", "sc", "T"),
    "u": NcarVariable("e5.oper.an.pl", "128_131", "u", "uv", "U"),
    "v": NcarVariable("e5.oper.an.pl", "128_132", "v", "uv", "V"),
    "q": NcarVariable("e5.oper.an.pl", "128_133", "q", "sc", "Q"),
    "r": NcarVariable("e5.oper.an.pl", "128_157", "r", "sc", "R"),
}

# Accumulated fields live in the forecast stream and are summed to form the
# makani channel. d633000 does not ship total precipitation directly, so tp is
# reconstructed from its two ERA5 components, tp = lsp + cp (both in metres).
# the nesting matters: this is ONE candidate made of two components that are
# summed, not two alternative sources. See makani.utils.dataloaders.channel_helpers.
accumulated_variables: Dict[str, tuple] = {
    "tp": (
        (
            NcarVariable("e5.oper.fc.sfc.accumu", "128_142", "lsp", "sc", "LSP"),
            NcarVariable("e5.oper.fc.sfc.accumu", "128_143", "cp", "sc", "CP"),
        ),
    ),
}


# Time invariant fields. The stream holds a single timestep per variable, stamped
# 1979-01-01 00Z, and is keyed by ECMWF short name: the surface geopotential is
# "z" here, which cannot be confused with the pressure level channels since the
# invariants are written to a file of their own.
invariant_variables: Dict[str, NcarVariable] = {
    "z": NcarVariable("e5.oper.invariant", "128_129", "z", "sc", "Z"),
    "lsm": NcarVariable("e5.oper.invariant", "128_172", "lsm", "sc", "LSM"),
    "slt": NcarVariable("e5.oper.invariant", "128_043", "slt", "sc", "SLT"),
    "cl": NcarVariable("e5.oper.invariant", "128_026", "cl", "sc", "CL"),
    "dl": NcarVariable("e5.oper.invariant", "228_007", "dl", "sc", "DL"),
    "cvl": NcarVariable("e5.oper.invariant", "128_027", "cvl", "sc", "CVL"),
    "cvh": NcarVariable("e5.oper.invariant", "128_028", "cvh", "sc", "CVH"),
    "tvl": NcarVariable("e5.oper.invariant", "128_029", "tvl", "sc", "TVL"),
    "tvh": NcarVariable("e5.oper.invariant", "128_030", "tvh", "sc", "TVH"),
    "sdor": NcarVariable("e5.oper.invariant", "128_160", "sdor", "sc", "SDOR"),
    "isor": NcarVariable("e5.oper.invariant", "128_161", "isor", "sc", "ISOR"),
    "anor": NcarVariable("e5.oper.invariant", "128_162", "anor", "sc", "ANOR"),
    "slor": NcarVariable("e5.oper.invariant", "128_163", "slor", "sc", "SLOR"),
    "sdfor": NcarVariable("e5.oper.invariant", "128_074", "sdfor", "sc", "SDFOR"),
}


def build_ncar_channel_groups(channel_names: List[str], skip_missing_channels: bool = False) -> List[ChannelGroup]:
    """Group makani channel names by the NCAR source file that provides them.

    Grouping matters for throughput: the pressure level files are chunked as
    ``(1, n_levels, nlat, nlon)``, so every level of a variable arrives in the
    same chunk and all levels of a variable should be filled from a single read.

    NCAR names every variable canonically, so there is nothing to resolve
    against the file, and ``available`` is left unset.

    Parameters
    ----------
    channel_names : list of str
        Channel names in channel index order, i.e. ``coords["channel"]`` from
        the dataset metadata.
    skip_missing_channels : bool, optional
        If True, channels with no known NCAR counterpart are dropped instead of
        raising.

    Returns
    -------
    list of ChannelGroup
        One entry per source variable, pressure level groups first. ``tp`` is
        the only group with more than one variable, since it is summed from
        ``lsp`` and ``cp``.

    Raises
    ------
    ValueError
        If a channel has no known NCAR counterpart and ``skip_missing_channels``
        is False.
    """
    return build_channel_groups(
        channel_names,
        atmospheric=atmospheric_variables,
        surface=surface_variables,
        accumulated=accumulated_variables,
        skip_missing_channels=skip_missing_channels,
        source="NCAR",
    )


# ---------------------------------------------------------------------------
# Object key construction
# ---------------------------------------------------------------------------


def analysis_pl_key(variable: NcarVariable, day: dt.date) -> str:
    """Return the object key of the pressure level file covering ``day``.

    Pressure level files hold 24 hours of one variable on all 37 levels.
    """
    stamp = day.strftime("%Y%m%d")
    return f"{variable.stream}/{day:%Y%m}/{variable.stream}.{variable.param}_{variable.short_name}.ll025{variable.grid}.{stamp}00_{stamp}23.nc"


def analysis_sfc_key(variable: NcarVariable, day: dt.date) -> str:
    """Return the object key of the surface analysis file covering ``day``.

    Surface analysis files hold a full calendar month of one variable.
    """
    last = calendar.monthrange(day.year, day.month)[1]
    return f"{variable.stream}/{day:%Y%m}/{variable.stream}.{variable.param}_{variable.short_name}.ll025{variable.grid}.{day:%Y%m}0100_{day:%Y%m}{last:02d}23.nc"


def accumulation_key(variable: NcarVariable, init_time: dt.datetime) -> str:
    """Return the object key of the accumulation file holding ``init_time``.

    The accumulated forecast stream is split into half months. The first file of
    a month covers the runs initialized from the 1st 06Z up to the 15th 18Z, the
    second the runs from the 16th 06Z to the end of the month; file names are
    stamped with the *valid* time bounds, hence the trailing 06Z.
    """
    if init_time.day < 16:
        start, end = f"{init_time:%Y%m}0106", f"{init_time:%Y%m}1606"
    else:
        nxt = (init_time.replace(day=28) + dt.timedelta(days=7)).replace(day=1)
        start, end = f"{init_time:%Y%m}1606", f"{nxt:%Y%m}0106"
    return f"{variable.stream}/{init_time:%Y%m}/{variable.stream}.{variable.param}_{variable.short_name}.ll025{variable.grid}.{start}_{end}.nc"


def invariant_key(variable: NcarVariable) -> str:
    """Return the object key of a time invariant field.

    The invariant stream holds a single file per variable, filed under January
    1979 and stamped with that one timestep.
    """
    return f"{variable.stream}/197901/{variable.stream}.{variable.param}_{variable.short_name}.ll025{variable.grid}.1979010100_1979010100.nc"


# ---------------------------------------------------------------------------
# Accumulation window arithmetic
# ---------------------------------------------------------------------------


def latest_forecast_init(time: dt.datetime) -> dt.datetime:
    """Return the most recent forecast initialization at or before ``time``."""
    for hour in sorted(ACCUM_INIT_HOURS, reverse=True):
        if time.hour >= hour:
            return time.replace(hour=hour, minute=0, second=0, microsecond=0)
    previous = time.date() - dt.timedelta(days=1)
    return dt.datetime(previous.year, previous.month, previous.day, max(ACCUM_INIT_HOURS), tzinfo=dt.timezone.utc)


def resolve_accumulation_segments(valid_time: dt.datetime, window_hours: int):
    """Decompose an accumulation window into per forecast run segments.

    Unlike the ECMWF forecast archive, where accumulated fields are running
    totals from the start of a run, d633000 stores them already de-accumulated:
    forecast hour ``h`` holds the accumulation over the single hour ending at
    ``init_time + h``. The value over a window is therefore a plain sum of the
    hours it spans, ``run[forecast_hour_start:forecast_hour_end]`` in zero based
    indexing.

    A run only reaches forecast hour 12 while runs start 12 hours apart, so a
    window is not always contained in a single run: a 12 hour window ending at
    00Z, for instance, starts at 12Z, which is between the 06Z and 18Z runs. The
    window is therefore walked forward and cut at run boundaries, and the
    resulting pieces sum to the total.

    Parameters
    ----------
    valid_time : datetime.datetime
        Timezone aware UTC time at which the accumulation window ends.
    window_hours : int
        Length of the accumulation window in hours. Must be positive.

    Returns
    -------
    list of (datetime.datetime, int, int)
        One ``(init_time, forecast_hour_start, forecast_hour_end)`` triple per
        contributing run, in chronological order. The contribution of a triple
        is the sum of the hourly values over the half open forecast hour range
        ``[forecast_hour_start, forecast_hour_end)`` in zero based indexing,
        which covers the wall clock interval
        ``(init_time + forecast_hour_start, init_time + forecast_hour_end]``.

    Raises
    ------
    ValueError
        If ``window_hours`` is not positive.
    """
    if window_hours < 1:
        raise ValueError(f"Accumulation window must be at least one hour, got {window_hours}.")

    segments = []
    position = valid_time - dt.timedelta(hours=window_hours)
    while position < valid_time:
        init_time = latest_forecast_init(position)
        segment_end = min(valid_time, init_time + dt.timedelta(hours=ACCUM_MAX_FORECAST_HOUR))
        segments.append(
            (
                init_time,
                int((position - init_time).total_seconds() // 3600),
                int((segment_end - init_time).total_seconds() // 3600),
            )
        )
        position = segment_end

    return segments


def to_ncar_hours(time: dt.datetime) -> int:
    """Convert a UTC datetime to the netCDF time coordinate, hours since 1900-01-01."""
    return int((time - NCAR_EPOCH).total_seconds() // 3600)
