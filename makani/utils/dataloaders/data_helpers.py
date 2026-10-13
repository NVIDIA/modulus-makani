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

import datetime as dt
import numpy as np


# this is a small helper to get the latitude and longitude grid consistently across
# functions
def get_lat_lon_grid(img_shape):
    latitude = np.linspace(90, -90, img_shape[0], endpoint=True)
    longitude = np.linspace(0, 360, img_shape[1], endpoint=False)
    return latitude, longitude


def _synthetic_num_channels(params):
    """Channel count of the (nonexistent) synthetic dataset, as the stats arrays would have it.

    The stats files a real dataset ships cover every channel in the dataset, and callers index
    the result with ``params.out_channels``, so the length has to follow the dataset's channel
    list rather than the model's input count.
    """
    if hasattr(params, "data_channel_names"):
        return len(params.data_channel_names)
    if hasattr(params, "channel_names"):
        return len(params.channel_names)

    return params.N_in_channels


def get_data_normalization(params):

    bias = None
    scale = None

    if getattr(params, "enable_synthetic_data", False):
        # Synthetic data is drawn from a uniform distribution and has no statistics to speak of;
        # the .npy files a real dataset ships do not exist for it, and loading them is what used
        # to make every synthetic-data run die in get_dataloader. Identity normalization keeps
        # the shapes right for every consumer without touching disk, which is the same choice
        # get_climatology already makes for the synthetic case, and matches the zero bias / unit
        # scale that DummyLoader hands out through get_input_normalization.
        num_channels = _synthetic_num_channels(params)
        bias = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
        scale = np.ones((1, num_channels, 1, 1), dtype=np.float32)

        return bias, scale

    if hasattr(params, "normalization"):
        if params.normalization == "minmax":
            mins = np.load(params.min_path)
            maxes = np.load(params.max_path)
            bias = mins
            scale = maxes - mins
        elif params.normalization == "zscore":
            means = np.load(params.global_means_path)
            stds = np.load(params.global_stds_path)
            bias = means
            scale = stds

        elif isinstance(params.normalization, dict):
            mins = np.load(params.min_path)
            maxes = np.load(params.max_path)
            means = np.load(params.global_means_path)
            stds = np.load(params.global_stds_path)

            bias = means
            scale = stds

            for ch in params.normalization.keys():
                c = params.data_channel_names.index(ch)
                if params.normalization[ch] == "minmax":
                    bias[:, c] = mins[:, c]
                    scale[:, c] = maxes[:, c] - mins[:, c]
                elif params.normalization[ch] == "zscore":
                    pass
                else:
                    raise ValueError(f"Unknown normalization mode {params.normalization[ch]}")

    return bias, scale


def get_time_diff_stds(params):

    time_diff_stds = None

    if getattr(params, "enable_synthetic_data", False):
        # same reasoning as in get_data_normalization: a unit scale, so that a loss configured
        # with temp_diff_normalization does not need statistics the synthetic dataset cannot have
        return np.ones((1, _synthetic_num_channels(params), 1, 1), dtype=np.float32)

    if hasattr(params, "time_diff_stds_path"):
        time_diff_stds = np.load(params.time_diff_stds_path)
    else:
        raise ValueError("time_diff_std_path not defined.")

    return time_diff_stds


def get_psd_stats(params):

    psd_means = None
    psd_stds = None

    if hasattr(params, "psd_means_path") and hasattr(params, "psd_stds_path"):
        psd_means = np.load(params.psd_means_path)
        psd_stds = np.load(params.psd_stds_path)

        # filter channels if requested
        if hasattr(params, "out_channels"):
            psd_means = psd_means[..., params.out_channels, :]
            psd_stds = psd_stds[..., params.out_channels, :]
    else:
        raise ValueError("psd_means_path or psd_stds_path not defined.")

    return psd_means, psd_stds


def get_climatology(params, mask_nan=True):
    """
    routine for fetching climatology and normalization factors
    """

    subsampling_factor = params.subsampling_factor

    # compute climatology
    if params.enable_synthetic_data:
        clim = np.zeros([1, params.N_out_channels, params.img_crop_shape_x, params.img_crop_shape_y], dtype=np.float32)
    else:
        # full bias and scale
        bias, scale = get_data_normalization(params)
        bias = bias[:, params.out_channels, ...]
        scale = scale[:, params.out_channels, ...]

        # we need this window
        start_x = params.img_crop_offset_x
        end_x = start_x + params.img_crop_shape_x
        start_y = params.img_crop_offset_y
        end_y = start_y + params.img_crop_shape_y

        # now we crop the time means
        time_means = np.load(params.time_means_path)
        time_means = time_means[..., params.out_channels, start_x:end_x, start_y:end_y]
        clim = (time_means - bias) / scale

        # impute NaN values if requested
        if mask_nan:
            clim = np.where(np.isnan(clim), 0.0, clim)

    # apply subsampling
    if subsampling_factor > 1:
        clim = clim[:, :, ::subsampling_factor, ::subsampling_factor]

    return clim


def get_hydrostatic_balance_climatology(params):
    hydrostatic_balance_means = np.load(params.hydrostatic_balance_means_path)

    return hydrostatic_balance_means


# compute UTC timestamp fom year and hour into year.
def get_timestamp(year, hour):

    # compute timestamp for january 1st in UTC. Important, we need to specify a timezone here
    jan_01_epoch = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)

    # compute offset
    return jan_01_epoch + dt.timedelta(hours=hour)


def as_utc(date):
    """Return ``date`` as a timezone aware UTC datetime.

    A naive datetime is taken to be UTC already and merely labelled as such; an
    aware one is converted. See :func:`get_date_from_string` for why the naive
    case cannot be left to ``astimezone`` or ``timestamp``, both of which would
    assume the local zone of the machine.
    """
    if date.tzinfo is None:
        return date.replace(tzinfo=dt.timezone.utc)

    return date.astimezone(dt.timezone.utc)


def get_date_from_string(isostring):
    """Parse an ISO 8601 date string into a timezone aware UTC datetime.

    A string carrying a designator (``2017-01-01T00:00:00Z``, or an explicit
    offset) is converted to UTC. One without (``2017-01-01T00:00:00``) is taken
    to be UTC already and merely labelled as such.

    ..note::
        The naive case has to be handled explicitly. Since Python 3.6,
        ``astimezone`` on a naive datetime does not raise: it assumes the value
        is in the *local* zone and converts from there, so a date string without
        a designator would silently shift by the offset of whichever machine
        parsed it -- an hour on a CET laptop, nothing on a UTC cluster. That is
        the failure this helper exists to prevent, so the check is on
        ``tzinfo`` rather than on an exception that no longer fires.
    """
    return as_utc(dt.datetime.fromisoformat(isostring))


def get_date_from_timestamp(timestamp):
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)


def get_seconds_from_times(times) -> np.ndarray:
    """Convert times to float seconds: since the epoch for datetimes, as offsets for timedeltas.

    Datetimes go through :func:`as_utc` first, since ``timestamp`` on a naive
    datetime would assume the local zone of the machine.
    """
    return np.asarray(
        [t.total_seconds() if isinstance(t, dt.timedelta) else as_utc(t).timestamp() for t in times], dtype=np.float64
    )


def get_timedelta_from_timestamp(timestamp):
    return dt.timedelta(seconds=timestamp)


def get_date_ranges(dates, lookback_hours: int, lookahead_hours: int):
    time_ranges = [
        ((date - dt.timedelta(hours=lookback_hours)), (date + dt.timedelta(hours=lookahead_hours))) for date in dates
    ]
    return time_ranges


def parse_data_normalization_mode(mode=None):
    """Parse the ``data_normalization_mode`` option of the DALI loader.

    The mode is ``"<mode>[-<axes>]"``, e.g. ``"offline"`` or ``"offline-FHWB"``: ``<mode>``
    is ``offline`` or ``online``, and ``<axes>`` are the DALI axis names normalized over,
    where a ``B`` normalizes over the batch as well.

    Returns
    -------
    norm_mode : str
    norm_channels : str
        The axis names without ``B``, ``"FHW"`` by default.
    norm_batch : bool

    Raises
    ------
    ValueError
        If ``<mode>`` is neither ``offline`` nor ``online``. The loader treats every mode other
        than ``offline`` as online, so a typo would otherwise silently drop the saved statistics.
    """
    if mode is None:
        return "offline", "FHW", False

    split = mode.split("-")
    norm_mode = split[0]
    if norm_mode not in ("offline", "online"):
        raise ValueError(f"Unsupported data_normalization_mode {mode!r}, expected offline or online")
    norm_channels = "FHW"
    norm_batch = False
    if len(split) > 1:
        norm_channels = split[1]
        if "B" in norm_channels:
            norm_batch = True
            norm_channels = norm_channels.replace("B", "")

    return norm_mode, norm_channels, norm_batch
