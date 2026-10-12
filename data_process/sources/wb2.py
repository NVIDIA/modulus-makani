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

"""WeatherBench2 / ARCO-ERA5 source, read from a zarr store such as the public ones on GCS."""

from typing import Dict, List, Optional
import numpy as np
import h5py as h5
import datetime as dt
import warnings
import xarray as xr

from makani.utils.dataloaders.wb2_helpers import (
    split_convert_channel_names,
    gcs_storage_options,
    invariant_variables,
    invariant_wb2_name,
)
from data_process.sources import InvariantSource, Source, Unit


def _open_aligned(input_file: str, lat: List[float], lon: List[float], coord_mode: str, verbose: bool) -> xr.Dataset:
    """Open a zarr store and align it to the metadata grid according to ``coord_mode``."""
    # storage options only for remote stores: zarr 3 rejects them, even empty, for a local path
    open_options = {}
    if input_file.startswith(("gs://", "gcs://")):
        open_options["storage_options"] = gcs_storage_options()
    wb2_data = xr.open_dataset(input_file, engine="zarr", **open_options)
    # some WB2 zarrs store atmospheric/surface fields as (..., longitude, latitude);
    # the readers assume (..., latitude, longitude).
    wb2_data = wb2_data.transpose(..., "latitude", "longitude")
    if coord_mode == "match":
        wb2_data = wb2_data.sel(latitude=lat, longitude=lon)
    elif coord_mode == "force-flip-lat":
        if verbose:
            warnings.warn("coord_mode='force-flip-lat': flipping latitude axis without coordinate matching")
        wb2_data = wb2_data.isel(latitude=slice(None, None, -1))
    elif coord_mode == "force":
        if verbose:
            warnings.warn("coord_mode='force': reading data as-is, assuming ordering matches metadata")
    else:
        raise ValueError(f"Unknown coord_mode: {coord_mode}. Must be one of: match, force-flip-lat, force")
    return wb2_data


class Wb2Source(Source):
    """Read ERA5 from a WeatherBench2 style zarr store, such as ARCO-ERA5.

    Work is split over individual timestamps and filled in batches of
    ``batch_size`` samples.

    Parameters
    ----------
    metadata : Dict
        Dataset metadata, see :class:`data_process.sources.Source`.
    comm_rank : int
        MPI rank, used to restrict informational output to rank 0.
    input_file : str
        Path or GCS specifier of the zarr store.
    batch_size : int
        Batch size in which the samples are processed. This is merely a performance setting. Bigger batches are more
        efficient but require more memory.
    coord_mode: str
        How to align input lat/lon to the metadata grid. One of:
        - "match": use xarray .sel() to reorder input data to match metadata coords (default).
        - "force-flip-lat": flip the latitude axis of the input data without coordinate matching.
        - "force": read data as-is, assume ordering already matches metadata.
    skip_missing_channels: bool
        Setting this flag to True will skip missing channels instead of failing.
    impute_missing_timestamps: bool
        Setting this flag to True will impute missing timestamps instead of failing.
    """

    def __init__(
        self,
        metadata: Dict,
        comm_rank: int,
        input_file: str,
        batch_size: Optional[int] = 32,
        coord_mode: Optional[str] = "match",
        skip_missing_channels: Optional[bool] = False,
        impute_missing_timestamps: Optional[bool] = False,
    ):
        super().__init__(metadata, comm_rank)
        # a nonpositive batch would never fill anything, leaving the files written but empty
        if batch_size <= 0:
            raise ValueError(f"batch_size has to be positive, got {batch_size}.")
        self.units_per_fill = batch_size
        self.impute_missing_timestamps = impute_missing_timestamps

        # split in surface and atmospheric channels
        (
            self.atmospheric_channel_names,
            self.atmospheric_channel_names_wb2,
            self.surface_channel_names,
            self.surface_channel_names_wb2,
            self.atmospheric_levels,
        ) = split_convert_channel_names(self.channel_names)

        wb2_data = _open_aligned(input_file, self.lat, self.lon, coord_mode, verbose=(comm_rank == 0))
        self.wb2_data = wb2_data

        # variables absent from the store, settled up front so that the
        # converter can write their channels as missing
        wanted = list(dict.fromkeys(self.surface_channel_names_wb2 + self.atmospheric_channel_names_wb2))
        self.missing_wb2 = [name for name in wanted if name not in wb2_data]
        if self.missing_wb2:
            if not skip_missing_channels:
                raise IndexError(f"Keys {self.missing_wb2} not found in dataset.")
            if comm_rank == 0:
                print(f"Keys {self.missing_wb2} not found in dataset, skipping")

    def skipped_channel_indices(self) -> List[int]:
        skipped = [
            self.channel_names.index(sc)
            for sc, scwb2 in zip(self.surface_channel_names, self.surface_channel_names_wb2)
            if scwb2 in self.missing_wb2
        ]
        for ac, acwb2 in zip(self.atmospheric_channel_names, self.atmospheric_channel_names_wb2):
            if acwb2 in self.missing_wb2:
                skipped += [self.channel_names.index(ac + str(alevel)) for alevel in self.atmospheric_levels]
        return skipped

    def fill(self, out: h5.File, entry_key: str, units: List[Unit]):
        samples = [sample for unit in units for sample in unit]
        # the converter hands out contiguous runs of samples
        tstart = samples[0][0]
        tend = tstart + len(samples)
        # naive UTC, which is how the zarr time coordinate is stored
        timebatch = [np.datetime64(t.astimezone(dt.timezone.utc).replace(tzinfo=None)) for _, t in samples]

        # surface channel variables
        for sc, scwb2 in zip(self.surface_channel_names, self.surface_channel_names_wb2):
            cidx = self.channel_names.index(sc)
            if scwb2 in self.missing_wb2:
                continue
            wb2_sel = self.wb2_data[scwb2]
            data = wb2_sel[wb2_sel["time"].isin(timebatch)].values

            # checks:
            if data.shape[0] != len(timebatch):
                if not self.impute_missing_timestamps:
                    raise IndexError(f"Dates {timebatch} not all found in dataset for {scwb2}.")
                data = self._impute(out, wb2_sel, scwb2, timebatch, tstart, cidx)

            out[entry_key][tstart:tend, cidx, ...] = data[...]

        # atmospheric level variables: one bulk read per variable per batch,
        # then subset levels in numpy. WB2 atmospheric vars are chunked
        # (time, level=all, H, W), so per-level .sel(level=...) inside the
        # loop made each level refetch the same chunk from storage.
        for ac, acwb2 in zip(self.atmospheric_channel_names, self.atmospheric_channel_names_wb2):
            if acwb2 in self.missing_wb2:
                continue

            wb2_sel_all = self.wb2_data[acwb2].sel(level=list(self.atmospheric_levels))
            bulk = wb2_sel_all[wb2_sel_all["time"].isin(timebatch)].values  # (T, L, H, W)

            for idl, alevel in enumerate(self.atmospheric_levels):
                cidx = self.channel_names.index(ac + str(alevel))
                if bulk.shape[0] == len(timebatch):
                    data = bulk[:, idl]
                else:
                    if not self.impute_missing_timestamps:
                        raise IndexError(f"Dates {timebatch} not all found in dataset for {acwb2}.")
                    data = self._impute(out, self.wb2_data[acwb2].sel(level=alevel), acwb2, timebatch, tstart, cidx)

                out[entry_key][tstart:tend, cidx, ...] = data[...]

    def _impute(self, out, wb2_sel, name_wb2, timebatch, tstart, cidx) -> np.ndarray:
        """Read ``timebatch`` one timestamp at a time, filling the missing ones with NaN and marking them invalid."""
        data = np.empty((len(timebatch), len(self.lat), len(self.lon)), dtype=np.float32)
        for tid, t in enumerate(timebatch):
            if t not in wb2_sel["time"]:
                print(f"Imputing timestamp {t} for {name_wb2}")
                data[tid, ...] = np.nan
                out["valid_data"][tstart + tid, cidx] = 0
            else:
                data[tid, ...] = wb2_sel[wb2_sel["time"].isin([t])].values[...]
        return data

    def summary(self) -> Optional[str]:
        return f"Skipped channels: {self.missing_wb2}"


class Wb2InvariantSource(InvariantSource):
    """Read the time invariant ERA5 fields from a WeatherBench2 style zarr store.

    WB2 stores the invariants without a time axis, while ARCO-ERA5 repeats them
    along it; the latter are read at the first timestep.

    Parameters
    ----------
    metadata : Dict
        Invariant metadata, see :class:`data_process.sources.InvariantSource`.
    input_file : str
        Path or GCS specifier of the zarr store.
    coord_mode: str
        How to align input lat/lon to the metadata grid, see :class:`Wb2Source`.
    skip_missing_channels: bool
        Setting this flag to True will skip channels that are unknown or absent
        from the store instead of failing.
    """

    def __init__(
        self,
        metadata: Dict,
        input_file: str,
        coord_mode: Optional[str] = "match",
        skip_missing_channels: Optional[bool] = False,
    ):
        super().__init__(metadata)
        self.wb2_data = _open_aligned(input_file, self.lat, self.lon, coord_mode, verbose=True)

        # unknown names and names absent from the store are both missing
        self.missing = [
            name
            for name in self.channel_names
            if name not in invariant_variables or invariant_wb2_name(name) not in self.wb2_data
        ]
        if self.missing:
            if not skip_missing_channels:
                raise IndexError(f"Invariants {self.missing} not found in dataset.")
            print(f"Invariants {self.missing} not found in dataset, skipping")

    def skipped_channel_indices(self) -> List[int]:
        return [self.channel_names.index(name) for name in self.missing]

    def fill(self, out: h5.File, entry_key: str):
        for cidx, name in enumerate(self.channel_names):
            if name in self.missing:
                continue
            field = self.wb2_data[invariant_wb2_name(name)]
            if "time" in field.dims:
                field = field.isel(time=0)
            out[entry_key][cidx, ...] = field.values

    def summary(self) -> Optional[str]:
        return f"Skipped channels: {self.missing}" if self.missing else None
