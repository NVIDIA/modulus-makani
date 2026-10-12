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

"""Convert the time invariant ERA5 fields to a single makani style file.

The counterpart of ``convert_era5_to_makani_input.py`` for fields without a time
axis, such as orography, land-sea mask, soil and vegetation type. It reads from
the same archives and takes the same metadata file, from which only the grid is
used, but writes a single file and runs serially: the invariants are a few
megabytes in total, so there is nothing to distribute.
"""

from typing import Callable, Dict, List, Optional
import os
import sys
import json
import time
import functools
import numpy as np
import h5py as h5
import datetime as dt
import argparse as ap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from makani.utils.dataloaders.ncar_helpers import NCAR_ERA5_BUCKET, invariant_variables
from data_process.sources import InvariantSource


def _create_output_file(ofile, entry_key, channel_names, lat, lon) -> h5.File:
    """Create an invariant makani file with its datasets and dimension scales.

    The layout is that of a yearly file without the time axis: ``entry_key`` is
    ``(channel, lat, lon)`` and ``valid_data`` is ``(channel,)``.
    """
    chanlen = max([len(v) for v in channel_names])
    dataset_shape = (len(channel_names), len(lat), len(lon))

    f = h5.File(ofile, "w")
    # NaN as fill value, so that channels which are never written are missing
    # without further ado; unlike the yearly files there is no parallel HDF5
    # here, so prefilling costs nothing worth avoiding
    f.create_dataset(entry_key, dataset_shape, dtype=np.float32, fillvalue=np.nan)

    # create dimension scales
    # datasets
    f.create_dataset("valid_data", data=np.ones((len(channel_names),), dtype=np.int32))
    f.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
    f["channel"][...] = channel_names
    f.create_dataset("lat", data=lat)
    f.create_dataset("lon", data=lon)
    # scales
    f["channel"].make_scale("channel")
    f["lat"].make_scale("lat")
    f["lon"].make_scale("lon")
    # label
    f[entry_key].dims[0].label = "Channel name"
    f[entry_key].dims[1].label = "Latitude in degrees"
    f[entry_key].dims[2].label = "Longitude in degrees"
    # attach
    f[entry_key].dims[0].attach_scale(f["channel"])
    f[entry_key].dims[1].attach_scale(f["lat"])
    f[entry_key].dims[2].attach_scale(f["lon"])

    return f


def _write_missing(out, entry_key, channel_indices):
    """Write NaN and clear ``valid_data`` for ``channel_indices``."""
    nan = np.full(out[entry_key].shape[1:], np.nan, dtype=np.float32)
    for cidx in channel_indices:
        out[entry_key][cidx, ...] = nan
        out["valid_data"][cidx] = 0


def convert(
    make_source: Callable[[Dict], InvariantSource],
    output_file: str,
    metadata_file: str,
    channel_names: List[str],
    entry_key: Optional[str] = "fields",
    force_overwrite: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    """Convert time invariant ERA5 fields from one of the supported archives to a single makani style file.

    The archive specific reading is delegated to a :class:`data_process.sources.InvariantSource`;
    this routine owns the output file.

    Parameters
    ----------
    make_source : Callable[[Dict], InvariantSource]
        Called with the invariant metadata dictionary, returns the source to read from.
    output_file : str
        Name of the file to write (makani format, without time axis).
    metadata_file : str
        Name of the dataset metadata file, see ``convert_era5_to_makani_input.py``. Only
        ``coords["lat"]`` and ``coords["lon"]`` are used, so the metadata of the dataset the
        invariants go with can be passed as is.
    channel_names : List[str]
        Invariant fields to extract, by ECMWF short name, e.g. ``["z", "lsm", "slt"]``.
    entry_key : str
        This is the HDF5 dataset name of the data in the file. Defaults to "fields".
    force_overwrite : bool
        Setting this flag to True will overwrite an existing file.
    verbose : bool
        Enable for more printing.
    """

    # timer
    start_time = time.perf_counter()

    if os.path.isfile(output_file) and not force_overwrite:
        print(f"File {output_file} already exists, skipping.")
        return

    # get metadata info, swapping the dataset channels for the invariant ones
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    lat = metadata["coords"]["lat"]
    lon = metadata["coords"]["lon"]
    metadata = {"coords": {"channel": list(channel_names), "lat": lat, "lon": lon}}

    source = make_source(metadata)

    if verbose:
        print(f"Extracting {len(channel_names)} invariants on a {len(lat)}x{len(lon)} grid: {', '.join(channel_names)}")

    # Write to a temporary file and move it into place only once it is complete:
    # a failed run must not leave a partial file behind, which the next run
    # would otherwise skip as already converted.
    tmp_file = f"{output_file}.{os.getpid()}.part"
    try:
        f = _create_output_file(tmp_file, entry_key, channel_names, lat, lon)
        try:
            # populate fields; channels the source cannot provide are written as missing
            source.fill(f, entry_key)
            skipped_channels = source.skipped_channel_indices()
            if skipped_channels:
                _write_missing(f, entry_key, skipped_channels)
        finally:
            f.close()
        os.replace(tmp_file, output_file)
        summary = source.summary()
    finally:
        source.close()
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)

    # end time
    end_time = time.perf_counter()
    run_time = str(dt.timedelta(seconds=end_time - start_time))

    print(f"All done. Run time {run_time}." + (f" {summary}" if summary else ""))

    return


def main(args):
    # backends are imported lazily, so that each only needs its own dependencies
    if args.source == "wb2":
        from data_process.sources.wb2 import Wb2InvariantSource

        make_source = functools.partial(
            Wb2InvariantSource,
            input_file=args.input_file,
            coord_mode=args.coord_mode,
            skip_missing_channels=args.skip_missing_channels,
        )
    elif args.source == "ncar":
        from data_process.sources.ncar import NcarInvariantSource

        make_source = functools.partial(
            NcarInvariantSource,
            bucket=args.bucket,
            skip_missing_channels=args.skip_missing_channels,
        )
    else:
        raise ValueError(f"Unknown source {args.source}.")

    convert(
        make_source=make_source,
        output_file=args.output_file,
        metadata_file=args.metadata_file,
        channel_names=args.channels,
        force_overwrite=args.force_overwrite,
        verbose=args.verbose,
    )


def build_parser() -> ap.ArgumentParser:
    """Command line parser with one subcommand per source, sharing the output related options."""
    common = ap.ArgumentParser(add_help=False)
    common.add_argument("--output_file", type=str, help="Local output file.", required=True)
    common.add_argument(
        "--metadata_file", type=str, help="Local file with metadata, only its grid is used.", required=True
    )
    common.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=list(invariant_variables),
        help=f"Invariants to extract, by ECMWF short name. Defaults to all of: {' '.join(invariant_variables)}",
    )
    common.add_argument("--skip_missing_channels", action="store_true", help="Skip missing channels and do not fail")
    common.add_argument("--force_overwrite", action="store_true", help="Overwrite an existing file")
    common.add_argument("--verbose", action="store_true")

    parser = ap.ArgumentParser(description="Convert time invariant ERA5 fields to a single makani style HDF5 file.")
    sources = parser.add_subparsers(dest="source", required=True, metavar="source")

    wb2 = sources.add_parser("wb2", parents=[common], help="WeatherBench2 / ARCO-ERA5 zarr store, e.g. on GCS")
    wb2.add_argument("--input_file", type=str, help="WB2 input file", required=True)
    wb2.add_argument(
        "--coord_mode",
        type=str,
        default="match",
        choices=["match", "force-flip-lat", "force"],
        help="How to align input lat/lon to metadata: match (default), force-flip-lat, force",
    )

    ncar = sources.add_parser("ncar", parents=[common], help="NSF NCAR ERA5 (RDA d633000) on S3")
    ncar.add_argument("--bucket", type=str, default=NCAR_ERA5_BUCKET, help="S3 bucket with NCAR ERA5 data")

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
