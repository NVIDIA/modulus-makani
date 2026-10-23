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

"""Render a rollout of one ensemble member next to the observation as a gif or mp4.

The prediction is a rollout file written by makani/inference.py with --output_file. The observation is read like the
data loaders read it: a directory of per-year files, a single concatenated or virtual (.h5v) file, or a zarr store.
Both are in physical units. The plotted field is a single channel or a combination of channels, using the ops of the
online visualization (makani.utils.visualize.PLOT_OPS), and defaults to the 10m wind speed sqrt(u10m^2 + v10m^2).

Example:

    python scripts/make_rollout_video.py --prediction rollout.h5 --observation /out_of_sample \\
        --timestamp 2020-06-01T00:00 --ensemble_members 0 --output wind10m.gif

    python scripts/make_rollout_video.py --prediction rollout.h5 --observation /out_of_sample \\
        --timestamp 2020-06-01T00:00 --channels z500 --output z500.mp4

Several members, or all of them with --ensemble_members all, are plotted one per row, each next to the observation.

With --data_output, the plotted fields are also written to an HDF5 file for numerical analysis: a dataset prediction of
shape (lead_time, ensemble_member, lat, lon) and a dataset observation of shape (lead_time, lat, lon), with lead time,
valid time, ensemble member, latitude and longitude as dimension scales and what was plotted (channels, op, members,
initial time, sources) as file attributes.
"""

import argparse
import datetime as dt
import json
import os

import h5py as h5
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # noqa: E402

from makani.utils.dataloaders.backends import get_backend  # noqa: E402
from makani.utils.visualize import PLOT_OPS, _check_plot_item  # noqa: E402


# the global map projections of matplotlib, which take longitude and latitude in radians. "none" plots lat/lon in
# degrees without a projection
PROJECTIONS = ["mollweide", "hammer", "aitoff", "lambert", "none"]


def _parse_utc(date_str):
    t = dt.datetime.fromisoformat(date_str)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    # the titles label times as UTC, so a time given in another zone is converted
    return t.astimezone(dt.timezone.utc)


def _decode(names):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def _metadata_channel_names(metadata_file):
    with open(metadata_file) as f:
        return json.load(f)["coords"]["channel"]


def _file_channel_names(files):
    """Channel names stored in the first observation file, if it is HDF5 and has them."""
    try:
        with h5.File(files[0], "r") as f:
            if "channel" in f:
                return _decode(f["channel"][...])
    except OSError:
        pass
    raise ValueError(f"{files[0]} has no channel names, pass the dataset metadata with --metadata_file")


class ObservationIndex:
    """
    Observations by time. The dataset is discovered by the same backends the data loaders use, so the observations
    can be a directory of per-year files, a single concatenated or virtual (.h5v) file, or a zarr store. Channel
    names come from --metadata_file, which a WeatherBench2 store requires, or else from the HDF5 files.
    """

    def __init__(self, path, dataset_key="fields", metadata_file=None, dhours=6):
        # the backend detection would report a missing path as an unknown layout
        if not os.path.exists(path):
            raise FileNotFoundError(f"observation {path} does not exist")

        # the backends search a directory for stores, so a single zarr store is found through its parent
        location, file_pattern = path, "*"
        if os.path.isdir(path) and path.rstrip("/").endswith(".zarr"):
            location, store = os.path.split(path.rstrip("/"))
            file_pattern = os.path.splitext(store)[0]

        # the WeatherBench2 layout addresses variables by name, so the backend needs the names up front
        channel_names = _metadata_channel_names(metadata_file) if metadata_file is not None else None

        # arbitrary file names, ordered by time, as for the inference datasets
        self.backend = get_backend(
            location, dataset_name=dataset_key, file_pattern=file_pattern, dhours=dhours, channel_names=channel_names
        )
        metadata = self.backend.discover()
        self.channels = channel_names if channel_names is not None else _file_channel_names(metadata.files)

        self.lookup = {}
        sample = 0
        for file_idx, num_samples in enumerate(metadata.samples_per_file):
            for local_idx in range(num_samples):
                key = int(round(metadata.timestamps[sample].timestamp()))
                self.lookup[key] = (file_idx, local_idx)
                sample += 1

    def read(self, time, channel_indices):
        key = int(round(time.timestamp()))
        if key not in self.lookup:
            raise KeyError(f"no observation at {time.isoformat()}")
        file_idx, local_idx = self.lookup[key]

        # backends read sorted channels, restore the requested order afterwards
        order = np.argsort(channel_indices)
        self.backend.open(file_idx)
        data = self.backend.read(file_idx, slice(local_idx, local_idx + 1), np.asarray(channel_indices)[order])[0]
        out = np.empty_like(data)
        out[order] = data
        return out


def _to_frame(fig):
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    # video codecs need even frame sizes
    h, w = frame.shape[0] - frame.shape[0] % 2, frame.shape[1] - frame.shape[1] % 2
    return frame[:h, :w].copy()


def write_plot_data(path, preds, observations, lead_times, valid_times, members, lat, lon, attrs):
    """
    Write the plotted fields to HDF5, as datasets prediction of shape (lead_time, ensemble_member, lat, lon) and
    observation of shape (lead_time, lat, lon).
    """
    with h5.File(path, "w") as f:
        for key, value in attrs.items():
            f.attrs[key] = value

        scales = {
            "lead_time": (np.asarray(lead_times, dtype=np.float64), "Lead time in hours"),
            "timestamp": (np.array([t.timestamp() for t in valid_times]), "Valid time in seconds since 1970, UTC"),
            "ensemble_member": (np.asarray(members, dtype=np.int64), "Ensemble member index in the rollout file"),
            "lat": (np.asarray(lat, dtype=np.float32), "Latitude in degrees"),
            "lon": (np.asarray(lon, dtype=np.float32), "Longitude in degrees"),
        }
        for key, (values, description) in scales.items():
            f[key] = values
            f[key].attrs["description"] = description
            f[key].make_scale(key)

        datasets = [
            ("prediction", preds, ["lead_time", "ensemble_member", "lat", "lon"]),
            ("observation", observations, ["lead_time", "lat", "lon"]),
        ]
        labels = dict(lead_time="Lead time", ensemble_member="Ensemble member", lat="Latitude", lon="Longitude")
        for key, data, dims in datasets:
            dset = f.create_dataset(key, data=np.stack(data).astype(np.float32))
            dset.attrs["description"] = f"{attrs['name']} of the {key}"
            for dim, scale in enumerate(dims):
                dset.dims[dim].attach_scale(f[scale])
                dset.dims[dim].label = labels[scale]
            # lead time and valid time both label the first dimension
            dset.dims[0].attach_scale(f["timestamp"])


def make_video(
    prediction,
    observation,
    timestamp,
    output,
    ensemble_members=(0,),
    channels=("u10m", "v10m"),
    op=None,
    name=None,
    metadata_file=None,
    dataset_key="fields",
    dhours=6,
    num_steps=None,
    fps=3,
    cmap="RdBu",
    diverging=False,
    clim="fixed",
    projection="mollweide",
    dpi=100,
    grid=False,
    data_output=None,
):
    if projection is None:
        projection = "none"
    if projection not in PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}, expected one of {PROJECTIONS}")
    projection = None if projection == "none" else projection

    channels = list(channels)
    if op is None:
        op = "select" if len(channels) == 1 else "magnitude"
    _check_plot_item({"op": op, "channels": channels})
    func = PLOT_OPS[op].func
    if name is None:
        name = channels[0] if op == "select" else f"|{', '.join(channels)}|"

    init_time = _parse_utc(timestamp)
    obs_index = ObservationIndex(observation, dataset_key=dataset_key, metadata_file=metadata_file, dhours=dhours)

    if not os.path.isfile(prediction):
        raise FileNotFoundError(f"prediction {prediction} does not exist")

    with h5.File(prediction, "r") as fpred:
        pred_channels = _decode(fpred["channel"][...])
        pred_times = fpred["timestamp"][...]
        lead_times = fpred["lead_time"][...]
        lat = fpred["lat"][...]
        lon = fpred["lon"][...]
        pred_data = fpred["fields"]

        # locate the initial condition and check the request against the file
        matches = np.nonzero(np.abs(pred_times - init_time.timestamp()) < 1.0)[0]
        if len(matches) == 0:
            available = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).isoformat() for t in pred_times[:5]]
            raise KeyError(
                f"no initial condition at {init_time.isoformat()} in {prediction}, it starts with {available}"
            )
        ic = int(matches[0])

        num_members = pred_data.shape[2]
        if isinstance(ensemble_members, str):
            if ensemble_members != "all":
                raise ValueError(f"ensemble members have to be indices or 'all', got {ensemble_members!r}")
            ensemble_members = list(range(num_members))
        ensemble_members = [int(m) for m in ensemble_members]
        if not ensemble_members:
            raise ValueError("no ensemble members selected")
        bad = [m for m in ensemble_members if not 0 <= m < num_members]
        if bad:
            raise IndexError(f"ensemble members {bad} out of range, the rollout has {num_members} members")

        missing = [c for c in channels if c not in pred_channels or c not in obs_index.channels]
        if missing:
            raise ValueError(f"channels {missing} are not in both the prediction and the observation")
        pred_cidx = [pred_channels.index(c) for c in channels]
        obs_cidx = [obs_index.channels.index(c) for c in channels]

        steps = len(lead_times) if num_steps is None else min(num_steps + 1, len(lead_times))
        valid_times = [init_time + dt.timedelta(hours=float(lead_times[s])) for s in range(steps)]

        # read everything first, the color limits of a fixed scale need all frames
        preds, observations = [], []
        for step, vtime in enumerate(valid_times):
            members = []
            for m in ensemble_members:
                pred = np.stack([pred_data[ic, step, m, c] for c in pred_cidx], axis=0)
                members.append(func(pred, list(range(len(channels)))))
            preds.append(np.stack(members, axis=0))
            observations.append(func(obs_index.read(vtime, obs_cidx), list(range(len(channels)))))

    if (observations[0].shape != preds[0].shape[1:]) or (observations[0].shape != (len(lat), len(lon))):
        raise ValueError(f"grid mismatch: prediction {preds[0].shape}, observation {observations[0].shape}")

    if data_output is not None:
        attrs = dict(
            name=name,
            op=op,
            channels=channels,
            ensemble_members=ensemble_members,
            init_time=init_time.isoformat(),
            init_timestamp=init_time.timestamp(),
            prediction_file=os.path.abspath(prediction),
            observation_path=os.path.abspath(observation),
        )
        write_plot_data(
            data_output, preds, observations, lead_times[:steps], valid_times, ensemble_members, lat, lon, attrs
        )

    def limits(t):
        # like the online visualization, the color scale follows the observation
        if diverging:
            vmax = np.nanmax(np.abs(t))
            return -vmax, vmax
        return np.nanmin(t), np.nanmax(t)

    if clim == "fixed":
        # combine the limits of every frame rather than stacking the frames into another copy
        frame_limits = np.array([limits(o) for o in observations])
        fixed = (np.nanmin(frame_limits[:, 0]), np.nanmax(frame_limits[:, 1]))

    # one figure with a row per member, its prediction next to the observation, updated per frame
    lon_plot = np.deg2rad(np.where(lon > 180.0, lon - 360.0, lon)) if projection else lon
    lat_plot = np.deg2rad(lat) if projection else lat
    order = np.argsort(lon_plot)
    Lon, Lat = np.meshgrid(lon_plot[order], lat_plot)

    titles = [title for m in ensemble_members for title in (f"Prediction, member {m}", "Observation")]
    nrows = len(ensemble_members)

    fig = plt.figure(figsize=(13, 3.2 * nrows + 1.0), dpi=dpi)
    axes, meshes = [], []
    for i, title in enumerate(titles):
        ax = fig.add_subplot(nrows, 2, i + 1, projection=projection)
        ax.set_title(title)
        mesh = ax.pcolormesh(Lon, Lat, np.zeros_like(Lon), cmap=cmap, shading="auto")
        ax.grid(grid)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        axes.append(ax)
        meshes.append(mesh)
    # the colorbar takes a fraction of the figure height, keep it the size of the single row layout
    fig.colorbar(
        meshes[-1],
        ax=axes,
        orientation="horizontal",
        fraction=0.05 / nrows,
        pad=0.04 / nrows,
        aspect=20 * nrows,
        label=name,
    )
    suptitle = fig.suptitle("")

    frames = []
    for step, (pred, targ, vtime) in enumerate(zip(preds, observations, valid_times)):
        vmin, vmax = fixed if clim == "fixed" else limits(targ)
        for mesh, field in zip(meshes, [field for member in pred for field in (member, targ)]):
            mesh.set_array(np.ma.masked_invalid(field[:, order]).ravel())
            mesh.set_clim(vmin, vmax)
        suptitle.set_text(
            f"{name}   init {init_time:%Y-%m-%d %H:%M} UTC   "
            f"+{lead_times[step]:.0f}h   valid {vtime:%Y-%m-%d %H:%M} UTC"
        )
        frames.append(_to_frame(fig))
    plt.close(fig)

    clip = ImageSequenceClip(frames, fps=fps)
    if output.lower().endswith(".gif"):
        clip.write_gif(output, fps=fps, logger=None)
    else:
        clip.write_videofile(output, fps=fps, codec="libx264", logger=None)

    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prediction", required=True, help="rollout file written by makani/inference.py")
    parser.add_argument(
        "--observation",
        required=True,
        help="observations as for the data loaders: a directory, an .h5/.h5v file or zarr",
    )
    parser.add_argument("--timestamp", required=True, help="initial condition, ISO format, UTC if no zone is given")
    parser.add_argument("--output", required=True, help="output file, .gif or a video format such as .mp4")
    parser.add_argument(
        "--ensemble_members",
        nargs="+",
        default=["0"],
        help="indices of the members to plot, or all, default 0",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["u10m", "v10m"],
        help="channels the op is applied to, default u10m v10m",
    )
    parser.add_argument(
        "--op",
        choices=sorted(PLOT_OPS),
        default=None,
        help="how the channels are combined, default select for one channel and magnitude for several",
    )
    parser.add_argument("--name", default=None, help="label of the plotted field")
    parser.add_argument(
        "--metadata_file", default=None, help="dataset metadata (data.json) with the channel names of the observations"
    )
    parser.add_argument("--dataset_key", default="fields", help="name of the data in the observation files")
    parser.add_argument(
        "--dhours", type=int, default=6, help="sample spacing, only used for per-year files without timestamps"
    )
    parser.add_argument("--num_steps", type=int, default=None, help="number of rollout steps, default all")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--cmap", default="RdBu", help="colormap, default RdBu as in the training visualization")
    parser.add_argument("--diverging", action="store_true", help="symmetric color scale around zero")
    parser.add_argument(
        "--clim",
        choices=["fixed", "per_frame"],
        default="fixed",
        help="color scale from the observation of all frames or of every frame",
    )
    parser.add_argument(
        "--projection",
        choices=PROJECTIONS,
        default="mollweide",
        help="map projection, default mollweide as in the training visualization, none for a lat/lon plot",
    )
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--grid", action="store_true", help="draw latitude and longitude lines")
    parser.add_argument(
        "--data_output",
        default=None,
        help="optional HDF5 file for the plotted fields, as datasets prediction (per member) and observation",
    )
    args = parser.parse_args()

    make_video(
        prediction=args.prediction,
        observation=args.observation,
        timestamp=args.timestamp,
        output=args.output,
        ensemble_members="all" if args.ensemble_members == ["all"] else [int(m) for m in args.ensemble_members],
        channels=args.channels,
        op=args.op,
        name=args.name,
        metadata_file=args.metadata_file,
        dataset_key=args.dataset_key,
        dhours=args.dhours,
        num_steps=args.num_steps,
        fps=args.fps,
        cmap=args.cmap,
        diverging=args.diverging,
        clim=args.clim,
        projection=args.projection,
        dpi=args.dpi,
        grid=args.grid,
        data_output=args.data_output,
    )


if __name__ == "__main__":
    main()
