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

import os
import io
from collections import namedtuple
from typing import Optional
import multiprocessing as mp
import numpy as np
import concurrent.futures as cf
from PIL import Image, ImageDraw
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import wandb

import torch


def _op_select(x, channels):
    return x[channels[0], ...]


def _op_magnitude(x, channels):
    return np.sqrt(sum(np.square(x[c, ...]) for c in channels))


_PlotOp = namedtuple("_PlotOp", ["func", "min_channels", "max_channels"])

# Registry of the operations a plot list entry may request. Entries name an op
# from this table rather than carrying lambda source, which keeps plot lists
# pure data: they are never evaluated as code, wherever they come from.
PLOT_OPS = {
    "select": _PlotOp(_op_select, 1, 1),
    "magnitude": _PlotOp(_op_magnitude, 1, None),
}


def _check_plot_item(item):
    """Validate a plot list entry against :data:`PLOT_OPS`, returning its channel list."""
    op = item["op"]
    if op not in PLOT_OPS:
        raise ValueError(f"unknown plot op {op!r}, expected one of {sorted(PLOT_OPS)}")

    channels = list(item["channels"])
    spec = PLOT_OPS[op]
    if len(channels) < spec.min_channels or (spec.max_channels is not None and len(channels) > spec.max_channels):
        upper = "any number of" if spec.max_channels is None else spec.max_channels
        raise ValueError(f"plot op {op!r} takes {spec.min_channels} to {upper} channels, got {len(channels)}")

    return channels


def resolve_plot_list(plot_list, channel_names):
    """
    Resolve symbolic channel references in a plot list.

    Each entry of ``plot_list`` is a dict naming an op from :data:`PLOT_OPS` and
    the channels it applies to, e.g.
    ``{"name": "z500", "op": "select", "channels": ["z500"], "diverging": False}``.
    This walks the list, collects the union of referenced channels in first-seen
    order, rewrites each entry's channels to indices into a stripped tensor
    holding just those channels, and returns the new plot list together with the
    indices into the original ``channel_names`` layout.
    """
    ordered_refs = []
    seen = set()
    for item in plot_list:
        for name in _check_plot_item(item):
            if name not in seen:
                seen.add(name)
                ordered_refs.append(name)

    stripped_index = {name: i for i, name in enumerate(ordered_refs)}

    channel_indices = []
    for name in ordered_refs:
        if name not in channel_names:
            raise ValueError(f"plot list references channel {name!r} which is not in channel_names")
        channel_indices.append(channel_names.index(name))

    new_plot_list = []
    for item in plot_list:
        new_item = dict(item)
        new_item["channels"] = [stripped_index[name] for name in item["channels"]]
        new_plot_list.append(new_item)

    return new_plot_list, channel_indices


# we can run matplotlib in Agg mode in the subprocesses to save some memory overhead
def _worker_init():
    os.environ["MPLBACKEND"] = "Agg"


# per-process cache of (figure, pred_axes, truth_axes, pred_mesh, truth_mesh)
# keyed on (H, W, figsize, projection, cmap). Each ProcessPoolExecutor worker
# has its own copy; setting up mollweide axes is the dominant per-frame cost,
# so amortizing it across calls is the main speedup here.
_figure_cache = {}


def _get_or_create_figure(H, W, lat, lon, figsize, projection, cmap):
    import matplotlib.pyplot as plt

    key = (H, W, figsize, projection, cmap)
    if key in _figure_cache:
        return _figure_cache[key]

    Lon, Lat = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=figsize)
    placeholder = np.zeros((H, W))

    ax_pred = fig.add_subplot(2, 1, 1, projection=projection)
    mesh_pred = ax_pred.pcolormesh(Lon, Lat, placeholder, cmap=cmap)
    ax_pred.grid(True)
    ax_pred.set_xticklabels([])
    ax_pred.set_yticklabels([])

    ax_truth = fig.add_subplot(2, 1, 2, projection=projection)
    mesh_truth = ax_truth.pcolormesh(Lon, Lat, placeholder, cmap=cmap)
    ax_truth.grid(True)
    ax_truth.set_xticklabels([])
    ax_truth.set_yticklabels([])

    fig.tight_layout()

    entry = (fig, ax_pred, ax_truth, mesh_pred, mesh_truth)
    _figure_cache[key] = entry
    return entry


def plot_comparison(
    pred,
    truth,
    lat=None,
    lon=None,
    pred_title="Prediction",
    truth_title="Ground truth",
    cmap="RdBu",
    projection="mollweide",
    diverging=False,
    figsize=(6, 7),
    vmax=None,
):
    """
    Visualization tool to plot a comparison between ground truth and prediction
    pred: 2d array
    truth: 2d array
    cmap: colormap
    projection: "mollweide", "hammer", "aitoff" or None
    """
    if len(pred.shape) != 2:
        raise ValueError(f"expected pred to be a 2D array, got shape {tuple(pred.shape)}")
    if len(truth.shape) != 2:
        raise ValueError(f"expected truth to be a 2D array, got shape {tuple(truth.shape)}")
    if pred.shape != truth.shape:
        raise ValueError(
            f"expected pred and truth to have the same shape, got {tuple(pred.shape)} and {tuple(truth.shape)}"
        )

    H, W = pred.shape
    if (lat is None) or (lon is None):
        lon = np.linspace(-np.pi, np.pi, W)
        lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H)

    # only normalize with the truth
    if diverging:
        vmax = vmax or np.abs(truth).max()
        vmin = -vmax
    else:
        vmax = truth.max()
        vmin = truth.min()

    fig, ax_pred, ax_truth, mesh_pred, mesh_truth = _get_or_create_figure(
        H,
        W,
        lat,
        lon,
        figsize,
        projection,
        cmap,
    )

    mesh_pred.set_array(pred.ravel())
    mesh_pred.set_clim(vmin, vmax)
    ax_pred.set_title(pred_title)

    mesh_truth.set_array(truth.ravel())
    mesh_truth.set_clim(vmin, vmax)
    ax_truth.set_title(truth_title)

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)

    img = Image.open(buf)
    img.load()
    return img


def plot_rollout_metrics(metric_curves, var_names, score_path=None, file_prefix="curve", dtxdh=6):
    "Plots rollout metrics such as RMSE and ACC and saves them to the experiment directory"

    for ivar, var_name in enumerate(var_names):

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        metric_curve = metric_curves[ivar]

        # prepare the plot
        fig, ax = plt.subplots()

        # get the time sclaling for the time axis
        t = np.arange(1, len(metric_curve) + 1, 1) * dtxdh
        ax.plot(t, metric_curve, ".-")
        xticks = np.arange(0, len(metric_curve) + 1, 1) * dtxdh
        x_locator = ticker.FixedLocator(xticks)
        ax.xaxis.set_major_locator(x_locator)
        y_locator = ticker.MaxNLocator(nbins=20)
        ax.yaxis.set_major_locator(y_locator)
        ax.grid(which="major", alpha=0.5)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(file_prefix + " " + var_name)
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

        # write out the plot
        if score_path is not None:
            fig.savefig(os.path.join(score_path, file_prefix + "_" + var_name + ".png"))


def _draw_progress_bar(image, progress: float, y_pos: float = 0.5, margin: int = 20, thickness: int = 6):
    """Overlay a horizontal progress bar on the seam between the pred/truth subplots."""
    w, h = image.size

    y_pos = min(max(y_pos, 0.0), 1.0)
    progress = min(max(progress, 0.0), 1.0)
    margin = min(max(margin, 0), w // 2 - 1)
    thickness = max(thickness, 1)

    dy_neg = thickness // 2
    dy_pos = thickness - dy_neg

    y_mid = min(max(int(y_pos * h), dy_neg), h - dy_pos)

    x0, x1 = margin, w - margin
    y0, y1 = y_mid - dy_neg, y_mid + dy_pos
    fill_x = int(x0 + progress * (x1 - x0))

    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], fill=(225, 225, 225))
    if fill_x > x0:
        draw.rectangle([x0, y0, fill_x, y1], fill=(40, 40, 40))
    return image


def visualize_field(
    tag, op, channels, prediction, target, lat, lon, scale, bias, diverging, progress: Optional[float] = None
):
    """
    Render a prediction/ground-truth comparison for a single field.

    ``op`` names an entry of :data:`PLOT_OPS` and ``channels`` are the channel
    indices it is applied to, after ``prediction`` and ``target`` have been
    unscaled with ``scale`` and ``bias``. Returns ``(tag, image)``.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("visualize_field")

    # look the op up in the registry; unknown ops are an error, never code to run
    channels = _check_plot_item({"op": op, "channels": channels})
    func_handle = PLOT_OPS[op].func

    # unscale:
    pred = scale * prediction + bias
    targ = scale * target + bias

    # apply op:
    pred = func_handle(pred, channels)
    targ = func_handle(targ, channels)

    # generate image
    image = plot_comparison(
        pred,
        targ,
        lat,
        lon,
        pred_title="Prediction",
        truth_title="Ground truth",
        projection="mollweide",
        diverging=diverging,
    )

    if progress is not None:
        image = _draw_progress_bar(image, progress)

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()

    return tag, image


class VisualizationWrapper(object):
    """
    Handles visualization during training.

    ``plot_list`` is a list of dicts, one per rendered field, each with a
    ``name``, an ``op`` naming an entry of :data:`PLOT_OPS`, the ``channels``
    that op applies to, and a ``diverging`` flag selecting the colormap. If
    ``channel_names`` is given, channels are named and resolved against it;
    otherwise they must already be integer indices into the tensors passed to
    :meth:`add`.
    """

    def __init__(
        self,
        log_to_wandb,
        path,
        prefix,
        plot_list,
        channel_names=None,
        lat=None,
        lon=None,
        scale=1.0,
        bias=0.0,
        num_workers=1,
    ):
        self.log_to_wandb = log_to_wandb
        self.generate_video = True
        self.path = path
        self.prefix = prefix

        # If channel_names is provided, resolve the symbolic channel names in the
        # plot list to indices into a stripped tensor that contains only the
        # referenced channels. This avoids shipping unused channels to the
        # renderer subprocesses. Without channel_names the plot list is expected
        # to index the incoming tensors directly.
        if channel_names is not None:
            self.plot_list, self.channel_indices = resolve_plot_list(plot_list, channel_names)
        else:
            for item in plot_list:
                if not all(isinstance(c, (int, np.integer)) for c in _check_plot_item(item)):
                    raise ValueError("without channel_names, plot list channels must be integer indices")
            self.plot_list = plot_list
            self.channel_indices = None

        # grid
        self.lat = lat
        self.lon = lon

        # normalization: slice along the channel axis if we have a stripped index list
        if self.channel_indices is not None and not isinstance(scale, (int, float)):
            scale = scale[self.channel_indices].copy()
        if self.channel_indices is not None and not isinstance(bias, (int, float)):
            bias = bias[self.channel_indices].copy()
        self.scale = scale
        self.bias = bias

        # this is for parallel processing
        ctx = mp.get_context("spawn")
        self.executor = cf.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_worker_init)
        self.requests = []

    def reset(self):
        self.requests = []

    def add(self, tag, prediction, target, progress: Optional[float] = None):
        if self.channel_indices is not None:
            pred = prediction[self.channel_indices].copy()
            tar = target[self.channel_indices].copy()
        else:
            pred = np.copy(prediction)
            tar = np.copy(target)

        for item in self.plot_list:
            field_name = item["name"]
            plot_diverge = item["diverging"]
            self.requests.append(
                self.executor.submit(
                    visualize_field,
                    (tag, field_name),
                    item["op"],
                    item["channels"],
                    pred,
                    tar,
                    self.lat,
                    self.lon,
                    self.scale,
                    self.bias,
                    plot_diverge,
                    progress=progress,
                )
            )

        return

    def finalize(self):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("VisualizationWrapper:finalize")

        results = {}
        for request in cf.as_completed(self.requests):
            token, image = request.result()
            tag, field_name = token
            prefix = field_name + "_" + tag
            results[prefix] = image

        if self.generate_video:
            if self.log_to_wandb:
                video = []

                # draw stuff that goes on every frame here
                for prefix, image in sorted(results.items()):
                    video.append(np.transpose(np.asarray(image), (2, 0, 1)))

                video = np.stack(video)
                results = [wandb.Video(video, fps=3, format="gif")]
            else:
                video = []

                # draw stuff that goes on every frame here
                for prefix, image in sorted(results.items()):
                    video.append(np.asarray(image))

                video = ImageSequenceClip(video, fps=3)
                video.write_gif("video_output.gif", logger=None)

        else:
            results = [wandb.Image(image, caption=prefix) for prefix, image in results.items()]

        if self.log_to_wandb and results:
            wandb.log({"Inference samples": results}, commit=False)

        # reset requests
        self.reset()

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        return
