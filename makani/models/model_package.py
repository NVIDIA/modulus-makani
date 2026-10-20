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

"""
Model package for easy inference/packaging. Model packages contain all the necessary data to
perform inference and its interface is compatible with earth2mip
"""
import os
import shutil
import json
import numpy as np
import torch
from makani.utils.YParams import ParamsBase, ensure_resampled_shapes
from makani.utils.driver import Driver
from makani.third_party.climt.zenith_angle_v2 import cos_zenith_angle
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.dataloaders.batch import ZENITH
from makani.models import model_registry
import datetime
import logging


logger = logging.getLogger(__name__)


class LocalPackage:
    r"""
    Filesystem-backed model package, implementing the PhysicsNeMo Package interface.

    A model package bundles everything needed to run a trained model outside the
    training environment: the checkpoint, the config, the normalization
    statistics, and the static fields. This class resolves the well-known
    filenames within a package directory, so consumers refer to package members
    by name rather than hard-coding the layout.

    Parameters
    ----------
    root : str
        Path to the package directory.
    """

    # These define the model package in terms of where makani expects the files to be located
    THIS_MODULE = "makani.models.model_package"
    MODEL_PACKAGE_CHECKPOINT_PATH = "training_checkpoints/best_ckpt_mp0.tar"
    MINS_FILE = "mins.npy"
    MAXS_FILE = "maxs.npy"
    MEANS_FILE = "global_means.npy"
    STDS_FILE = "global_stds.npy"
    INVARIANTS_FILE = "invariants.h5"

    def __init__(self, root):
        self.root = root

    def get(self, path):
        r"""
        Resolve a package-relative path to an absolute one.

        Parameters
        ----------
        path : str
            Path relative to the package root, typically one of the class-level
            filename constants.

        Returns
        -------
        str
            The absolute path. Existence is not checked.
        """
        return os.path.join(self.root, path)

    @staticmethod
    def _load_static_data(package, params):
        if params.get("invariants", None):
            params.invariants_path = package.get(LocalPackage.INVARIANTS_FILE)

        # alweays load all normalization files
        if params.get("global_means_path", None) is not None:
            params.global_means_path = package.get(LocalPackage.MEANS_FILE)
        if params.get("global_stds_path", None) is not None:
            params.global_stds_path = package.get(LocalPackage.STDS_FILE)
        if params.get("min_path", None) is not None:
            params.min_path = package.get(LocalPackage.MINS_FILE)
        if params.get("max_path", None) is not None:
            params.max_path = package.get(LocalPackage.MAXS_FILE)


class ModelWrapper(torch.nn.Module):
    r"""
    Model wrapper to make inference simple outside of makani.

    Presents a trained model as a plain callable that takes physical fields and
    a valid time, so downstream consumers do not need makani's training
    machinery. The wrapper owns the pieces that would otherwise be the caller's
    problem: input and output normalization, the lat-lon grid definition, and
    the solar zenith angle channel, which has to be computed from the valid time
    rather than read from the data.

    Parameters
    ----------
    model : torch.nn.Module
        ML model that is wrapped.
    params : ParamsBase
        Parameter object containing information on how the model was
        initialized in makani. Params assembled outside
        :func:`load_model_package` (e.g. by earth2studio) are tolerated and do
        not need to carry the resampled shapes.

    Attributes
    ----------
    model : torch.nn.Module
        ML model that is wrapped.
    params : ParamsBase
        parameter object containing information on how the model was initialized in makani
    lats : numpy.ndarray
        Latitudes of the model grid, taken from ``params`` if present and
        otherwise assumed equiangular from 90 to -90.
    lons : numpy.ndarray
        Longitudes of the model grid, taken from ``params`` if present and
        otherwise assumed equiangular from 0 to 360.

    Methods
    -------
    forward(x, time):
        performs a single prediction steps
    """

    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params
        # tolerate params assembled outside load_model_package (e.g. earth2studio),
        # which need not carry the resampled shapes
        ensure_resampled_shapes(params)
        nlat = params.img_shape_x_resampled
        nlon = params.img_shape_y_resampled

        # configure lats
        if "lat" in self.params:
            self.lats = np.asarray(self.params.lat)
        else:
            self.lats = np.linspace(90, -90, nlat, endpoint=True)

        # configure lons
        if "lon" in self.params:
            self.lons = np.asarray(self.params.lon)
        else:
            self.lons = np.linspace(0, 360, nlon, endpoint=False)

        # zenith angle
        self.add_zenith = params.get("add_zenith", False)
        if self.add_zenith:
            self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)

        # load the normalization files
        bias, scale = get_data_normalization(self.params)

        # convert them to torch
        in_bias = torch.as_tensor(bias[:, self.params.in_channels]).to(torch.float32)
        in_scale = torch.as_tensor(scale[:, self.params.in_channels]).to(torch.float32)
        out_bias = torch.as_tensor(bias[:, self.params.out_channels]).to(torch.float32)
        out_scale = torch.as_tensor(scale[:, self.params.out_channels]).to(torch.float32)

        self.register_buffer("in_bias", in_bias, persistent=True)
        self.register_buffer("in_scale", in_scale, persistent=True)
        self.register_buffer("out_bias", out_bias, persistent=True)
        self.register_buffer("out_scale", out_scale, persistent=True)

    @property
    def in_channels(self):
        r"""
        Names of the input channels.

        Returns
        -------
        list of str or None
            Channel names, or ``None`` if the config does not record them.
        """
        return self.params.get("channel_names", None)

    @property
    def out_channels(self):
        r"""
        Names of the output channels.

        Returns
        -------
        list of str or None
            Channel names, or ``None`` if the config does not record them.
        """
        return self.params.get("channel_names", None)

    @property
    def timestep(self):
        r"""
        Model time step in hours.

        Returns
        -------
        int or float
            ``dt * dhours``: the number of data steps per model step times the
            spacing of the data in hours.
        """
        return self.params.dt * self.params.dhours

    def update_state(self, replace_state=True, batch_size=None):
        """Advance the stochastic state without running a forward pass.

        :meth:`forward` sizes the state itself when called with
        ``replace_state=True``; prime it here only to continue a sequence from
        the first call on, at the batch you intend to run.

        Parameters
        ----------
        replace_state : bool, optional
            Draw a fresh state instead of taking one autoregressive step. By
            default ``True``.
        batch_size : int, optional
            Size the noise state to this batch. ``None`` keeps the current size,
            which for a freshly loaded package is ``params.batch_size`` (usually 1).
        """
        self.model.preprocessor.update_internal_state(replace_state=replace_state, batch_size=batch_size)
        return

    def set_rng(self, reset=True, seed=333):
        r"""
        Re-seed the model's stochastic noise, optionally clearing its state.

        Use this to make an ensemble member reproducible, or to decorrelate
        members by giving each a distinct seed.

        Parameters
        ----------
        reset : bool, optional
            Also zero the noise state, by default ``True``.
        seed : int, optional
            New seed, by default ``333``.
        """
        self.model.preprocessor.set_rng(reset=reset, seed=seed)
        return

    def _zenith_features(self, x, time):
        """Build the cosine-zenith feature for ``x`` at valid time(s) ``time``.

        Returns a tensor of shape ``(B, nhist, 1, H, W)``, a dynamic feature as
        the preprocessor takes it, where ``B == x.shape[0]`` and
        ``nhist == n_history + 1``.

        See :meth:`forward` for the ``time`` contract this enforces.
        """
        nhist = self.model.preprocessor.n_history + 1
        batch_size = x.shape[0]

        # cos_zenith_angle promotes scalars via np.atleast_1d, so this is always
        # (T, H, W) -- never (H, W).
        cosz = cos_zenith_angle(time, self.lon_grid, self.lat_grid).astype(np.float32)
        z = torch.as_tensor(cosz).to(device=x.device)
        n_times = z.shape[0]

        if n_times == batch_size * nhist:
            # one time per (member, history step), member-major
            z = z.reshape(batch_size, nhist, *z.shape[-2:])
        elif n_times == nhist:
            # a single history window shared by every member (e.g. a
            # perturbed-initial-condition ensemble, all valid at the same time)
            z = z.reshape(1, nhist, *z.shape[-2:]).expand(batch_size, -1, -1, -1)
        else:
            raise ValueError(
                f"add_zenith: got {n_times} time(s) for a batch of {batch_size} with "
                f"n_history={nhist - 1}. Pass either {batch_size * nhist} times (one per "
                f"member per history step, member-major) or {nhist} time(s) to share one "
                f"history window across the whole batch."
            )

        return z.unsqueeze(2)

    def _prepare_input(self, x, time, normalized_data):
        """Normalize the input.

        ``x`` is ``(B, (n_history + 1) * C, H, W)``, history in the flattened
        channel axis, oldest first, or ``(B, n_history + 1, C, H, W)``. Returns
        the input with the time axis explicit, as :class:`StepWrapper` takes it.
        """
        preprocessor = self.model.preprocessor
        if x.dim() == 4:
            x = preprocessor.expand_history(x, preprocessor.n_history + 1)

        if not normalized_data:
            # the statistics are per channel, (1, C, 1, 1); broadcast them over the time steps
            x = (x - self.in_bias.unsqueeze(1)) / self.in_scale.unsqueeze(1)

        return x

    def _step_inputs(self, x, time, normalized_data, replace_state):
        """Everything :class:`StepWrapper` takes for one step: the window, its dynamic features and its noise.

        The noise state is advanced here, replaced if ``replace_state`` is set, at the batch of ``x``;
        see :meth:`forward` for the batching contract on ``time`` and the noise.
        """
        x = self._prepare_input(x, time, normalized_data)
        features = {ZENITH: self._zenith_features(x, time)} if self.add_zenith else {}

        noise = None
        noise_source = self.model.preprocessor.noise_source(batch_size=x.shape[0])
        if noise_source is not None:
            noise_source.update(replace_state=replace_state)
            noise = noise_source()

        return x, features, noise

    def encode_process(self, x, time, normalized_data=True, replace_state=None):
        """Return the backbone's processed latent state without running its decoder.

        Input normalization, zenith-angle generation, history handling, and
        static-feature preparation are identical to :meth:`forward` -- including
        the batching contract documented there. Only the decoder, bias
        correction, and output denormalization are skipped, so the result stays
        in the backbone's latent feature space and ``normalized_data`` affects
        only how ``x`` is interpreted, never the output.

        Like :meth:`forward`, the call advances the noise state by one step, so
        ``encode_process`` followed by ``forward`` on the same input draws two
        different noise samples.

        Parameters
        ----------
        x, time, normalized_data, replace_state
            As in :meth:`forward`.

        Returns
        -------
        torch.Tensor
            The latent features, ``(B, embed_dim, h, w)`` on the internal grid of
            the backbone.

        Raises
        ------
        NotImplementedError
            If the backbone does not implement ``encode_process`` (currently only
            FCN3 does).
        """
        if not hasattr(self.model, "encode"):
            raise NotImplementedError(f"{type(self.model).__name__} does not expose encode_process().")
        x, features, noise = self._step_inputs(x, time, normalized_data, replace_state)
        return self.model.encode(x, features, noise)

    def forward(self, x, time, normalized_data=True, replace_state=None):
        """Advance the model state by one timestep.

        Parameters
        ----------
        x : torch.Tensor
            Input state, shape ``(B, (n_history + 1) * C, H, W)`` with the
            history in the flattened channel axis, oldest first, or
            ``(B, n_history + 1, C, H, W)``.
        time : datetime or array of datetimes
            Valid time(s) for ``x``; see the contract below.
        normalized_data : bool, optional
            If False, ``x`` is normalized on the way in and the prediction is
            denormalized on the way out. If True (default), both are assumed to
            already be in normalized space.
        replace_state : bool or None, optional
            If true, draw a fresh noise state at the batch of ``x`` before the
            step; otherwise continue the noise sequence by one step. Ignored for
            a model without input noise.

        Returns
        -------
        torch.Tensor
            Prediction for the next timestep, shape ``(B, C_out, H, W)``,
            denormalized unless ``normalized_data`` is ``True``.

        Notes
        -----
        **Batching contract.**
        Any batch size ``B`` is supported. The cosine-zenith feature is built to
        ``(B, n_history + 1, 1, H, W)`` on every call, and the stochastic noise
        state, the only state the package keeps between calls, is sized to
        ``B``. Note this is independent of ``params.batch_size``, which is a
        *training* setting and is typically 1 in a packaged model.

        For a model with input noise, the first call at a new ``B`` must draw a
        fresh state: pass ``replace_state=True``, or prime once up front with
        ``update_state(replace_state=True, batch_size=B)``. Resizing the noise
        state mid-sequence is refused rather than silently restarting the
        autoregressive noise from zero -- see
        :meth:`Preprocessor2D.update_internal_state`. Models without input noise
        are unaffected.

        ``time`` must supply either one time per member per history step
        (``B * (n_history + 1)`` entries, ordered member-major), or exactly
        ``n_history + 1`` entries to share a single history window across the
        whole batch -- the natural form for a perturbed-initial-condition
        ensemble in which every member is valid at the same time. Any other
        count raises rather than silently mis-broadcasting. At ``B == 1`` the two
        forms coincide.

        Keep ``B`` fixed across a rollout. Changing it mid-rollout is refused for
        stateful noise unless ``replace_state=True``, which starts a fresh noise
        sequence.

        Raises
        ------
        RuntimeError
            If stateful noise would have to be resized while continuing its
            sequence, see above.
        """
        x, features, noise = self._step_inputs(x, time, normalized_data, replace_state)

        # one step, (B, 1, C_out, H, W)
        out = self.model.predict(x, features, noise).squeeze(1)

        if not normalized_data:
            out = out * self.out_scale + self.out_bias

        return out


def save_model_package(params):
    """
    Saves out a self-contained model-package.
    The idea is to save anything necessary for inference beyond the checkpoints in one location.
    """
    if params.get("enable_synthetic_data", False):
        # A package exists to make a checkpoint usable for inference elsewhere: it bundles the
        # normalization statistics and the invariant fields next to the config. A synthetic run
        # has neither -- the .npy and .h5 files it would copy do not exist, which is what used to
        # make this raise -- and its checkpoint is random weights over made-up data, so there is
        # nothing worth packaging even if the files were there.
        logging.info("Skipping model package: synthetic data carries no statistics or invariants to package.")
        return

    # save out the current state of the parameters, make it human readable
    config_path = os.path.join(params.experiment_dir, "config.json")

    with open(config_path, "w") as f:
        msg = json.dumps(params.to_dict(), indent=4, sort_keys=True)
        f.write(msg)

    # copy static data into the package under the canonical file names expected
    # by LocalPackage, so packages are self-consistent regardless of how the
    # source files happen to be named on a given system
    if params.get("invariants", None):
        shutil.copy(params.invariants_path, os.path.join(params.experiment_dir, LocalPackage.INVARIANTS_FILE))

    # always save out all normalization files under their canonical names
    if params.get("global_means_path", None) is not None:
        shutil.copy(params.global_means_path, os.path.join(params.experiment_dir, LocalPackage.MEANS_FILE))
    if params.get("global_stds_path", None) is not None:
        shutil.copy(params.global_stds_path, os.path.join(params.experiment_dir, LocalPackage.STDS_FILE))
    if params.get("min_path", None) is not None:
        shutil.copy(params.min_path, os.path.join(params.experiment_dir, LocalPackage.MINS_FILE))
    if params.get("max_path", None) is not None:
        shutil.copy(params.max_path, os.path.join(params.experiment_dir, LocalPackage.MAXS_FILE))

    # write out earth2mip metadata.json
    fcn_mip_data = {
        "entrypoint": {"name": f"{LocalPackage.THIS_MODULE}:load_time_loop"},
    }
    with open(os.path.join(params.experiment_dir, "metadata.json"), "w") as f:
        msg = json.dumps(fcn_mip_data, indent=4, sort_keys=True)
        f.write(msg)


# TODO: this is not clean and should be reworked to allow restoring from params + checkpoint file
def load_model_package(package, pretrained=True, device="cpu", multistep=False):
    """
    Loads model package and return the wrapper which can be used for inference.

    ``multistep`` is accepted for backward compatibility and has no effect: the
    same wrapper serves single steps and rollouts.
    """
    path = package.get("config.json")
    params = ParamsBase.from_json(path)
    # resampled shapes are set at runtime from the dataset during training and are
    # absent from packages written before resampling existed
    ensure_resampled_shapes(params)
    LocalPackage._load_static_data(package, params)

    # assume we are not distributed
    # distributed checkpoints might be saved with different params values
    params.img_local_offset_x = 0
    params.img_local_offset_y = 0
    params.img_local_shape_x = params.img_shape_x
    params.img_local_shape_y = params.img_shape_y

    # get the model and
    model = model_registry.get_model(params).to(device)

    if pretrained:
        best_checkpoint_path = package.get(LocalPackage.MODEL_PACKAGE_CHECKPOINT_PATH)
        Driver.restore_from_checkpoint(best_checkpoint_path, model)

    model = ModelWrapper(model, params=params)

    # by default we want to do evaluation so setting it to eval here
    model.eval()

    return model


def load_time_loop(package, device=None, time_step_hours=None):
    """This function loads an earth2mip TimeLoop object that
    can be used for inference.

    A TimeLoop encapsulates normalization, regridding, and other logic, so is a
    very minimal interface to expose to a framework like earth2mip.

    See https://github.com/NVIDIA/earth2mip/blob/main/docs/concepts.rst
    for more info on this interface.
    """

    from earth2mip.networks import Inference
    from earth2mip.grid import equiangular_lat_lon_grid

    config = package.get("config.json")
    params = ParamsBase.from_json(config)

    if params.in_channels != params.out_channels:
        raise NotImplementedError("Non-equal input and output channels are not implemented yet.")

    names = [params.data_channel_names[i] for i in params.in_channels]
    params.min_path = package.get(LocalPackage.MINS_FILE)
    params.max_path = package.get(LocalPackage.MAXS_FILE)
    params.global_means_path = package.get(LocalPackage.MEANS_FILE)
    params.global_stds_path = package.get(LocalPackage.STDS_FILE)

    center, scale = get_data_normalization(params)

    model = load_model_package(package, pretrained=True, device=device)
    shape = (params.img_crop_shape_x, params.img_crop_shape_y)

    # TODO: insert a check to see if the grid e2mip computes is the same that makani uses
    grid = equiangular_lat_lon_grid(
        nlat=params.img_crop_shape_x, nlon=params.img_crop_shape_y, includes_south_pole=True
    )

    if time_step_hours is None:
        hour = datetime.timedelta(hours=1)
        time_step = hour * params.get("dt", 6)
    else:
        time_step = datetime.timedelta(hours=time_step_hours)

    # Here we use the built-in class earth2mip.networks.Inference
    # will later be extended to use the makani inferencer
    inference = Inference(
        model=model,
        channel_names=names,
        center=center[:, params.in_channels],
        scale=scale[:, params.out_channels],
        grid=grid,
        n_history=params.n_history,
        time_step=time_step,
    )
    inference.to(device)
    return inference
