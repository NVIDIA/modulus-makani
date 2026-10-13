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

"""The DALI training dataloader.

Wraps :class:`~makani.utils.dataloaders.sample_source.SampleSource` in a DALI
pipeline: the source assembles one window at a time in a python worker, and DALI
batches it, moves it to the GPU and applies normalization there. Which files the
samples come from is the backends' concern, and how a window is built is the
source's; this module is the pipeline around them.

Two things are worth knowing about it. The pipeline can be checkpointed mid-epoch
and rebuilt from that state, which is what lets a training run resume without
replaying an epoch -- see :meth:`DaliDataloader.state_dict`. And restoring is
always a rebuild, since DALI only accepts serialized state at construction.
"""

import os
import logging
from typing import Any, Dict, Optional

import torch
import numpy as np

# DALI stuff
import nvidia.dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# distributed stuff
from makani.utils import comm
from makani.utils.dataloaders.data_helpers import parse_data_normalization_mode

# es helper
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.dataloaders.data_shapes import DataShapes
from makani.utils.dataloaders.batch import make_batch
from makani.utils.dataloaders.sample_source import SampleSource
from makani.utils.grids import GridConverter


# bump whenever the layout of the dict returned by DaliDataloader.state_dict changes
_DATALOADER_STATE_VERSION = 1


class DaliDataloader(object):
    """Feeds batches to training, reading through a dataset backend.

    Iterating yields the tensors named when the pipeline was built: the input and
    target windows, followed by the zenith angles and timestamps when those were
    asked for. Normalization and any grid conversion happen on the GPU, so what
    comes out is ready for the model.

    Built from ``params``, which names the dataset and its layout, the channels,
    the window geometry and the decomposition. Set
    ``checkpoint_dataloader_state`` to make the pipeline resumable.
    """

    def get_pipeline(self, checkpoint: Optional[bytes] = None):
        pipeline = Pipeline(
            batch_size=self.batchsize,
            num_threads=2,
            device_id=self.device_index,
            py_num_workers=self.num_data_workers,
            py_start_method="spawn",
            seed=self.global_seed,
            prefetch_queue_depth=2,
            enable_checkpointing=self.enable_checkpointing,
            checkpoint=checkpoint,
        )

        img_shape_x = self.data_shapes.img_shape_x
        img_shape_y = self.data_shapes.img_shape_y
        in_channels = self.in_channels
        out_channels = self.out_channels

        # one layout per output of the source, in the order it returns them
        layout = self.extsource.output_layout()
        num_outputs = len(layout)

        with pipeline:
            # get input and target
            data = fn.external_source(
                source=self.extsource,
                num_outputs=num_outputs,
                layout=layout,
                batch=False,
                no_copy=True,
                parallel=True,
                prefetch_queue_depth=self.num_data_workers,
            )

            inp = data[0]
            tar = data[1]
            off = 2
            if self.add_zenith:
                izen = data[2]
                tzen = data[3]
                off += 2

            itime = data[0 + off]
            ttime = data[1 + off]

            # upload to GPU
            if self.dali_device == "gpu":
                inp = inp.gpu()
                tar = tar.gpu()
                if self.add_zenith:
                    izen = izen.gpu()
                    tzen = tzen.gpu()

            # normalize if requested
            if self.normalize:
                inp = fn.normalize(
                    inp,
                    device=self.dali_device,
                    axis_names=self.norm_channels,
                    batch=self.norm_batch,
                    mean=self.in_bias,
                    stddev=self.in_scale,
                )

                tar = fn.normalize(
                    tar,
                    device=self.dali_device,
                    axis_names=self.norm_channels,
                    batch=self.norm_batch,
                    mean=self.out_bias,
                    stddev=self.out_scale,
                )

            # add zenith angle if requested
            pout = (inp, tar)

            if self.add_zenith:
                pout = pout + (izen, tzen)

            pout = pout + (itime, ttime)

            pipeline.set_outputs(*pout)

        return pipeline

    def __init__(self, params, location, train, seed=333, dali_device="gpu"):
        # set up workers and devices
        self.num_data_workers = params.num_data_workers
        # The parallel external_source path requires py_num_workers >= 1 and uses
        # num_data_workers as the ES prefetch_queue_depth (must be >= 1 too). With 0,
        # DALI silently disables the worker pool while parallel=True still requests
        # parallel scheduling, which is an inconsistent configuration.
        if self.num_data_workers < 1:
            raise ValueError(f"num_data_workers must be >= 1 for the DALI loader, got {self.num_data_workers}.")
        self.dali_device = dali_device
        if self.dali_device == "gpu":
            self.device_index = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{self.device_index}")
        elif self.dali_device == "cpu":
            self.device_index = None
            self.device = torch.device("cpu")
        else:
            raise NotImplementedError(f"Makani currently does not support dali_device {self.dali_device}")

        # batch size
        self.batchsize = int(params.batch_size)

        # whether the pipeline traces its state so that it can be checkpointed and restored
        self.enable_checkpointing = params.get("checkpoint_dataloader_state", True)

        # this is already rank-gated by the training scripts, so it keeps warnings on rank 0
        self.log_to_screen = params.get("log_to_screen", False)

        # set up seeds
        # this one is the same on all ranks
        self.global_seed = seed
        # this one is the same for all ranks of the same model
        model_id = comm.get_world_rank() // comm.get_size("model")
        self.model_seed = self.global_seed + model_id
        # this seed is supposed to be diffferent for every rank
        self.local_seed = self.global_seed + comm.get_world_rank()

        # we need to copy those
        self.location = location
        self.train = train
        self.dt = params.dt
        self.dhours = params.dhours
        self.n_history = params.n_history
        self.n_future = params.n_future if train else params.valid_autoreg_steps
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.subsampling_factor = params.get("subsampling_factor", 1)
        self.add_zenith = params.get("add_zenith", False)
        if hasattr(params, "lat") and hasattr(params, "lon"):
            self.lat_lon = (params.lat, params.lon)
        else:
            self.lat_lon = None
        self.dataset_name = params.h5_path
        if train:
            self.n_samples = params.get("n_train_samples", None)
            self.n_samples_per_epoch = params.get("n_train_samples_per_epoch", None)
        else:
            self.n_samples = params.get("n_eval_samples", None)
            self.n_samples_per_epoch = params.get("n_eval_samples_per_epoch", None)

        # by default we normalize offline, over space
        self.norm_mode, self.norm_channels, self.norm_batch = parse_data_normalization_mode(
            params.get("data_normalization_mode", None)
        )

        # set sharding
        self.num_shards = params.data_num_shards
        self.shard_id = params.data_shard_id

        # get cropping:
        crop_size = [params.get("crop_size_x", None), params.get("crop_size_y", None)]
        crop_anchor = [params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)]

        if not (os.path.isfile(self.location) or os.path.isdir(self.location)):
            raise IOError(f"Path {self.location} does not exist.")

        # get list of excluded timestamps
        timestamp_boundary_list = params.get("analysis_epoch_start_dates", [])

        # the storage layout is detected by the backend; a run that knows what it
        # has can name it instead through the "dataset_backend" parameter
        self.extsource = SampleSource(
            self.location,
            max_samples=self.n_samples,
            samples_per_epoch=self.n_samples_per_epoch,
            train=self.train,
            batch_size=self.batchsize,
            dt=self.dt,
            dhours=self.dhours,
            n_history=self.n_history,
            n_future=self.n_future,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            crop_size=crop_size,
            crop_anchor=crop_anchor,
            subsampling_factor=self.subsampling_factor,
            grid_type=params.data_grid_type,
            num_shards=self.num_shards,
            shard_id=self.shard_id,
            io_grid=params.get("io_grid", [1, 1, 1]),
            io_rank=params.get("io_rank", [0, 0, 0]),
            device_id=self.device_index,
            truncate_old=True,
            zenith_angle=self.add_zenith,
            # every batch carries its timestamps
            return_timestamp=True,
            lat_lon=self.lat_lon,
            dataset_name=self.dataset_name,
            timestamp_name=params.get("timestamp_name", "timestamp"),
            channel_names=params.get("channel_names", None),
            enable_odirect=params.enable_odirect,
            odirect_alignment=params.get("odirect_alignment", 0),
            enable_s3=params.enable_s3,
            enable_logging=params.log_to_screen,
            seed=self.global_seed,
            is_parallel=True,
            timestamp_boundary_list=timestamp_boundary_list,
            backend=params.get("dataset_backend", None),
        )

        # grid types
        self.grid_converter = GridConverter(
            params.data_grid_type,
            params.model_grid_type,
            torch.deg2rad(torch.tensor(self.extsource.lat_lon_local[0])).to(torch.float32).to(self.device),
            torch.deg2rad(torch.tensor(self.extsource.lat_lon_local[1])).to(torch.float32).to(self.device),
        )

        # some image properties
        # the geometry the run needs, in the one shape every loader reports it
        self.data_shapes = DataShapes.from_loader(self.extsource, grid_converter=self.grid_converter)
        self.lat_lon_local = self.extsource.lat_lon_local

        self.num_steps_per_epoch = self.extsource.num_steps_per_epoch

        # load stats
        self.normalize = True

        # in
        if self.norm_mode == "offline":
            bias, scale = get_data_normalization(params)

            if bias is not None:
                self.in_bias = bias[:, self.in_channels]
                self.out_bias = bias[:, self.out_channels]
            else:
                self.in_bias = np.zeros((1, len(self.in_channels), 1, 1))
                self.out_bias = np.zeros((1, len(self.out_channels), 1, 1))

            if scale is not None:
                self.in_scale = scale[:, self.in_channels]
                self.out_scale = scale[:, self.out_channels]
            else:
                self.in_scale = np.ones((1, len(self.in_channels), 1, 1))
                self.out_scale = np.ones((1, len(self.out_channels), 1, 1))

            # reformat the biases
            if self.norm_channels == "FHW":
                in_shape = (1, len(self.in_channels), 1, 1)
                out_shape = (1, len(self.out_channels), 1, 1)
            else:
                in_shape = (1, *self.in_bias.shape)
                out_shape = (1, *self.out_bias.shape)

            self.in_bias = np.reshape(self.in_bias, in_shape)
            self.in_scale = np.reshape(self.in_scale, in_shape)
            self.out_bias = np.reshape(self.out_bias, out_shape)
            self.out_scale = np.reshape(self.out_scale, out_shape)
        else:
            # in case of online normalization,
            # we do not need to set it here
            self.in_bias = None
            self.in_scale = None
            self.out_bias = None
            self.out_scale = None

        # create pipeline and iterator
        self.pipeline = None
        self.iterator = None
        self._build_pipeline()

    def _build_pipeline(self, checkpoint: Optional[bytes] = None):
        """
        (Re-)create the DALI pipeline and the iterator wrapping it.

        If ``checkpoint`` is passed, the pipeline resumes at the iteration the checkpoint was
        taken at instead of starting at the beginning of the first epoch. Rebuilding is the only
        way to restore a DALI pipeline, since the serialized state can only be handed over at
        construction time.
        """

        # tear down a previously built pipeline first, so that its python worker pool is
        # released before the new one spawns its own workers
        self.close()

        # build() is auto-called by DALIGenericIterator on first run since DALI 1.46 (#5754).
        # start_py_workers() is still explicit to keep worker startup ordered before any CUDA
        # context is acquired in this process.
        self.pipeline = self.get_pipeline(checkpoint=checkpoint)
        self.pipeline.start_py_workers()

        # create iterator
        outnames = ["inp", "tar"]
        if self.add_zenith:
            outnames += ["izen", "tzen"]
        outnames += ["itime", "ttime"]

        self.iterator = DALIGenericIterator(
            [self.pipeline],
            outnames,
            auto_reset=True,
            size=-1,
            last_batch_policy=LastBatchPolicy.DROP,
            prepare_first_batch=True,
        )

    def close(self):
        """
        Shut the DALI pipeline down: stop its python workers and release its native threads
        and device buffers. The loader cannot be iterated afterwards.

        Without an explicit close, the pipeline lives until the garbage collector happens to
        destroy it, and its native threads keep running into whatever the process does next.
        """

        # the iterator holds a reference to the pipeline, so it has to go first
        self.iterator = None
        if self.pipeline is not None:
            self.pipeline._shutdown()
            self.pipeline = None

    def state_dict(self) -> Dict[str, Any]:
        """
        Capture the state of the data pipeline so that training can be resumed mid-epoch.

        The heavy lifting is done by DALI: ``pipeline_checkpoint`` is the serialized state of
        the pipeline and of the iterator wrapping it, which encodes how many batches have been
        consumed and in which epoch. The external source callback (``SampleSource``) derives its
        shuffling permutation and sample index purely from the epoch/iteration counters DALI
        passes in, so restoring those counters restores the full sample order.

        The remaining entries are not restored but validated on load: replaying a state into a
        differently sharded or differently seeded pipeline would silently produce a different
        sample sequence.

        The serialized DALI state is stored as a uint8 tensor rather than as raw bytes, since
        makani checkpoints are read back with the restricted (``weights_only=True``) unpickler,
        which handles tensors but not arbitrary byte blobs.
        """

        if not self.enable_checkpointing:
            raise RuntimeError(
                "The dataloader state cannot be captured because checkpointing was disabled. "
                "Set checkpoint_dataloader_state to True to use this feature."
            )

        # one pipeline per dataloader, hence a single checkpoint
        checkpoints = self.iterator.checkpoints()

        # bytearray because frombuffer requires a writable buffer
        pipeline_checkpoint = torch.frombuffer(bytearray(checkpoints[0]), dtype=torch.uint8)

        return {
            "format_version": _DATALOADER_STATE_VERSION,
            "dali_version": nvidia.dali.__version__,
            "pipeline_checkpoint": pipeline_checkpoint,
            "num_shards": self.num_shards,
            "shard_id": self.shard_id,
            "batch_size": self.batchsize,
            "num_steps_per_epoch": self.num_steps_per_epoch,
            "global_seed": self.global_seed,
            "train": self.train,
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> bool:
        """
        Restore the state of the data pipeline captured by :meth:`state_dict`.

        Returns True if the state was restored and False if it was skipped because it does not
        match the current pipeline (only possible with ``strict=False``, otherwise a mismatch
        raises). Restoring rebuilds the pipeline, which restarts the python worker pool.
        """

        if not self.enable_checkpointing:
            raise RuntimeError(
                "The dataloader state cannot be restored because checkpointing was disabled. "
                "Set checkpoint_dataloader_state to True to use this feature."
            )

        def _mismatch(message):
            if strict:
                raise ValueError(
                    f"Error, the stored dataloader state does not match the current dataloader: {message}. "
                    "Set strict_restore to False to skip restoring the dataloader state instead."
                )
            if self.log_to_screen:
                logging.getLogger().warning(f"Skipping restore of the dataloader state: {message}.")
            return False

        version = state_dict.get("format_version", None)
        if version != _DATALOADER_STATE_VERSION:
            return _mismatch(f"state format version is {version} but {_DATALOADER_STATE_VERSION} is expected")

        # the sample sequence is a function of these, so a mismatch means the restored pipeline
        # would not continue the sequence which was interrupted
        for key, current in [
            ("num_shards", self.num_shards),
            ("shard_id", self.shard_id),
            ("batch_size", self.batchsize),
            ("num_steps_per_epoch", self.num_steps_per_epoch),
            ("global_seed", self.global_seed),
            ("train", self.train),
        ]:
            stored = state_dict.get(key, None)
            if stored != current:
                return _mismatch(f"{key} is {current} but the state was stored with {stored}")

        # DALI expects the serialized state as bytes again
        pipeline_checkpoint = state_dict["pipeline_checkpoint"].numpy().tobytes()

        self._build_pipeline(checkpoint=pipeline_checkpoint)

        return True

    def get_input_normalization(self):
        if self.norm_mode == "offline":
            return self.in_bias, self.in_scale
        else:
            return 0.0, 1.0

    def get_output_normalization(self):
        if self.norm_mode == "offline":
            return self.out_bias, self.out_scale
        else:
            return 0.0, 1.0

    def reset_pipeline(self):
        self.pipeline.reset()
        self.iterator.reset()

    def __len__(self):
        return self.num_steps_per_epoch

    def __iter__(self):
        for token in self.iterator:
            outputs = token[0]

            # the grid is converted after DALI, which has no op for it
            with torch.no_grad():
                inp = self.grid_converter(outputs["inp"])
                tar = self.grid_converter(outputs["tar"])
                inp_zenith = self.grid_converter(outputs["izen"]) if self.add_zenith else None
                tar_zenith = self.grid_converter(outputs["tzen"]) if self.add_zenith else None

            yield make_batch(
                inp=inp,
                inp_timestamp=outputs["itime"],
                tar=tar,
                tar_timestamp=outputs["ttime"],
                inp_zenith=inp_zenith,
                tar_zenith=tar_zenith,
            )
