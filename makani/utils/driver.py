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
import abc
import gc

from typing import Any, List, Optional, Dict, Tuple
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import logging
import wandb

# wandb.util.generate_id was the public spelling for years and is gone in recent releases (it is
# absent in 0.30, still present in 0.22). This location holds in every version from the declared
# floor of 0.13.7 through 0.30, so it is importable across the whole supported range rather than
# forcing the dependency to a narrower one.
from wandb.sdk.lib.runid import generate_id as generate_run_id

# makani dependencies
from makani.utils.YParams import YParams
from makani.utils.features import get_auxiliary_channels, is_dynamic_aux_channel, is_static_aux_channel
from makani.utils.grid_types import DEFAULT_GRID_TYPE
from makani.utils import comm
from makani.utils.benchmark import StepTimer
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.training.training_helpers import get_parameter_groups
from makani.utils.checkpoint_helpers import (
    load_checkpoint,
    gather_model_state_dict,
    scatter_model_state_dict,
    gather_optimizer_state_dict,
    scatter_optimizer_state_dict,
    prepend_prefix_to_state_dict,
    get_model_state_dict_prefix,
    supports_dataloader_state,
    gather_dataloader_state_dict,
    scatter_dataloader_state_dict,
)


class Driver(metaclass=abc.ABCMeta):
    """
    Driver class acts as abstract base class for all derived training and inference classes

    The driver class sets up default parameters, logging infrastructure, wandb infrastructure, a single eval dataset
    """

    def _log_timers(self):
        print_prefix = "    "
        if self.timers and self.log_to_screen:
            self.logger.info("Initialization time breakdown:")
            for k, v in self.timers.items():
                self.logger.info(f"{print_prefix}{k} [s]: {v:.2f}")

    def _watch_model(self, log="all"):
        """Register wandb gradient/parameter logging, unless the model will be torch.compiled.

        ``wandb.watch`` installs per-module/per-parameter hooks that compute histograms via
        ``.item()``/``.tolist()``. Under ``torch.compile`` these hooks get traced into the graph:
        every ``.item()`` is a graph break and the parameter ``name`` is specialized as a
        constant, so dynamo recompiles once per parameter until it hits ``recompile_limit`` and
        abandons compilation for that frame. The hook-based design is fundamentally incompatible
        with a single compiled graph, so we skip watching when ``jit_mode`` enables compilation.
        (If gradient/parameter histograms are needed under compile, log them out-of-graph in the
        training loop after ``backward()`` instead.)
        """
        if not self.log_to_wandb:
            return
        if self.params.get("jit_mode", "none") == "inductor":
            if self.log_to_screen:
                self.logger.info(
                    "Skipping wandb.watch under jit_mode=inductor: its hooks would be traced into "
                    "the compiled graph and trigger a recompile per parameter (recompile_limit). "
                    "Gradient/parameter histograms are disabled under torch.compile."
                )
            return
        wandb.watch(self.model, log=log)

    def __init__(self, params: YParams = None, world_rank: Optional[int] = 0, device: Optional[str] = None):
        # define timer dict
        self.timers = {}

        # update params
        self.params = self._set_default_parameters(params)

        # per-step timer for benchmarking. Disabled unless benchmark_mode is set, in which
        # case makani/benchmark.py reads the timings back out after the loop -- the training
        # and inference loops call into it unconditionally and pay nothing when it is off.
        self.step_timer = StepTimer(
            enabled=bool(self.params.get("benchmark_mode", False)),
            warmup_steps=int(self.params.get("benchmark_warmup_steps", 0)),
        )

        # set up distributed communicators, even if it is a non-distributed instance
        self.world_rank = world_rank
        self.data_parallel_rank = comm.get_rank("data")

        # set the default device
        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                self.device = torch.device("cpu")

        # set the logger
        self.log_to_screen = (
            self.params.log_to_screen if (hasattr(params, "log_to_screen") and params.log_to_screen) else False
        )
        if self.log_to_screen:
            self.logger = logging.getLogger()

        # set wandb
        self.log_to_wandb = (
            self.params.log_to_wandb if (hasattr(params, "log_to_wandb") and params.log_to_wandb) else False
        )

    def __del__(self):
        if hasattr(self, "log_to_wandb") and self.log_to_wandb:
            wandb.finish()

    def _set_dataloader(self, params, path, train=False, device=None):
        # initialize data loader
        if params.log_to_screen:
            self.logger.info(f"Using channel names: {params.channel_names}")
            self.logger.info("initializing data loader")

    def _set_default_parameters(self, params):
        """
        Routine for updating parameters internally. This is intended to be the only place where the parameters are modified
        """

        if not hasattr(params, "gradient_accumulation_steps"):
            params["gradient_accumulation_steps"] = 1

        if not hasattr(params, "log_to_screen"):
            params["log_to_screen"] = False

        if not hasattr(params, "history_normalization_mode"):
            params["history_normalization_mode"] = "none"

        if not hasattr(params, "num_visualization_workers"):
            params["num_visualization_workers"] = 1

        if not hasattr(params, "log_video"):
            params["log_video"] = 0

        if not hasattr(params, "dump_weights_and_grads"):
            params["dump_weights_and_grads"] = 0

        # how to handle checkpoints
        if not hasattr(params, "load_checkpoint"):
            params["load_checkpoint"] = "legacy"

        if not hasattr(params, "save_checkpoint"):
            params["save_checkpoint"] = "legacy"

        if not hasattr(params, "load_optimizer"):
            params["load_optimizer"] = True

        if not hasattr(params, "load_scheduler"):
            params["load_scheduler"] = True

        if not hasattr(params, "load_counters"):
            params["load_counters"] = True

        # allows resuming a run but restarting the data pipeline at the beginning of the epoch,
        # in the same way load_optimizer / load_scheduler / load_counters can be turned off
        if not hasattr(params, "load_dataloader_state"):
            params["load_dataloader_state"] = True

        if not hasattr(params, "checkpoint_num_versions"):
            params["checkpoint_num_versions"] = 3

        # whether the state of the training data pipeline is stored in and restored from the checkpoint
        if not hasattr(params, "checkpoint_dataloader_state"):
            params["checkpoint_dataloader_state"] = True

        # a real dataset declares its grid in the metadata, which is required and
        # verified there. Reaching here without one means no metadata was parsed
        # at all, which is the synthetic case, and the synthetic loader builds an
        # equiangular grid. Stated once so that no consumer has to assume it.
        if not hasattr(params, "data_grid_type"):
            params["data_grid_type"] = DEFAULT_GRID_TYPE

        return params

    def _set_data_shapes(self, params, data_shapes):
        """Copy the loader's geometry into ``params``.

        The rest of the run reads its shapes from ``params`` rather than from the
        loader, so this is where the two meet. ``data_shapes`` is a
        :class:`~makani.utils.dataloaders.data_shapes.DataShapes`, which every
        loader reports in the same form.
        """

        params.N_in_channels = len(data_shapes.in_channels)
        params.N_out_channels = len(data_shapes.out_channels)

        params.img_shape_x = data_shapes.img_shape_x
        params.img_shape_y = data_shapes.img_shape_y

        params.img_crop_shape_x = data_shapes.img_crop_shape_x
        params.img_crop_shape_y = data_shapes.img_crop_shape_y
        params.img_crop_offset_x = data_shapes.img_crop_offset_x
        params.img_crop_offset_y = data_shapes.img_crop_offset_y

        params.img_local_shape_x = data_shapes.img_local_shape_x
        params.img_local_shape_y = data_shapes.img_local_shape_y
        params.img_local_offset_x = data_shapes.img_local_offset_x
        params.img_local_offset_y = data_shapes.img_local_offset_y

        params.img_local_shape_x_resampled = data_shapes.img_local_shape_x_resampled
        params.img_local_shape_y_resampled = data_shapes.img_local_shape_y_resampled
        params.img_shape_x_resampled = data_shapes.img_shape_x_resampled
        params.img_shape_y_resampled = data_shapes.img_shape_y_resampled
        params.subsampling_factor = data_shapes.subsampling_factor

        if (params.subsampling_factor > 1) and (
            (params.img_crop_shape_x != params.img_shape_x) or (params.img_crop_shape_y != params.img_shape_y)
        ):
            raise ValueError(
                "Image cropping and data subsampling cannot be used together. Please set the crop shape to the image shape or set subsampling factor to 1."
            )

        # derived quantities
        params["N_in_predicted_channels"] = params.N_in_channels

        # sanitization:
        params["add_zenith"] = params.get("add_zenith", False)

        # input channels
        # zenith channel is appended to all the samples, so we need to do it here
        params["N_dynamic_channels"] = 0
        if params.add_zenith:
            params.N_dynamic_channels += 1

        params.n_noise_chan = 0
        if params.get("input_noise", None) is not None:
            if params.input_noise["mode"] == "concatenate":
                if "n_channels" in params.input_noise:
                    params.n_noise_chan = params.input_noise["n_channels"]
                else:
                    params.n_noise_chan = 1
        params.N_dynamic_channels += params.n_noise_chan

        # get names of additional channels; the static channel count follows from them
        params["aux_channel_names"] = get_auxiliary_channels(**params.to_dict())
        params["N_static_channels"] = len([c for c in params.aux_channel_names if is_static_aux_channel(c)])
        n_dynamic_aux = len([c for c in params.aux_channel_names if is_dynamic_aux_channel(c)])
        if n_dynamic_aux != params.N_dynamic_channels:
            raise ValueError(
                f"Auxiliary channel names {params.aux_channel_names} account for {n_dynamic_aux} dynamic channels, "
                f"but {params.N_dynamic_channels} are appended."
            )

        # update input channels withj the dynamic channels
        params.N_in_channels += params.N_dynamic_channels

        # dynamic channels are replicated at each step
        if params.n_history >= 1:
            params.N_in_channels = (params.n_history + 1) * params.N_in_channels
            params.N_in_predicted_channels *= params.n_history + 1

        # update input channels with the static channels
        params.N_in_channels += params.N_static_channels

        # target channels
        params.N_target_channels = (params.n_future + 1) * params.N_out_channels

    def _init_wandb(self, params, job_type):
        """
        Convenience routine for setting up wandb
        """

        # set up wandb logging
        if self.log_to_wandb:
            # login first:
            wandb.login(anonymous="never")

            # check if we want to resume or not
            if not params.resuming:
                # generate run id
                params["wandb_run_id"] = generate_run_id()

                # create a lost of tags:
                # paralellism:
                tags = [f"ngpu{comm.get_world_size()}", f'mp{comm.get_size("model")}', f'sp{comm.get_size("spatial")}']

                # initialize wandb
                self.wandb_run = wandb.init(
                    dir=params.wandb_dir,
                    job_type=job_type,
                    config=params,
                    name=params.wandb_name,
                    group=params.wandb_group,
                    project=params.wandb_project,
                    entity=params.wandb_entity,
                    tags=tags,
                    id=params["wandb_run_id"],
                )

                # store params in wandb folder
                params.to_yaml(os.path.join(params.wandb_dir, "wandb", "makani_restart.yaml"), overwrite=True)
            else:
                # retrieve run id from wandb config file:
                # wandb_config = YParams(os.path.join(params.wandb_dir, "wandb", "latest-run", "files", "config.yaml"), "params")
                # params["wandb_run_id"] = wandb_config["value"]["wandb_run_id"]
                tmpparams = YParams(os.path.join(params.wandb_dir, "wandb", "makani_restart.yaml"))
                params["wandb_run_id"] = tmpparams["wandb_run_id"]

                # initialize wandb: resume=must is super strict
                # but its better to fail than doing the wrong thing silently
                self.wandb_run = wandb.init(
                    dir=params.wandb_dir,
                    project=params.wandb_project,
                    entity=params.wandb_entity,
                    id=params["wandb_run_id"],
                    resume="must",
                )

            # create wandb dataset artifact
            if hasattr(params, "dataset"):
                # try using, otherwise create it:
                dataset_string = params["dataset"]["name"]
                # truncate to 128-7-1=120 characters
                if len(dataset_string) >= 120:
                    dataset_string = dataset_string[:120]
                dataset_tag = dataset_string + ":latest"

                if not wandb.run.offline:
                    api = wandb.Api()
                    if api.artifact_collection_exists(dataset_string, type="dataset"):
                        # try using existing dataset
                        self.wandb_dataset = wandb.use_artifact(dataset_tag, type="dataset")
                        print(f"Using dataset artifact {dataset_tag}")
                    else:
                        # create new one if it does not exist
                        self.wandb_dataset = wandb.Artifact(
                            name=dataset_string, description=params["dataset"]["description"], type="dataset"
                        )
                        self.wandb_dataset.add_file(params["dataset"]["metadata_file"], name="metadata")
                        wandb.log_artifact(self.wandb_dataset)
                        print(f"Creating artifact {dataset_string}")

            # create data normalization artifact
            if hasattr(params, "normalization"):

                if hasattr(params, "dataset"):
                    norm_string = params["dataset"]["name"] + "_"
                else:
                    norm_string = ""

                # generate name string
                if isinstance(params.normalization, dict):
                    norm_string += "zscore_minmax_" + "_".join([f"{k}-{v}" for k, v in params.normalization.items()])
                else:
                    norm_string += params.normalization
                # truncate
                if len(norm_string) >= 120:
                    norm_string = norm_string[:120]
                norm_tag = norm_string + ":latest"

                if not wandb.run.offline:
                    api = wandb.Api()
                    if api.artifact_collection_exists(norm_string, type="dataset_normalization"):
                        # try using existing normalization
                        self.wandb_normalization = wandb.use_artifact(norm_tag, type="dataset_normalization")
                        print(f"Using normalization artifact {norm_tag}")
                    else:
                        # create normalization artifact
                        self.wandb_normalization = wandb.Artifact(
                            name=norm_string, description="data normalization", type="dataset_normalization"
                        )
                        bias, scale = get_data_normalization(params)
                        # filter only used channels
                        bias = bias.flatten()[params.in_channels]
                        scale = scale.flatten()[params.in_channels]
                        data = np.stack([scale, bias], axis=0).tolist()
                        data[0].insert(0, "scale")
                        data[1].insert(0, "bias")
                        print(len(data[0]), len(data[1]))
                        # create columns
                        columns = ["type"] + params.channel_names
                        # create table
                        tab = wandb.Table(columns=columns, data=data)
                        self.wandb_normalization.add(tab, name="data")
                        wandb.log_artifact(self.wandb_normalization)
                        print(f"Creating artifact {norm_string}")

    def get_train_dataloader_state(self) -> Optional[List[Dict[str, Any]]]:
        """
        Collect the state of the training data pipeline for checkpointing, on all ranks.

        Only the training pipeline is checkpointed: the validation pipeline is restarted from
        the beginning of the validation set in every epoch anyway. Returns None if dataloader
        checkpointing is disabled or unsupported by the loader in use.
        """
        if not self.params.get("checkpoint_dataloader_state", True):
            return None

        return self.gather_dataloader_state(getattr(self, "train_dataloader", None))

    def get_train_dataloader_for_restore(self) -> Optional[Any]:
        """
        The training dataloader if its state should be restored from the checkpoint, else None.

        Returns None when either the feature is turned off entirely
        (``checkpoint_dataloader_state``) or only the restore side is
        (``load_dataloader_state``), which resumes a run with a data pipeline starting at the
        beginning of the epoch. Note that finetuning does not go through here at all: the
        trainers only hand the dataloader to the restore when resuming.
        """
        if not self.params.get("checkpoint_dataloader_state", True):
            return None

        if not self.params.get("load_dataloader_state", True):
            return None

        dataloader = getattr(self, "train_dataloader", None)
        if not supports_dataloader_state(dataloader):
            return None

        return dataloader

    @staticmethod
    def gather_dataloader_state(dataloader: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Routine for collecting the dataloader state of all data-parallel ranks.

        This has to be called by every rank, in contrast to :meth:`save_checkpoint`, which is
        only called by rank 0 of the data-parallel comm. The returned list is what is handed to
        :meth:`save_checkpoint` as ``dataloader_state``. Returns None for dataloaders which do
        not support checkpointing.
        """
        return gather_dataloader_state_dict(dataloader)

    @staticmethod
    def restore_from_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        dataloader: Optional[Any] = None,
        checkpoint_mode: str = "legacy",
        strict: bool = True,
        log_to_screen: bool = False,
    ):
        """
        Routine for restoring a checkpoint from a path.
        """
        with torch.no_grad():
            if checkpoint_mode == "legacy":
                # legacy mode
                Driver._restore_checkpoint_legacy(
                    checkpoint_path,
                    model,
                    loss,
                    optimizer,
                    scheduler,
                    counters,
                    dataloader,
                    strict=strict,
                    log_to_screen=log_to_screen,
                )
            elif checkpoint_mode == "flexible":
                # new flexible mode allows to load models in arbitrary model-parallel configurations
                Driver._restore_checkpoint_flexible(
                    checkpoint_path,
                    model,
                    loss,
                    optimizer,
                    scheduler,
                    counters,
                    dataloader,
                    strict=strict,
                    log_to_screen=log_to_screen,
                )
            else:
                raise ValueError(f"Unknown checkoint mode {checkpoint_mode}.")

        # clean up
        gc.collect()

        return

    @staticmethod
    def _restore_dataloader_state(
        checkpoint: Dict[str, Any], dataloader: Any, strict: bool = True, log_to_screen: bool = False
    ):
        """
        Restore the state of the data pipeline from an already loaded checkpoint.
        """

        if not supports_dataloader_state(dataloader):
            raise ValueError(
                "Error, the dataloader state was requested to be restored, but the dataloader does not "
                "support checkpointing. Only the DALI dataloaders with checkpoint_dataloader_state "
                "set to True do."
            )

        if "dataloader_state" not in checkpoint:
            # checkpoints written before dataloader checkpointing was introduced, and checkpoints
            # written with a dataloader which does not support it, do not carry this entry. this
            # is not treated as an error even in strict mode, so that older checkpoints stay
            # resumable; the data pipeline then simply starts at the beginning of the epoch
            if log_to_screen:
                logging.getLogger().warning(
                    "The checkpoint does not contain a dataloader state, so the data pipeline starts at the "
                    "beginning of the epoch instead of resuming where the checkpoint was taken."
                )
            return

        try:
            state_dict = scatter_dataloader_state_dict(checkpoint["dataloader_state"])
        except ValueError as err:
            if strict:
                raise
            if log_to_screen:
                logging.getLogger().warning(f"Skipping restore of the dataloader state: {err}")
            return

        dataloader.load_state_dict(state_dict, strict=strict)

        return

    @staticmethod
    def _restore_checkpoint_legacy(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        dataloader: Optional[Any] = None,
        strict: bool = True,
        log_to_screen: bool = False,
        validate_comms: bool = True,
    ):
        checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))
        checkpoint = load_checkpoint(checkpoint_fname, map_location="cpu")

        # check compatibility of the comm grid stored inside the file
        if validate_comms:
            # load the comm dict
            if "comm_grid" not in checkpoint:
                from warnings import warn

                warn(
                    "It is highly recommended to upgrade the checkpointing format to include model parallel comm grid information, so that a correct restoring can be guaranteed. This can be achieved by loading and saving a model with a newer version of makani. In future versions of makani, this warning will become an error.",
                    DeprecationWarning,
                )
            else:
                comm_dict = checkpoint["comm_grid"]
                comm_names = comm.get_model_comm_names()
                for cname in comm_names:
                    # check comm
                    if cname not in comm_dict.keys():
                        raise RuntimeError(
                            f"Error, communicator name {cname} not found in communicator information stored in file, but present in the current comm table."
                        )
                    # check size
                    if comm.get_size(cname) != comm_dict[cname]["size"]:
                        raise RuntimeError(
                            f"Error, communicator {cname} has size {comm.get_size(cname)}, but expected size {comm_dict[cname]['size']}"
                        )
                    # check rank
                    if comm.get_rank(cname) != comm_dict[cname]["rank"]:
                        raise RuntimeError(
                            f"Error, communicator {cname} rank {comm.get_rank(cname)} is trying to load a file from rank {comm_dict[cname]['rank']}"
                        )

        # if all those test pass, we are good to go
        # this is reworked to avoid loading modules related to the SHT
        state_dict = checkpoint["model_state"]
        prefix = get_model_state_dict_prefix(model)
        if prefix:
            prepend_prefix_to_state_dict(state_dict, prefix)

        # load state dict
        model.load_state_dict(state_dict, strict=strict)

        # the loss is also restored in the case that it has a state
        if loss is not None:
            loss.load_state_dict(checkpoint["loss_state_dict"])

        # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if counters is not None:
            counters["iters"] = checkpoint["iters"]
            counters["start_epoch"] = checkpoint["epoch"]

        if dataloader is not None:
            Driver._restore_dataloader_state(checkpoint, dataloader, strict=strict, log_to_screen=log_to_screen)

        return

    @staticmethod
    def _restore_checkpoint_flexible(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        dataloader: Optional[Any] = None,
        strict: bool = True,
        log_to_screen: bool = False,
    ):
        # when loading the weights in flexble mode we exclusively use mp_rank=0 and load them onto the cpu
        checkpoint_fname = checkpoint_path.format(mp_rank=0)
        checkpoint = load_checkpoint(checkpoint_fname, map_location="cpu")

        # this is reworked to avoid loading modules related to the SHT
        state_dict = checkpoint["model_state"]

        prefix = get_model_state_dict_prefix(model)
        if prefix:
            prepend_prefix_to_state_dict(state_dict, prefix)

        if comm.get_size("model") > 1:
            state_dict = scatter_model_state_dict(model, state_dict, strict)

        # load state dict
        model.load_state_dict(state_dict, strict=strict)

        # the loss is also restored in the case that it has a state
        if loss is not None:
            loss.load_state_dict(checkpoint["loss_state_dict"])

        # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
        if optimizer is not None:
            if comm.get_size("model") > 1:
                checkpoint["optimizer_state_dict"] = scatter_optimizer_state_dict(
                    model, optimizer, checkpoint["optimizer_state_dict"]
                )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if counters is not None:
            counters["iters"] = checkpoint["iters"]
            counters["start_epoch"] = checkpoint["epoch"]

        if dataloader is not None:
            Driver._restore_dataloader_state(checkpoint, dataloader, strict=strict, log_to_screen=log_to_screen)

        return

    @staticmethod
    def save_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        dataloader_state: Optional[List[Dict[str, Any]]] = None,
        checkpoint_mode: str = "legacy",
    ):
        """
        Save out checkpoint

        ``dataloader_state`` is the list of per-data-parallel-rank states returned by
        :meth:`gather_dataloader_state`, which has to be collected by all ranks before this
        routine is entered by the checkpoint-writing rank.
        """
        with torch.no_grad():
            # legacy mode
            if checkpoint_mode == "legacy":
                Driver._save_checkpoint_legacy(
                    checkpoint_path, model, loss, optimizer, scheduler, counters, dataloader_state
                )
            elif checkpoint_mode == "flexible":
                Driver._save_checkpoint_flexible(
                    checkpoint_path, model, loss, optimizer, scheduler, counters, dataloader_state
                )
            else:
                raise ValueError(f"Unknown checkoint mode {checkpoint_mode}.")

        # clean up
        gc.collect()

        return

    @staticmethod
    def _save_checkpoint_legacy(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        dataloader_state: Optional[List[Dict[str, Any]]] = None,
    ):
        # maybe the logic regarding the mp rank should be moved to somewhere else?
        checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))

        # attach sharding information to model state:
        state_dict = model.state_dict()

        # strip all wrapper prefixes (torch.compile -> "_orig_mod.", DDP -> "module.")
        # so checkpoints are always stored in canonical form without wrapper-specific keys
        prefix = get_model_state_dict_prefix(model)
        if prefix:
            nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix)

        # check for model parallelism
        # Build a name-keyed map using the same prefix that was stripped from the
        # state dict, so lookups are safe regardless of parameter iteration order.
        # removeprefix is a no-op when prefix is "", so no guard needed.
        param_map = {name.removeprefix(prefix): param for name, param in model.named_parameters()}
        for sk, tensor in state_dict.items():
            param = param_map.get(sk)
            if param is not None and hasattr(param, "sharded_dims_mp"):
                tensor.sharded_dims_mp = param.sharded_dims_mp

        # add model state dict to store dict
        store_dict = {"model_state": state_dict}

        # comm infrastructure:
        comm_names = comm.get_model_comm_names()
        comm_dict = OrderedDict()
        for cname in comm_names:
            rank = comm.get_rank(cname)
            size = comm.get_size(cname)
            comm_dict[cname] = {"size": size, "rank": rank}
        store_dict["comm_grid"] = comm_dict

        if loss is not None:
            store_dict["loss_state_dict"] = loss.state_dict()

        if optimizer is not None:
            store_dict["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            store_dict["scheduler_state_dict"] = scheduler.state_dict()

        if counters is not None:
            store_dict["iters"] = counters["iters"]
            store_dict["epoch"] = counters["epoch"]

        # state of the training data pipeline, one entry per data-parallel rank
        if dataloader_state is not None:
            store_dict["dataloader_state"] = dataloader_state

        torch.save(store_dict, checkpoint_fname)

        return

    @staticmethod
    def _save_checkpoint_flexible(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        dataloader_state: Optional[List[Dict[str, Any]]] = None,
    ):
        # checkpoint name
        checkpoint_fname = checkpoint_path.format(mp_rank=0)

        # iterate over parameters and gather them from the ranks
        if comm.get_size("model") > 1:
            state_dict = gather_model_state_dict(model)
        else:
            state_dict = model.state_dict()

        # strip all wrapper prefixes (torch.compile -> "_orig_mod.", DDP -> "module.")
        # so checkpoints are always stored in canonical form without wrapper-specific keys
        prefix = get_model_state_dict_prefix(model)
        if prefix:
            nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix)

        # attach sharding metadata to state_dict tensors (same as legacy) so torch.save preserves it
        for name, param in model.named_parameters():
            if name in state_dict and hasattr(param, "sharded_dims_mp"):
                state_dict[name].sharded_dims_mp = param.sharded_dims_mp
        for name, buf in model.named_buffers():
            if name in state_dict and hasattr(buf, "sharded_dims_mp"):
                state_dict[name].sharded_dims_mp = buf.sharded_dims_mp

        store_dict = {"model_state": state_dict}

        if loss is not None:
            store_dict["loss_state_dict"] = loss.state_dict()

        if optimizer is not None:
            if comm.get_size("model") > 1:
                store_dict["optimizer_state_dict"] = gather_optimizer_state_dict(model, optimizer)
            else:
                store_dict["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            store_dict["scheduler_state_dict"] = scheduler.state_dict()

        if counters is not None:
            store_dict["iters"] = counters["iters"]
            store_dict["epoch"] = counters["epoch"]

        # state of the training data pipeline, one entry per data-parallel rank
        if dataloader_state is not None:
            store_dict["dataloader_state"] = dataloader_state

        # in flexible mode only rank 0 needs to save the data to disk
        if comm.get_world_rank() == 0:
            torch.save(store_dict, checkpoint_fname)

        return

    @staticmethod
    def dump_weights_and_grads(weights_and_grads_path: str, model: nn.Module, step: int = 0):
        """
        Helper routine intended for debugging purposes to dump weights and grads
        """

        mp_rank = comm.get_rank("model")
        weights_and_grads_fname = os.path.join(weights_and_grads_path, f"weights_and_grads_step{step}_mp{mp_rank}.tar")

        weights_dict = {k: v for k, v in model.named_parameters()}
        grad_dict = {k: v.grad for k, v in model.named_parameters()}

        store_dict = {"step": step, "grads": grad_dict, "weights": weights_dict}
        torch.save(store_dict, weights_and_grads_fname)

    # TODO: would be nice to convert this to static methods
    def get_optimizer(self, model, params):
        """
        Convenience routine for setting up the optimizer
        """

        # optimizer setup
        betas = (params.optimizer_beta1, params.optimizer_beta2)
        # build parameter groups according to the weight-decay mode ("full" decays
        # everything; "transformer" excludes biases and norm affine params). Each group
        # carries its own weight_decay, so it is no longer passed at the optimizer level.
        weight_decay = params.get("weight_decay", 0)
        weight_decay_mode = params.get("weight_decay_mode", "full")
        all_parameters = get_parameter_groups(model, weight_decay, weight_decay_mode)
        if params.optimizer_type == "Adam":
            if self.log_to_screen:
                self.logger.info(f"using Adam optimizer (weight_decay_mode={weight_decay_mode})")
            optimizer = optim.Adam(
                all_parameters,
                lr=params.get("lr", 1e-3),
                betas=betas,
                eps=params.get("optimizer_eps", 1e-8),
                foreach=True,
            )
        elif params.optimizer_type == "AdamW":
            if self.log_to_screen:
                self.logger.info(f"using AdamW optimizer (weight_decay_mode={weight_decay_mode})")
            optimizer = optim.AdamW(
                all_parameters,
                lr=params.get("lr", 1e-3),
                betas=betas,
                eps=params.get("optimizer_eps", 1e-8),
                foreach=True,
            )
        elif params.optimizer_type == "SGD":
            if self.log_to_screen:
                self.logger.info(f"using SGD optimizer (weight_decay_mode={weight_decay_mode})")
            optimizer = optim.SGD(
                all_parameters,
                lr=params.get("lr", 1e-3),
                momentum=params.get("momentum", 0),
                nesterov=params.get("nesterov", False),
                foreach=True,
            )
        elif params.optimizer_type == "SIRFShampoo":
            if self.log_to_screen:
                self.logger.info("using SIRFShampoo optimizer")
            from sirfshampoo import SIRFShampoo

            optimizer = SIRFShampoo(model, lr=params.get("lr", 1e-3))
        else:
            raise ValueError(f"Unknown optimizer type {params.optimizer_type}")

        return optimizer

    # TODO: would be nice to convert this to static methods
    def get_scheduler(self, optimizer, params):
        """Convenience routine for setting up the scheduler"""

        if params.scheduler == "ReduceLROnPlateau":
            if not hasattr(params, "scheduler_mode"):
                params["scheduler_mode"] = "min"
            if params.get("skip_validation", False):
                raise ValueError("Error, you cannot skip validation when using ReduceLROnPlateau scheduler.")
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=params.scheduler_factor,
                patience=params.scheduler_patience,
                mode=params.scheduler_mode,
            )
        elif params.scheduler == "StepLR":
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=params.scheduler_step_size, gamma=params.scheduler_gamma
            )
        elif params.scheduler == "CosineAnnealingLR":
            if not hasattr(params, "scheduler_min_lr"):
                params["scheduler_min_lr"] = 0.0
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.scheduler_T_max, eta_min=params.scheduler_min_lr
            )
        elif params.scheduler == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(
                optimizer, max_lr=params.lr, total_steps=params.scheduler_T_max, steps_per_epoch=1
            )
        elif params.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.get("scheduler_T_0", 10),
                T_mult=params.get("scheduler_T_mult", 1),
                eta_min=params.get("scheduler_min_lr", 0.0),
            )
        else:
            scheduler = None

        # warmup scheduler
        if params.lr_warmup_steps > 0:
            if params.scheduler == "ReduceLROnPlateau":
                raise NotImplementedError("Error, warmup scheduler not implemented for ReduceLROnPlateau scheduler")
            warmup_scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=params.get("lr_start", 1e-5),
                end_factor=1.0,
                total_iters=params.get("lr_warmup_steps", 0),
            )

            scheduler = lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], milestones=[params.get("lr_warmup_steps", 0)]
            )

        return scheduler

    def init_visualizer(
        self,
        params: YParams,
        lat_lon_global: Tuple[np.ndarray, np.ndarray],
        out_bias: torch.Tensor,
        out_scale: torch.Tensor,
        device: torch.device,
    ):
        """
        Initialize the visualizer
        """
        from makani.utils import visualize

        # Plot entries name an op from visualize.PLOT_OPS and reference channels
        # symbolically. The visualizer resolves the names against channel_names,
        # ships only the referenced channels to the renderer subprocesses, and
        # rewrites the channel lists to index into the stripped tensor.
        cnames = params.channel_names
        plot_list = []

        if "u10m" in cnames and "v10m" in cnames:
            plot_list.append(
                {
                    "name": "windspeed_uv10",
                    "op": "magnitude",
                    "channels": ["u10m", "v10m"],
                    "diverging": False,
                }
            )

        if "z500" in cnames:
            plot_list.append(
                {
                    "name": "geopotential_z500",
                    "op": "select",
                    "channels": ["z500"],
                    "diverging": False,
                }
            )

        if "q100" in cnames:
            plot_list.append(
                {
                    "name": "specific_humidity_q100",
                    "op": "select",
                    "channels": ["q100"],
                    "diverging": False,
                }
            )

        if plot_list:
            visualizer = visualize.VisualizationWrapper(
                params.log_to_wandb,
                path=None,
                prefix=None,
                plot_list=plot_list,
                channel_names=cnames,
                lat=np.deg2rad(lat_lon_global[0]),
                lon=np.deg2rad(lat_lon_global[1]) - np.pi,
                scale=out_scale[0, ...],
                bias=out_bias[0, ...],
                num_workers=params.num_visualization_workers,
            )
            # allocate pinned tensors for faster copy:
            if device.type == "cuda":
                visualizer.stream = torch.Stream(device="cuda")
                pin_memory = True
            else:
                visualizer.stream = None
                pin_memory = False

            visualizer.prediction_cpu = torch.empty(
                (
                    (params.N_target_channels // (params.n_future + 1)),
                    params.img_shape_x_resampled,
                    params.img_shape_y_resampled,
                ),
                device="cpu",
                pin_memory=pin_memory,
            )
            visualizer.target_cpu = torch.empty(
                (
                    (params.N_target_channels // (params.n_future + 1)),
                    params.img_shape_x_resampled,
                    params.img_shape_y_resampled,
                ),
                device="cpu",
                pin_memory=pin_memory,
            )
        else:
            visualizer = None

        return visualizer

    def _setup_visualizer(self, out_bias: torch.Tensor, out_scale: torch.Tensor):
        """Initialize self.visualizer on rank 0; set to None on all other ranks."""
        if self.world_rank == 0:
            self.visualizer = self.init_visualizer(self.params, self.lat_lon_global, out_bias, out_scale, self.device)
            if self.visualizer is None:
                self.logger.info("No channels to visualize, skipping visualization.")
        else:
            self.visualizer = None

    def _visualize_step(
        self, pred_gather: torch.Tensor, targ_gather: torch.Tensor, eval_steps: int, idt: int, ndt: Optional[int] = None
    ):
        """Copy gathered tensors to the visualizer and queue a frame. No-op when self.visualizer is None."""
        if self.visualizer is None:
            return
        if self.visualizer.stream is not None:
            self.visualizer.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.visualizer.stream):
                self.visualizer.prediction_cpu.copy_(pred_gather, non_blocking=True)
                self.visualizer.target_cpu.copy_(targ_gather, non_blocking=True)
            self.visualizer.stream.synchronize()
        else:
            self.visualizer.prediction_cpu.copy_(pred_gather)
            self.visualizer.target_cpu.copy_(targ_gather)
        pred_cpu = self.visualizer.prediction_cpu.to(torch.float32).numpy()
        targ_cpu = self.visualizer.target_cpu.to(torch.float32).numpy()
        progress = idt / max(ndt - 1, 1) if ndt is not None else None
        self.visualizer.add(f"step{eval_steps}_time{str(idt).zfill(3)}", pred_cpu, targ_cpu, progress=progress)
