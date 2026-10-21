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

import time
import logging
from typing import Optional, Union, List

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.utils.data as tud

from makani.utils.driver import Driver
from makani.utils import LossHandler, MetricsHandler
from makani.utils.functions import expand_ensemble
from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset
from makani.utils.dataloaders.batch import (
    batch_to_device,
    INPUT,
    INPUT_DYNAMIC_FEATURES,
    INPUT_TIMESTAMP,
)
from makani.utils.dataloaders.rollout import Rollout
from makani.utils.dataloaders.data_helpers import get_date_from_timestamp, get_data_normalization, get_climatology
from makani.utils.YParams import YParams

from makani.models import model_registry

# distributed computing stuff
from makani.utils import comm
from makani.utils.precision import AutocastManager
from makani.utils.dataloaders.data_helpers import get_date_from_string

# inference specific stuff
from makani.utils.inference.helpers import (
    split_list,
    SortedIndexSampler,
    translate_date_sampler_to_timedelta_sampler,
    compute_inference_range,
)
from makani.utils.inference.rollout_buffer import (
    RolloutBuffer,
    TemporalAverageBuffer,
    SpectrumAverageBuffer,
    ZonalSpectrumAverageBuffer,
)

# checkpoint helpers
from makani.utils.checkpoint_helpers import get_latest_checkpoint_version
from makani.utils.training.training_helpers import get_memory_usage


class Inferencer(Driver):
    """
    Inferencer class holding all the necessary information to perform inference. Design is similar to Trainer, however only keeping the necessary information.

    Parameters
    ============
    params : YParams
        Parameter object for initialization
    world_rank : int
        Rank in the world communicator group
    device : str
        Device on which to perform the inference
    """

    def __init__(self, params: Optional[YParams] = None, world_rank: Optional[int] = 0, device: Optional[str] = None):
        # init the trainer
        super().__init__(params, world_rank, device)

        # init wandb
        if self.log_to_wandb:
            self._init_wandb(self.params, job_type="inference")

        # set up mixed precision (amp dtype + optional transformer-engine fp8 recipe).
        # the manager parses the mode once and hands out a single nested autocast cm.
        amp_mode = self.params.amp_mode if hasattr(self.params, "amp_mode") else "none"
        self.autocast = AutocastManager(amp_mode, device_type="cuda", fp8_group=comm.get_group("data"))
        self.amp_dtype = self.autocast.amp_dtype
        self.amp_enabled = self.autocast.amp_enabled
        if self.amp_enabled and self.log_to_screen:
            self.logger.info(f"Enabling automatic mixed precision in '{amp_mode}'.")

        # resuming needs is set to False so loading checkpoints does not attempt to set the optimizer state
        self.params["resuming"] = False

        # data loader
        if self.params.log_to_screen:
            self.logger.info("initializing data loader")

        # Only the multifiles and the synthetic dataloaders can serve inference: both address
        # samples by index and by time, which is how the rollout schedule is built. Anything else
        # is overridden to multifiles.
        #
        # ..note::
        #     The second attribute check here used to read ``hasattr(self.params, "amp")`` while
        #     assigning ``enable_synthetic_data``, which disabled synthetic data for every config
        #     that did not happen to carry an ``amp`` key -- i.e. effectively always. Whatever it
        #     was meant to guard, it was not that.
        if not hasattr(self.params, "enable_synthetic_data"):
            self.params["enable_synthetic_data"] = False

        if not self.params.enable_synthetic_data:
            self.params["multifiles"] = True

        # the file path is taken from inf_data_path to perform inference on the out of sample dataset
        self.valid_dataloader, self.valid_data_shapes, _ = get_dataloader(
            self.params, self.params.inf_data_path, mode="inference", device=self.device
        )
        self._set_data_shapes(self.params, self.valid_data_shapes)

        # inference addresses samples by index and by time, which is the dataset's
        # own business rather than anything the shapes describe
        self.valid_dataset = self.valid_dataloader.dataset

        if self.log_to_screen:
            self.logger.info("data loader initialized")

        self.mask_dataset = None
        if self.params.get("mask_file", None) is not None:
            self.mask_dataset = MultifilesDataset(
                location=self.params.get("mask_file"),
                dt=1,  # not relevant since we are looking up samples directly
                in_channels=self.params.get("in_channels"),
                out_channels=self.params.get("out_channels"),
                n_history=0,  # since we only use it for the output
                n_future=0,
                add_zenith=0,
                data_grid_type=params.data_grid_type,
                model_grid_type=params.get("model_grid_type", "equiangular"),
                crop_size=(params.get("crop_size_x", None), params.get("crop_size_y", None)),
                crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
                relative_timestamp=True,
                return_target=False,
                dataset_name="fields",
                enable_s3=params.get("enable_s3", False),
                io_grid=params.get("io_grid", [1, 1, 1]),
                io_rank=params.get("io_rank", [0, 0, 0]),
                enable_logging=False,
            )

            # we need grid quadrature as well in this case
            from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

            quadrature_rule = grid_to_quadrature_rule(params.get("model_grid_type", "equiangular"))
            self.quadrature = GridQuadrature(
                quadrature_rule,
                img_shape=self.mask_dataset.img_shape,
                crop_shape=self.mask_dataset.crop_size,
                crop_offset=self.mask_dataset.crop_anchor,
                normalize=True,
                distributed=(comm.get_size("spatial") > 1),
            ).to(self.device)

        self.climatology_dataset = None
        if self.params.get("climatology_file", None) is not None:
            bias, scale = get_data_normalization(self.params)
            self.climatology_dataset = MultifilesDataset(
                location=self.params.get("climatology_file"),
                dt=1,  # not relevant since we are looking up samples directly
                in_channels=self.params.get("in_channels"),
                out_channels=self.params.get("out_channels"),
                n_history=0,  # since we only use it for the output
                n_future=0,
                add_zenith=0,
                data_grid_type=params.data_grid_type,
                model_grid_type=params.get("model_grid_type", "equiangular"),
                bias=bias,  # we subtract the bias to avoid subtracting too big numbers
                scale=scale,  # we need to set that to make sure the climatology is properly scaled
                crop_size=(params.get("crop_size_x", None), params.get("crop_size_y", None)),
                crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
                relative_timestamp=True,
                return_target=False,
                dataset_name="fields",
                enable_s3=params.get("enable_s3", False),
                io_grid=params.get("io_grid", [1, 1, 1]),
                io_rank=params.get("io_rank", [0, 0, 0]),
                enable_logging=False,
            )

        # get model
        self.model = model_registry.get_model(self.params).to(self.device)
        self.preprocessor = self.model.preprocessor

        # print model
        if self.world_rank == 0:
            print(self.model)

        # Performance work -- benchmarking the rollout, or the output writer with O_DIRECT/GDS --
        # does not depend on the weights being trained, and requiring a checkpoint would mean
        # staging one just to measure IO. Skipping the restore is opt-in and loud: scoring random
        # weights while believing them trained would be a far worse failure than a missing file.
        self.random_weights = self.params.get("allow_random_weights", False)

        if self.random_weights:
            logging.warning(
                "Skipping the checkpoint restore: allow_random_weights is set, so this model carries "
                "randomly initialized weights. Any forecast or score produced by this run is meaningless."
            )
            self.checkpoint_version_current = 0
        else:
            if self.log_to_screen:
                self.logger.info(
                    f"Loading pretrained checkpoint {self.params.checkpoint_path} in {self.params.load_checkpoint} mode"
                )

            # restore from checkpoint
            checkpoint_path = self.params.checkpoint_path
            self.checkpoint_version_current = get_latest_checkpoint_version(checkpoint_path)
            checkpoint_path = checkpoint_path.format(
                checkpoint_version=self.checkpoint_version_current, mp_rank="{mp_rank}"
            )

            self.restore_from_checkpoint(
                checkpoint_path,
                model=self.model,
                checkpoint_mode=self.params.load_checkpoint,
                strict=self.params.get("strict_restore", True),
            )

        # loss handler
        self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()
        self.preprocessor.eval()

    # shorthand for inference range running over the full dataset
    def inference_epoch(
        self,
        rollout_steps: int,
        dhours: int,
        compute_metrics: bool = False,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        bias_file: Optional[str] = None,
        spectrum_file: Optional[str] = None,
        zonal_spectrum_file: Optional[str] = None,
        wb2_compatible: Optional[bool] = False,
        enable_odirect: bool = False,
        odirect_alignment: int = 0,
        enable_gds: bool = False,
        profiler=None,
    ):
        """
        Runs the model in autoregressive inference mode on the entire validation dataset. Computes metrics and scores the model.
        """

        # get the number of maximum samples
        num_samples = len(self.valid_dataset)

        # split the samples across ranks
        samples_local = split_list(list(range(0, num_samples)), comm.get_size("batch"))[comm.get_rank("batch")]

        # distribute samples across all ranks
        start = min(samples_local)
        end = max(samples_local) + 1

        logs = self.inference_range(
            start,
            end,
            1,
            rollout_steps=rollout_steps,
            dhours=dhours,
            batch_size=self.params.batch_size,
            compute_metrics=compute_metrics,
            output_channels=output_channels,
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            bias_file=bias_file,
            spectrum_file=spectrum_file,
            zonal_spectrum_file=zonal_spectrum_file,
            wb2_compatible=wb2_compatible,
            enable_odirect=enable_odirect,
            odirect_alignment=odirect_alignment,
            enable_gds=enable_gds,
            profiler=profiler,
        )

        return logs

    # routine which allows to run over a given range
    def inference_range(
        self,
        start: int,
        end: int,
        step: int,
        rollout_steps: int,
        dhours: int,
        batch_size: int,
        compute_metrics: bool = False,
        metrics_file: Optional[str] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        bias_file: Optional[str] = None,
        spectrum_file: Optional[str] = None,
        zonal_spectrum_file: Optional[str] = None,
        wb2_compatible: Optional[bool] = False,
        enable_odirect: bool = False,
        odirect_alignment: int = 0,
        enable_gds: bool = False,
        profiler=None,
    ):

        # create index list for range
        indices = list(range(start, end, step))

        # run inference on indexlist
        logs = self.inference_indexlist(
            indices,
            rollout_steps=rollout_steps,
            dhours=dhours,
            batch_size=batch_size,
            compute_metrics=compute_metrics,
            metrics_file=metrics_file,
            output_channels=output_channels,
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            bias_file=bias_file,
            spectrum_file=spectrum_file,
            zonal_spectrum_file=zonal_spectrum_file,
            wb2_compatible=wb2_compatible,
            enable_odirect=enable_odirect,
            odirect_alignment=odirect_alignment,
            enable_gds=enable_gds,
            profiler=profiler,
        )

        return logs

    def inference_indexlist(
        self,
        indices: Union[List[int], torch.Tensor],
        rollout_steps: int,
        dhours: int,
        batch_size: int,
        compute_metrics: bool = False,
        metrics_file: Optional[str] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        bias_file: Optional[str] = None,
        spectrum_file: Optional[str] = None,
        zonal_spectrum_file: Optional[str] = None,
        wb2_compatible: Optional[bool] = False,
        enable_odirect: bool = False,
        odirect_alignment: int = 0,
        enable_gds: bool = False,
        profiler=None,
    ):

        # initialize metrics handler if requested
        if compute_metrics:
            # metrics handler
            if self.climatology_dataset is None:
                clim = get_climatology(self.params)
                clim = torch.from_numpy(clim).to(torch.float32)
            else:
                clim = None
            metric_channels = self.params.channel_names
            metrics = MetricsHandler(
                params=self.params,
                climatology=clim,
                num_rollout_steps=rollout_steps,
                device=self.device,
                l1_var_names=metric_channels,
                rmse_var_names=metric_channels,
                acc_var_names=metric_channels,
                crps_var_names=metric_channels,
                spread_var_names=metric_channels,
                ssr_var_names=metric_channels,
                rh_var_names=metric_channels,
                wb2_compatible=wb2_compatible,
            )
            metrics.initialize_buffers()
        else:
            metrics = None

        # get scaling factors for the output
        bias, scale = get_data_normalization(self.params)
        bias = torch.from_numpy(bias[:, self.params.out_channels, ...]).to(dtype=torch.float32)
        scale = torch.from_numpy(scale[:, self.params.out_channels, ...]).to(dtype=torch.float32)

        # get shapes
        img_shape = [self.params.img_shape_x, self.params.img_shape_y]
        local_shape = [
            self.params.get("img_local_shape_x", img_shape[0]),
            self.params.get("img_local_shape_y", img_shape[1]),
        ]
        local_offset = [self.params.get("img_local_offset_x", 0), self.params.get("img_local_offset_y", 0)]

        if output_file is not None:
            # intiialize the rollout buffer
            rollout_buffer = RolloutBuffer(
                num_samples=len(indices),
                batch_size=batch_size,
                num_rollout_steps=rollout_steps,
                rollout_dt=self.params.dt * dhours,
                ensemble_size=self.params.local_ensemble_size,
                img_shape=img_shape,
                local_shape=local_shape,
                local_offset=local_offset,
                channel_names=self.params.channel_names,
                lat_lon=(self.params.lat, self.params.lon),
                device=self.device,
                scale=scale,
                bias=bias,
                output_file=output_file,
                output_channels=output_channels,
                output_memory_buffer_size=output_memory_buffer_size,
                # GDS writes straight from GPU memory, so the staging buffer has to live there too
                buffer_device=self.device if enable_gds else torch.device("cpu"),
                enable_odirect=enable_odirect,
                odirect_alignment=odirect_alignment,
                enable_gds=enable_gds,
            )
        else:
            rollout_buffer = None

        if bias_file is not None:
            # we use all channels for the bias computation
            bias_channels = output_channels if output_channels else self.params.channel_names
            # intiialize buffer for bias computation
            # set bias to none since we only pass the differences
            bias_buffer = TemporalAverageBuffer(
                num_rollout_steps=rollout_steps,
                rollout_dt=self.params.dt,
                img_shape=img_shape,
                local_shape=local_shape,
                local_offset=local_offset,
                channel_names=self.params.channel_names,
                lat_lon=(self.params.lat, self.params.lon),
                device=self.device,
                scale=scale,
                bias=None,
                output_file=bias_file,
                output_channels=bias_channels,
            )
        else:
            bias_buffer = None

        if spectrum_file is not None:
            # we use all channels for the spectral computation
            spectrum_channels = output_channels if output_channels else self.params.channel_names
            spectrum_buffer = SpectrumAverageBuffer(
                num_rollout_steps=rollout_steps,
                rollout_dt=self.params.dt,
                img_shape=img_shape,
                ensemble_size=self.params.local_ensemble_size,
                grid_type=self.params.model_grid_type,
                channel_names=self.params.channel_names,
                device=self.device,
                scale=scale,
                bias=bias,
                output_channels=spectrum_channels,
                output_file=spectrum_file,
                spatial_distributed=True,
            )
        else:
            spectrum_buffer = None

        if zonal_spectrum_file is not None:
            # we use all channels for the zonal spectral computation
            zonal_spectrum_channels = output_channels if output_channels else self.params.channel_names
            zonal_spectrum_buffer = ZonalSpectrumAverageBuffer(
                num_rollout_steps=rollout_steps,
                rollout_dt=self.params.dt,
                img_shape=img_shape,
                ensemble_size=self.params.local_ensemble_size,
                channel_names=self.params.channel_names,
                lat_lon=(self.params.lat, self.params.lon),
                device=self.device,
                scale=scale,
                bias=bias,
                output_channels=zonal_spectrum_channels,
                output_file=zonal_spectrum_file,
                spatial_distributed=True,
            )
        else:
            zonal_spectrum_buffer = None

        # if wb2 compatibility enabled check if a mask was specified
        if wb2_compatible and self.log_to_screen:
            if self.mask_dataset is None:
                self.logger.warning(
                    "WeatherBench compatibility enabled but no mask specified. Results may differ, please specify a mask."
                )
            if self.climatology_dataset is None:
                self.logger.warning(
                    "WeatherBench compatibility enabled but no climatology specified. Results may differ, please specify a wb2-compatible climatology."
                )

        logs = self._inference_indexlist(
            indices,
            rollout_steps,
            batch_size,
            metrics=metrics,
            rollout_buffer=rollout_buffer,
            bias_buffer=bias_buffer,
            spectrum_buffer=spectrum_buffer,
            zonal_spectrum_buffer=zonal_spectrum_buffer,
            profiler=profiler,
        )

        if compute_metrics and metrics_file is not None:
            if comm.get_rank("world") == 0:
                metrics.save(metrics_file)

        return logs

    def _inference_indexlist(
        self,
        indices: Union[List[int], torch.Tensor],
        rollout_steps: int,
        batch_size: int,
        profiler: Optional = None,
        metrics: Optional = None,
        rollout_buffer: Optional = None,
        bias_buffer: Optional = None,
        spectrum_buffer: Optional = None,
        zonal_spectrum_buffer: Optional = None,
    ):
        """
        main routine that implements autoregressive inference over a number of indices
        """

        # set to eval
        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        if metrics is not None:
            metrics.zero_buffers()

        # initialize rollout buffers
        if rollout_buffer is not None:
            rollout_buffer.zero_buffers()

        if bias_buffer is not None:
            bias_buffer.zero_buffers()

        if spectrum_buffer is not None:
            spectrum_buffer.zero_buffers()

        if zonal_spectrum_buffer is not None:
            zonal_spectrum_buffer.zero_buffers()

        # this is the biggest index we can produce
        num_samples = self.valid_dataset.n_samples_total

        date_fn = np.vectorize(get_date_from_timestamp)

        # now we need to reorganize the samples so that the loader first rolls out the first batch indices, then the next,
        # etc: generate batched list first:
        # use sorted index sampler, which does the trick
        sampler = SortedIndexSampler(indices, num_samples, batch_size, rollout_steps, self.params.dt)
        subset_dataloader = tud.DataLoader(
            self.valid_dataset,
            batch_sampler=sampler,
            prefetch_factor=6,
            pin_memory=True,
            num_workers=self.params.num_data_workers,
        )

        # get timedelta with respect to beginning of year
        if self.mask_dataset is not None:
            mask_sampler = translate_date_sampler_to_timedelta_sampler(sampler, self.valid_dataset, self.mask_dataset)
            self.mask_dataloader = tud.DataLoader(
                self.mask_dataset,
                batch_sampler=mask_sampler,
                prefetch_factor=6,
                pin_memory=True,
                num_workers=self.params.num_data_workers,
            )
            mask_iterator = iter(self.mask_dataloader)

        if self.climatology_dataset is not None:
            climatology_sampler = translate_date_sampler_to_timedelta_sampler(
                sampler, self.valid_dataset, self.climatology_dataset
            )
            self.climatology_dataloader = tud.DataLoader(
                self.climatology_dataset,
                batch_sampler=climatology_sampler,
                prefetch_factor=6,
                pin_memory=True,
                num_workers=self.params.num_data_workers,
            )
            climatology_iterator = iter(self.climatology_dataloader)

        # create loader for the full epoch
        rollout = None  # the rollout of the current initial condition, the ensemble folded into the batch
        idt = 0
        E = self.params.local_ensemble_size
        with torch.inference_mode():
            with torch.no_grad():
                for token in tqdm(subset_dataloader, desc="Inference progress", disable=not self.log_to_screen):

                    # the step within the rollout of the current initial condition
                    idte = idt % (rollout_steps + 1)

                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_push(f"inference step {idt}, rollout step {idte}")
                    self.step_timer.begin_step()

                    # move input to GPU; the loader runs without targets, so every step is
                    # read as an input: the initial condition first, then the verification
                    gbatch = batch_to_device(token, self.device)
                    gfield = gbatch[INPUT]
                    gtime = gbatch[INPUT_TIMESTAMP]

                    # get mask
                    if self.mask_dataset is not None:
                        # we need target mask
                        masks = next(mask_iterator)[INPUT]
                        # send to device and remove time dim
                        masks = masks.to(self.device).squeeze(1)

                        # we should normalize the masks just in case:
                        masks_norm = self.quadrature(masks).unsqueeze(-1).unsqueeze(-1)
                        masks = masks / masks_norm
                    else:
                        masks = None

                    # get climatology
                    if self.climatology_dataset is not None:
                        # we need the target clims
                        clims = next(climatology_iterator)[INPUT]
                        # send to device and remove time dim
                        clims = clims.to(self.device).squeeze(1)
                    else:
                        clims = None

                    if idte == 0:
                        # episode start: a new initial condition. The rollout folds the ensemble
                        # dim into the batch dim, (B*E, ...) going into the model, and replaces
                        # the noise state at its first step; it evolves via AR updates after that.
                        # the initial condition is the latest step of the window, the steps before it are history
                        tinp = gtime[:, -1]

                        dates = date_fn(tinp.detach().cpu()).tolist()
                        datestring = ", ".join([d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates])
                        print(f"inferencing dates: {datestring}")

                        noise_source = self.preprocessor.noise_source(batch_size=gfield.shape[0] * E)
                        rollout = Rollout(gbatch, n_steps=rollout_steps, noise_source=noise_source, ensemble_size=E)

                        if rollout_buffer is not None:
                            # every member starts from the same initial condition, (B, E, C, H, W)
                            ic = rollout.unfold(expand_ensemble(gfield[:, -1], E))
                            rollout_buffer.update(ic, tinp, idt=idte)

                    else:
                        # rollout step: the sample is the verification, its latest step is the
                        # predicted time
                        tar, ttar = gfield[:, -1:], gtime[:, -1:]
                        tar_features = {
                            name: feature[:, -1:] for name, feature in gbatch[INPUT_DYNAMIC_FEATURES].items()
                        }

                        # the losses, metrics and buffers take the channels flat
                        targ = self.preprocessor.flatten_history(tar)

                        with self.autocast():
                            # single forward on the folded batch
                            step = next(rollout)
                            pred_step = self.model.predict_step(step)

                            # the batch has no targets: the verification sample supplies the
                            # dynamic features and the time the window moves on to
                            rollout.advance(pred_step, dynamic_features=tar_features, timestamp=ttar)

                            # split the ensemble back out, (B, E, ...), for metrics / loss / buffers
                            pred = rollout.unfold(self.preprocessor.flatten_history(pred_step))

                            # no weight here, since it is not clear what it would be for the spectral
                            # components in spectral CRPS
                            loss = self.loss_obj(pred, targ)

                            if rollout_buffer is not None:
                                rollout_buffer.update(pred, ttar[:, 0], idt=idte)

                        # update the metrics starting from the first actual prediction
                        if metrics is not None:

                            # subtract clim
                            if clims is not None:
                                predc = pred - clims.unsqueeze(1)
                                targc = targ - clims
                            else:
                                predc = pred
                                targc = targ

                            # update metrics
                            metrics.update(predc, targc, loss, idte - 1, masks)

                        # update the bias computation
                        if bias_buffer is not None:
                            diff = pred - targ.unsqueeze(dim=1)
                            B, E, C, H, W = diff.shape
                            diff = diff.reshape(B * E, C, H, W).contiguous()
                            bias_buffer.update(diff, idte - 1)

                        if spectrum_buffer is not None:
                            if pred.dim() == 4:
                                prede = pred.unsqueeze(1)
                            else:
                                prede = pred

                            spectrum_buffer.update(prede, targ.unsqueeze(1), idte - 1)

                        if zonal_spectrum_buffer is not None:
                            if pred.dim() == 4:
                                prede = pred.unsqueeze(1)
                            else:
                                prede = pred

                            zonal_spectrum_buffer.update(prede, targ.unsqueeze(1), idte - 1)

                    self.step_timer.end_step()
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()

                    # increment counter
                    idt += 1

                    if profiler is not None:
                        profiler.step()

        # barrier to ensure everyone is here before calling finalize
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # create final logs
        if metrics is not None:
            logs = metrics.finalize()
        else:
            logs = dict()

        # wait for the copying and writing to disk to finish. Timed separately and recorded: in
        # buffered mode most of the write lands here rather than in the rollout steps, so a
        # per-step figure alone would understate what writing the output actually costs.
        self.output_finalize_seconds = 0.0
        if rollout_buffer is not None:
            finalize_start = time.perf_counter_ns()
            rollout_buffer.finalize()
            self.output_finalize_seconds = (time.perf_counter_ns() - finalize_start) * 1e-9

        # record bias
        if bias_buffer is not None:
            bias_buffer.finalize()

        # record spectrum
        if spectrum_buffer is not None:
            spectrum_buffer.finalize()

        # record zonal spectrum
        if zonal_spectrum_buffer is not None:
            zonal_spectrum_buffer.finalize()

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        return logs

    def log_score(self, scoring_logs, scoring_time):
        # separator
        separator = "".join(["-" for _ in range(50)])
        print_prefix = "    "

        def get_pad(nchar):
            return "".join([" " for x in range(nchar)])

        if self.log_to_screen:
            # header:
            self.logger.info(separator)
            self.logger.info("Scoring summary:")
            self.logger.info("Total scoring time is {:.2f} sec".format(scoring_time))

            # compute padding:
            print_list = list(scoring_logs["metrics"].keys())
            max_len = max([len(x) for x in print_list])
            pad_len = [max_len - len(x) for x in print_list]
            # validation summary
            self.logger.info("Metrics:")
            for idk, key in enumerate(print_list):
                value = scoring_logs["metrics"][key]
                self.logger.info(f"{print_prefix}{key}: {get_pad(pad_len[idk])}{value}")
            self.logger.info(separator)

        return

    def score_model(
        self,
        metrics_file: Optional[str] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        bias_file: Optional[str] = None,
        spectrum_file: Optional[str] = None,
        zonal_spectrum_file: Optional[str] = None,
        start_date=None,
        end_date=None,
        date_step=1,
        wb2_compatible=False,
        enable_odirect: bool = False,
        odirect_alignment: int = 0,
        enable_gds: bool = False,
        profiler=None,
    ):
        """
        main routine for scoring models. Runs the inference over the entire dataset and computes the score. Then writes them to disk
        """

        # log parameters
        if self.log_to_screen:
            # log memory usage so far
            all_mem_gb, max_mem_gb = get_memory_usage(self.device)
            self.logger.info(
                f"Scaffolding memory high watermark: {all_mem_gb:.2f} GB ({max_mem_gb:.2f} GB for pytorch)"
            )
            # announce training start
            self.logger.info("Starting Scoring...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        # parse string dates (if supplied) into datetimes; helper expects datetimes
        if start_date is not None:
            start_date = get_date_from_string(start_date)
        if end_date is not None:
            end_date = get_date_from_string(end_date)

        # resolve range, validate, and snap dates to dataset cadence
        start_index, end_index, step, start_date, end_date = compute_inference_range(
            self.valid_dataset, start_date, end_date, date_step
        )
        if self.log_to_screen:
            self.logger.info(f"Using date range: {start_date} to {end_date} with a step of {date_step} hours.")
            if output_channels:
                self.logger.info(f"Logging following channels: {output_channels}")

        # split the samples across ranks
        samples_local = split_list(list(range(start_index, end_index, step)), comm.get_size("batch"))[
            comm.get_rank("batch")
        ]

        # distribute samples across all ranks
        start = min(samples_local)
        end = max(samples_local) + 1

        # start timer
        scoring_start = time.time()

        rollout_steps = self.params.get("valid_autoreg_steps", 0) + 1
        scoring_logs = self.inference_range(
            start,
            end,
            step,
            dhours=self.valid_dataset.dhours,
            rollout_steps=rollout_steps,
            batch_size=self.params.batch_size,
            compute_metrics=True,
            metrics_file=metrics_file,
            output_channels=output_channels,
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            bias_file=bias_file,
            spectrum_file=spectrum_file,
            zonal_spectrum_file=zonal_spectrum_file,
            wb2_compatible=wb2_compatible,
            enable_odirect=enable_odirect,
            odirect_alignment=odirect_alignment,
            enable_gds=enable_gds,
            profiler=profiler,
        )

        # end timer
        scoring_end = time.time()

        self.log_score(scoring_logs, scoring_end - scoring_start)

        return
