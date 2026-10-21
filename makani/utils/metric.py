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

from typing import Optional
from functools import partial

import h5py as h5

import torch
import wandb

# distributed computing stuff
from makani.utils import comm

# loss stuff
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.losses import LossType
from makani.utils.missing_data import fill_missing, missing_values, missing_weights
from makani.utils.metrics.functions import (
    GeometricL1,
    GeometricRMSE,
    GeometricACC,
    GeometricSpread,
    GeometricSSR,
    GeometricCRPS,
    GeometricRankHistogram,
    Quadrature,
)
import torch.distributed as dist
from torch_harmonics.distributed import compute_split_shapes
from makani.mpu.mappings import gather_from_parallel_region, reduce_from_parallel_region
from makani.mpu.helpers import barrier


class MetricRollout:
    def __init__(
        self,
        metric_name,
        metric_channels,
        metric_handle,
        channel_names,
        num_rollout_steps,
        dtphys,
        device,
        aux_shape=None,
        aux_shape_finalized=None,
        scale=None,
        mask_target_nan=True,
        integrate=False,
        report_metric=True,
    ):

        # store members
        self.metric_name = metric_name
        self.metric_channels = metric_channels
        self.device = device
        self.integrate = integrate
        # we assume that the prediction are nan-free (the model should take care of this)
        self.mask_target_nan = mask_target_nan
        self.report_metric = report_metric
        self.num_rollout_steps = num_rollout_steps
        self.dtphys = dtphys
        # setting this allows the rollout handler to
        # work with metrics which do have additional dimensions.
        # it will be inserted after the channels dim
        self.aux_shape = aux_shape
        self.aux_shape_finalized = aux_shape_finalized

        # instantiate handle
        self.metric_func = metric_handle().to(self.device)
        self.metric_type = self.metric_func.type
        # self.metric_func = torch.compile(self.metric_func, mode="max-autotune-no-cudagraphs")

        if self.metric_func.batch_reduction == "none":
            raise ValueError("Batch reduction mode 'none' is not supported for rollout handlers")

        # get mapping from channels to all channels
        self.channel_mask = [channel_names.index(c) for c in metric_channels]

        # deal with scale
        if scale is not None:
            self.scale = scale[self.channel_mask].to(self.device)

        # create arrays:
        self.num_channels = len(self.metric_channels)
        if self.aux_shape is None:
            data_shape = (self.num_rollout_steps, self.num_channels)
            counter_shape = (self.num_rollout_steps, self.num_channels)
        else:
            data_shape = (self.num_rollout_steps, self.num_channels, *self.aux_shape)
            counter_shape = (self.num_rollout_steps, self.num_channels, *(1 for _ in range(len(self.aux_shape))))
        self.rollout_curve = torch.zeros(data_shape, dtype=torch.float32, device=self.device)
        self.rollout_counter = torch.zeros(counter_shape, dtype=torch.float32, device=self.device)

        # CPU buffers
        pin_memory = self.device.type == "cuda"

        if self.aux_shape_finalized is None:
            data_shape_finalized = (self.num_rollout_steps, self.num_channels)
            integral_shape = self.num_channels
        else:
            data_shape_finalized = (self.num_rollout_steps, self.num_channels, *self.aux_shape_finalized)
            integral_shape = (self.num_channels, *self.aux_shape_finalized)

        self.rollout_curve_cpu = torch.zeros(
            data_shape_finalized, dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )

        if self.integrate:
            self.rollout_integral = torch.zeros(integral_shape, dtype=torch.float32, device=self.device)
            self.rollout_integral_cpu = torch.zeros(
                integral_shape, dtype=torch.float32, device="cpu", pin_memory=pin_memory
            )
            self.simpquad = Quadrature(self.num_rollout_steps - 1, 1.0 / float(self.num_rollout_steps), self.device)

    @property
    def type(self):
        return self.metric_type

    def zero_buffers(self):
        """set buffers to zero"""
        with torch.no_grad():
            self.rollout_curve.fill_(0.0)
            self.rollout_counter.fill_(0.0)
            if self.integrate:
                self.rollout_integral.fill_(0.0)
        return

    def update(self, inp: torch.Tensor, tar: torch.Tensor, idt: int, wgt: Optional[torch.Tensor] = None):

        # check dimension
        inpp = inp[..., self.channel_mask, :, :]
        tarp = tar[..., self.channel_mask, :, :]

        wgtt = wgt[..., self.channel_mask, :, :] if wgt is not None else None

        # missing target values get zero weight, and the metric is renormalized
        # to the valid area. Only masked if there actually are any, since the
        # weighted path costs more
        if self.mask_target_nan:
            missing = missing_values(tarp)
            if torch.any(missing):
                tarp = fill_missing(tarp, 0.0, missing)
                wgtt = missing_weights(missing, wgtt)

        # compute metric
        metric = self.metric_func(inpp, tarp, wgtt)

        if hasattr(self, "scale"):
            metric = metric * self.scale

        # compute counts
        counts_new = self.metric_func.compute_counts(inpp, wgtt)

        # stack with previous values and counts
        vals = torch.stack([self.rollout_curve[idt, ...], metric], dim=0)
        counts = torch.stack([self.rollout_counter[idt, ...], counts_new], dim=0)
        vals, counts = self.metric_func.combine(vals, counts, dim=0)
        self.rollout_curve[idt, ...].copy_(vals)
        self.rollout_counter[idt, ...].copy_(counts)

        return

    def _combine_helper(self, vallist, countlist):
        vals = torch.stack(vallist, dim=0).contiguous()
        counts = torch.stack(countlist, dim=0).contiguous()
        for idt in range(self.num_rollout_steps):
            vtmp, ctmp = self.metric_func.combine(vals[:, idt, ...], counts[:, idt, ...], dim=0)
            self.rollout_curve[idt, ...].copy_(vtmp)
            self.rollout_counter[idt, ...].copy_(ctmp)

    def reduce(self, non_blocking=False):
        # sum here
        with torch.no_grad():
            if dist.is_initialized():
                # gather results
                # values
                vallist = [torch.empty_like(self.rollout_curve) for _ in range(comm.get_size("batch"))]
                vallist[comm.get_rank("batch")] = self.rollout_curve
                valreq = dist.all_gather(
                    vallist, self.rollout_curve, group=comm.get_group("batch"), async_op=non_blocking
                )
                # counter
                countlist = [torch.empty_like(self.rollout_counter) for _ in range(comm.get_size("batch"))]
                countlist[comm.get_rank("batch")] = self.rollout_counter
                countreq = dist.all_gather(
                    countlist, self.rollout_counter, group=comm.get_group("batch"), async_op=non_blocking
                )
                if valreq is not None:
                    valreq.wait()
                if countreq is not None:
                    countreq.wait()

                # combine
                self._combine_helper(vallist, countlist)
        return

    def finalize(self, non_blocking=False):
        """Finalize routine to gather the metrics to rank 0 and assemble logs"""

        # sum here
        with torch.no_grad():
            rollout_curve_normalized = self.metric_func.finalize(self.rollout_curve, self.rollout_counter)

            # copy to host
            self.rollout_curve_cpu.copy_(rollout_curve_normalized, non_blocking=non_blocking)

            # integrate
            if self.integrate:
                self.rollout_integral.copy_(self.simpquad(rollout_curve_normalized, dim=0))
                self.rollout_integral_cpu.copy_(self.rollout_integral, non_blocking=non_blocking)

        return

    def report(self, index=0, table=False):
        log = {}

        if self.report_metric:
            rollout_curve_arr = self.rollout_curve_cpu.numpy()
            for idx, var_name in enumerate(self.metric_channels):
                log[f"{self.metric_name} {var_name}({(index+1) * self.dtphys})"] = rollout_curve_arr[index, idx]

            if self.integrate:
                rollout_integral_arr = self.rollout_integral_cpu.numpy()
                for idx, var_name in enumerate(self.metric_channels):
                    log[f"{self.metric_name} AUC {var_name}({self.num_rollout_steps * self.dtphys})"] = (
                        rollout_integral_arr[idx]
                    )

        report = log

        if table:
            if self.report_metric:
                table_data = []
                for d in range(0, self.num_rollout_steps):
                    for idx, var_name in enumerate(self.metric_channels):
                        table_data.append(
                            [self.metric_name, var_name, (d + 1) * self.dtphys, rollout_curve_arr[d, idx]]
                        )

                report = (log, table_data)
            else:
                report = (log, [])

        return report


class MetricsHandler:
    """
    Handler object which takes care of computation of metrics. Keeps buffers for the computation of
    """

    def __init__(
        self,
        params,
        climatology,
        num_rollout_steps,
        device,
        l1_var_names=["u10m", "t2m", "sp", "sst", "u500", "z500", "q500", "q50"],
        rmse_var_names=["u10m", "t2m", "sp", "sst", "u500", "z500", "q500", "q50"],
        acc_var_names=["u10m", "t2m", "sp", "sst", "u500", "z500", "q500", "q50"],
        crps_var_names=["u10m", "t2m", "sp", "sst", "u500", "z500", "q500", "q50"],
        spread_var_names=["u10m", "t2m", "sp", "sst", "u500", "z500", "q500", "q50"],
        ssr_var_names=["u10m", "t2m", "sp", "sst", "u500", "z500", "q500", "q50"],
        rh_var_names=[],
        mask_target_nan=True,
        wb2_compatible=False,
    ):

        self.device = device
        self.log_to_screen = params.log_to_screen
        self.log_to_wandb = params.log_to_wandb
        # these are the names of the channels emitted by the dataloader in that order
        self.channel_names = params.channel_names
        self.spatial_distributed = comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)
        self.ensemble_distributed = comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)

        # set a stream
        if self.device.type == "cuda":
            self.stream = torch.Stream(device="cuda")
        else:
            self.stream = None

        # determine effective time interval and number of steps per day:
        self.dtxdh = params.dt * params.dhours
        self.dd = 24 // self.dtxdh

        # select the vars which are actually present
        self.l1_var_names = [x for x in l1_var_names if x in self.channel_names]
        self.rmse_var_names = [x for x in rmse_var_names if x in self.channel_names]
        self.acc_var_names = [x for x in acc_var_names if x in self.channel_names]
        self.crps_var_names = [x for x in crps_var_names if x in self.channel_names]
        self.spread_var_names = [x for x in spread_var_names if x in self.channel_names]
        self.ssr_var_names = [x for x in ssr_var_names if x in self.channel_names]
        self.rh_var_names = [x for x in rh_var_names if x in self.channel_names]

        # split channels not supported atm
        self.split_data_channels = params.split_data_channels
        if self.split_data_channels:
            raise NotImplementedError("Error, split_data_channels is not supported")

        # load normalization term:
        bias, scale = get_data_normalization(params)
        if bias is not None:
            bias = torch.from_numpy(bias).to(torch.float32)
            # filter by used channels
            bias = bias[:, params.out_channels, ...].contiguous()
        else:
            bias = torch.zeros((1, params.N_out_channels, 1, 1), dtype=torch.float32)

        if scale is not None:
            scale = torch.from_numpy(scale).to(torch.float32)
            # filter by used channels: this is done inside the metrics rollout handler
            scale = scale[:, params.out_channels, ...].contiguous()
        else:
            scale = torch.ones((1, params.N_out_channels, 1, 1), dtype=torch.float32)

        # how many steps to run in acc curve
        self.num_rollout_steps = num_rollout_steps

        # store shapes
        self.img_shape = (params.img_shape_x_resampled, params.img_shape_y_resampled)
        self.crop_shape = (params.img_shape_x_resampled, params.img_shape_y_resampled)
        self.crop_offset = (params.img_crop_offset_x, params.img_crop_offset_y)

        # grid extraction
        grid_type = params.get("model_grid_type", "equiangular")

        # mask nan
        self.mask_target_nan = mask_target_nan

        # enable overwriting this with weatherbench2 compatible metrics
        if wb2_compatible:
            if grid_type == "equiangular":
                grid_type = "weatherbench2"
            else:
                raise ValueError(
                    f"weatherbench2 compatibility only supported on equiangular grids. Got {grid_type} instead"
                )

        # set up handles
        self.metric_handles = []

        # L1
        if self.l1_var_names:
            handle = partial(
                GeometricL1,
                grid_type=grid_type,
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                normalize=True,
                channel_reduction="none",
                batch_reduction="sum",
                spatial_distributed=self.spatial_distributed,
            )

            self.metric_handles.append(
                MetricRollout(
                    metric_name="L1",
                    metric_channels=self.l1_var_names,
                    metric_handle=handle,
                    channel_names=self.channel_names,
                    num_rollout_steps=self.num_rollout_steps,
                    dtphys=self.dtxdh,
                    device=self.device,
                    integrate=False,
                    mask_target_nan=self.mask_target_nan,
                    report_metric=True,
                )
            )

        if self.rmse_var_names:
            handle = partial(
                GeometricRMSE,
                grid_type=grid_type,
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                normalize=True,
                channel_reduction="none",
                batch_reduction="sum",
                spatial_distributed=self.spatial_distributed,
            )

            self.metric_handles.append(
                MetricRollout(
                    metric_name="RMSE",
                    metric_channels=self.rmse_var_names,
                    metric_handle=handle,
                    channel_names=self.channel_names,
                    num_rollout_steps=self.num_rollout_steps,
                    dtphys=self.dtxdh,
                    device=self.device,
                    scale=scale.reshape(-1),
                    integrate=False,
                    mask_target_nan=self.mask_target_nan,
                    report_metric=True,
                )
            )
            self.rmse_curve = self.metric_handles[-1].rollout_curve

        if self.acc_var_names:
            # if we use dynamic climatology, then we have to set it to None here:
            # important: we assume that the climatology is normalized with bias and scale!
            if climatology is not None:
                channel_mask = [self.channel_names.index(c) for c in self.acc_var_names]
                clim = climatology.to(torch.float32)
                # project channels
                clim = clim[:, channel_mask, ...].contiguous().to(self.device)
            else:
                clim = None

            handle = partial(
                GeometricACC,
                grid_type=grid_type,
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                normalize=True,
                channel_reduction="none",
                batch_reduction="sum",
                method="macro",
                bias=clim,
                spatial_distributed=self.spatial_distributed,
            )

            self.metric_handles.append(
                MetricRollout(
                    metric_name="ACC",
                    metric_channels=self.acc_var_names,
                    metric_handle=handle,
                    channel_names=self.channel_names,
                    num_rollout_steps=self.num_rollout_steps,
                    dtphys=self.dtxdh,
                    device=self.device,
                    integrate=True,
                    mask_target_nan=self.mask_target_nan,
                    report_metric=True,
                )
            )
            self.acc_curve = self.metric_handles[-1].rollout_curve

        if self.crps_var_names:
            handle = partial(
                GeometricCRPS,
                grid_type=grid_type,
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                normalize=True,
                crps_type="skillspread",
                channel_reduction="none",
                batch_reduction="sum",
                spatial_distributed=self.spatial_distributed,
                ensemble_distributed=self.ensemble_distributed,
            )

            self.metric_handles.append(
                MetricRollout(
                    metric_name="CRPS",
                    metric_channels=self.crps_var_names,
                    metric_handle=handle,
                    channel_names=self.channel_names,
                    num_rollout_steps=self.num_rollout_steps,
                    dtphys=self.dtxdh,
                    device=self.device,
                    scale=scale.reshape(-1),
                    integrate=False,
                    mask_target_nan=self.mask_target_nan,
                    report_metric=True,
                )
            )
            self.crps_curve = self.metric_handles[-1].rollout_curve

        if self.spread_var_names:
            handle = partial(
                GeometricSpread,
                grid_type=grid_type,
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                normalize=True,
                channel_reduction="none",
                batch_reduction="sum",
                spatial_distributed=self.spatial_distributed,
                ensemble_distributed=self.ensemble_distributed,
            )

            self.metric_handles.append(
                MetricRollout(
                    metric_name="Spread",
                    metric_channels=self.spread_var_names,
                    metric_handle=handle,
                    channel_names=self.channel_names,
                    num_rollout_steps=self.num_rollout_steps,
                    dtphys=self.dtxdh,
                    device=self.device,
                    scale=scale.reshape(-1),
                    integrate=False,
                    mask_target_nan=self.mask_target_nan,
                    report_metric=True,
                )
            )

        if self.ssr_var_names:
            handle = partial(
                GeometricSSR,
                grid_type=grid_type,
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                normalize=True,
                channel_reduction="none",
                batch_reduction="sum",
                spatial_distributed=self.spatial_distributed,
                ensemble_distributed=self.ensemble_distributed,
            )

            self.metric_handles.append(
                MetricRollout(
                    metric_name="SSR",
                    metric_channels=self.ssr_var_names,
                    metric_handle=handle,
                    channel_names=self.channel_names,
                    num_rollout_steps=self.num_rollout_steps,
                    dtphys=self.dtxdh,
                    device=self.device,
                    integrate=False,
                    mask_target_nan=self.mask_target_nan,
                    report_metric=True,
                )
            )
            self.ssr_curve = self.metric_handles[-1].rollout_curve

        if self.rh_var_names:
            handle = partial(
                GeometricRankHistogram,
                grid_type=grid_type,
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                normalize=True,
                channel_reduction="none",
                batch_reduction="sum",
                spatial_distributed=self.spatial_distributed,
                ensemble_distributed=self.ensemble_distributed,
            )

            # get ensemble size:
            ens_size = params.get("ensemble_size", 1)

            self.metric_handles.append(
                MetricRollout(
                    metric_name="Rank Histogram",
                    metric_channels=self.rh_var_names,
                    metric_handle=handle,
                    channel_names=self.channel_names,
                    num_rollout_steps=self.num_rollout_steps,
                    dtphys=self.dtxdh,
                    device=self.device,
                    aux_shape=(ens_size + 1,),
                    aux_shape_finalized=(ens_size + 1,),
                    integrate=False,
                    mask_target_nan=self.mask_target_nan,
                    report_metric=False,
                )
            )
            self.rh_curve = self.metric_handles[-1].rollout_curve

        # we need gather shapes
        if comm.get_size("spatial") > 1:
            self.gather_shapes_h = compute_split_shapes(self.crop_shape[0], comm.get_size("h"))
            self.gather_shapes_w = compute_split_shapes(self.crop_shape[1], comm.get_size("w"))

    @torch.compiler.disable(recursive=False)
    def _gather_input(self, x: torch.Tensor) -> torch.Tensor:
        """gather and crop the data"""

        if comm.get_size("h") > 1:
            x = gather_from_parallel_region(x, -2, self.gather_shapes_h, "h")

        if comm.get_size("w") > 1:
            x = gather_from_parallel_region(x, -1, self.gather_shapes_w, "w")

        return x

    def initialize_buffers(self):
        """initialize buffers for computing metrics"""

        # initialize buffers for the validation metrics
        self.valid_buffer = torch.zeros((2), dtype=torch.float32, device=self.device)
        self.valid_loss = self.valid_buffer[0].view(-1)
        self.valid_steps = self.valid_buffer[1].view(-1)

    def zero_buffers(self):
        """set buffers to zero"""

        with torch.no_grad():
            self.valid_buffer.fill_(0)

        for handle in self.metric_handles:
            handle.zero_buffers()

        return

    def update(self, prediction, target, loss, idt, weight=None):
        """update function to update buffers on each autoregressive rollout step"""

        if prediction.dim() == 5:
            prediction_mean = torch.mean(prediction, dim=1)
            if self.ensemble_distributed:
                prediction_mean = reduce_from_parallel_region(prediction_mean, "ensemble") / float(
                    comm.get_size("ensemble")
                )
        else:
            prediction_mean = prediction
            prediction = prediction.unsqueeze(1)

        for handle in self.metric_handles:
            if handle.type == LossType.Deterministic:
                handle.update(prediction_mean, target, idt, weight)
            elif handle.type == LossType.Probabilistic:
                handle.update(prediction, target, idt, weight)
            else:
                raise NotImplementedError(f"Error, LossType {handle.type} not implemented.")

        # only update this at the first step
        if idt == 0:
            self.valid_steps += 1.0
            self.valid_loss += loss

        return

    def finalize(self):
        """Finalize routine to gather all of the metrics to rank 0 and assemble logs"""

        # sync here
        if dist.is_initialized():
            barrier(self.device)

        with torch.no_grad():

            if self.stream is not None:
                self.stream.wait_stream(torch.cuda.current_stream())

            # reductions
            with torch.cuda.stream(self.stream):

                if dist.is_initialized():
                    req = dist.all_reduce(
                        self.valid_buffer, op=dist.ReduceOp.SUM, group=comm.get_group("batch"), async_op=True
                    )
                    req.wait()

                for handle in self.metric_handles:
                    handle.reduce(non_blocking=True)

            # finalize computations
            with torch.cuda.stream(self.stream):

                self.valid_loss /= self.valid_steps

                for handle in self.metric_handles:
                    handle.finalize(non_blocking=True)

            if self.stream is not None:
                self.stream.synchronize()

            # prepare logs with the minimum content
            valid_loss = self.valid_loss.item()
            valid_steps = self.valid_steps.item()
            logs = {"base": {"validation steps": valid_steps, "validation loss": valid_loss}, "metrics": {}}

            table_data = []
            for handle in self.metric_handles:

                if handle.metric_name != "ACC":
                    # report single step
                    tmplog = handle.report(index=0, table=False)
                    for key in tmplog:
                        logs["metrics"]["validation " + key] = tmplog[key]

                # report final rollout step
                tmplog, tmptable = handle.report(index=self.num_rollout_steps - 1, table=True)
                for key in tmplog:
                    logs["metrics"]["validation " + key] = tmplog[key]
                table_data += tmptable

            # add table
            logs["metrics"]["rollouts"] = wandb.Table(
                data=table_data, columns=["metric type", "variable name", "time [h]", "value"]
            )

        self.logs = logs

        return logs

    def save(self, metrics_file: str):
        """Save metrics to an output hdf5 file"""

        # write everything into an output file
        metrics_file = h5.File(metrics_file, "w")

        # iterate over handles and write the output to the h5py file
        for handle in self.metric_handles:
            # create group
            handle_name = handle.metric_name
            handle_group = metrics_file.create_group(handle.metric_name)

            # write data
            handle_group.create_dataset("metric_data", data=handle.rollout_curve_cpu.numpy())

            # make dimension scales
            dset = handle_group.create_dataset("channel", data=handle.metric_channels)
            dset.make_scale("channel")
            dset = handle_group.create_dataset(
                "lead_time", data=handle.dtphys * torch.arange(1, handle.num_rollout_steps + 1).numpy()
            )
            dset.make_scale("lead_time")

            # annotate
            handle_group["metric_data"].dims[0].attach_scale(handle_group["lead_time"])
            handle_group["metric_data"].dims[0].label = "Lead time relative to timestamp"
            handle_group["metric_data"].dims[1].attach_scale(handle_group["channel"])
            handle_group["metric_data"].dims[1].label = "Channel name"

        metrics_file.close()

        return
