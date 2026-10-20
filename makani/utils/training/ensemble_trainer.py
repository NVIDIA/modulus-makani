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
import gc
import time
from typing import Optional
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch import amp
import torch.distributed as dist

import wandb

# timers
from makani.utils.profiling import Timer

# makani depenedencies
from makani.utils import LossHandler, MetricsHandler
from makani.utils.driver import Driver
from makani.utils.training import Trainer
from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.data_helpers import get_climatology
from makani.utils.dataloaders.batch import batch_to_device, INPUT, TARGET
from makani.utils.YParams import YParams

# model registry
from makani.models import model_registry

# distributed computing stuff
from makani.utils import comm
from makani.utils.precision import AutocastManager

from makani.mpu.mappings import init_gradient_reduction_hooks
from makani.mpu.helpers import sync_params, gather_uneven

# for counting model parameters
from makani.models.helpers import count_parameters

# checkpoint helpers
from makani.utils.checkpoint_helpers import get_latest_checkpoint_version

# weight normalizing helper
from makani.utils.training.training_helpers import get_memory_usage, clip_grads, NonFiniteLossCheck


class EnsembleTrainer(Trainer):
    """
    Trainer class holding all the necessary information to perform training.
    """

    def __init__(self, params: Optional[YParams] = None, world_rank: Optional[int] = 0, device: Optional[str] = None):
        Driver.__init__(self, params, world_rank, device)

        # init wandb
        with Timer() as timer:
            if self.log_to_wandb:
                self._init_wandb(self.params, job_type="ensemble")
        self.timers["wandb init"] = timer.time

        # set checkpoint version: start at -1 so that first version which is written is 0
        self.checkpoint_version_current = -1

        # init nccl: do a single AR to make sure that SHARP locks
        # on to the right tree, and that barriers can be used etc
        with Timer() as timer:
            if dist.is_initialized():
                tens = torch.ones(1, device=self.device)
                dist.all_reduce(tens, group=comm.get_group("data"))
        self.timers["nccl init"] = timer.time

        # set up mixed precision (amp dtype + optional transformer-engine fp8 recipe).
        # the manager parses the mode once and hands out a single nested autocast cm.
        amp_mode = self.params.amp_mode if hasattr(self.params, "amp_mode") else "none"
        self.autocast = AutocastManager(amp_mode, device_type="cuda", fp8_group=comm.get_group("data"))
        self.amp_dtype = self.autocast.amp_dtype
        self.amp_enabled = self.autocast.amp_enabled
        if self.amp_enabled and self.log_to_screen:
            self.logger.info(f"Enabling automatic mixed precision in '{amp_mode}'.")

        # initialize data loader
        with Timer() as timer:
            if self.log_to_screen:
                self.logger.info(f"Using channel names: {self.params.channel_names}")
                self.logger.info("initializing data loader")
            self.train_dataloader, self.train_data_shapes, self.train_sampler = get_dataloader(
                self.params, self.params.train_data_path, mode="train", device=self.device
            )
            self.valid_dataloader, self.valid_data_shapes, self.valid_sampler = get_dataloader(
                self.params, self.params.valid_data_path, mode="eval", device=self.device
            )
            self._set_data_shapes(self.params, self.valid_data_shapes)
            # obtain the true lon lat grid after cropping and resampling
            self.lat_global = torch.as_tensor(self.valid_data_shapes.lat_lon_local[0]).to(self.device)
            self.lon_global = torch.as_tensor(self.valid_data_shapes.lat_lon_local[1]).to(self.device)
            if comm.get_size("h") > 1:
                self.lat_global = gather_uneven(self.lat_global, 0, "h")
            if comm.get_size("w") > 1:
                self.lon_global = gather_uneven(self.lon_global, 0, "w")
            self.lat_lon_global = (self.lat_global.cpu().numpy(), self.lon_global.cpu().numpy())
        self.timers["dataloader init"] = timer.time

        if self.log_to_screen:
            self.logger.info("data loader initialized")

        # record data required to reproduce workflow using a model package
        with Timer() as timer:
            if self.world_rank == 0:
                from makani.models.model_package import save_model_package

                save_model_package(self.params)
        self.timers["save model package"] = timer.time

        # init preprocessor and model
        with Timer() as timer:
            self.model = model_registry.get_model(self.params).to(self.device)
            self.preprocessor = self.model.preprocessor
        self.timers["model init"] = timer.time

        # print aux channel names:
        if self.log_to_screen:
            self.logger.info(f"Auxiliary channel names: {self.params.aux_channel_names}")

        # if model-parallelism is enabled, we need to sure that shared weights are matching across ranks
        # as random seeds might get out of sync during initialization
        # DEBUG: this also needs to be fixed in NCCL
        # if comm.get_size("model") > 1:
        with Timer() as timer:
            sync_params(self.model, mode="broadcast")
        self.timers["sync parameters"] = timer.time

        # add a barrier here
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # define process group for DDP, we might need to override that
        if dist.is_initialized() and not self.params.disable_ddp:
            ddp_process_group = comm.get_group("data")

        # log gradients to wandb (skipped under torch.compile; see Driver._watch_model)
        self._watch_model(log="all")

        # print model
        if self.log_to_screen:
            self.logger.info(f"\n{self.model}")

        # metrics handler
        with Timer() as timer:
            clim = get_climatology(self.params)
            clim = torch.from_numpy(clim).to(torch.float32)
            rollout_length = params.get("valid_autoreg_steps", 0) + 1
            self.metrics = MetricsHandler(
                params=self.params, climatology=clim, num_rollout_steps=rollout_length, device=self.device
            )
            self.metrics.initialize_buffers()
        self.timers["metric handler init"] = timer.time

        # loss handler
        with Timer() as timer:
            self.loss_obj = LossHandler(self.params)
            self.loss_obj = self.loss_obj.to(self.device)
        self.timers["loss handler init"] = timer.time

        # optimizer and scheduler setup
        # model
        with Timer() as timer:
            self.model_optimizer = self.get_optimizer(self.model, self.params)
            self.model_scheduler = self.get_scheduler(self.model_optimizer, self.params)
        self.timers["optimizer and scheduler init"] = timer.time

        # gradient scaler
        self.gscaler = amp.GradScaler("cuda", enabled=self.autocast.grad_scaler_enabled)
        self.nonfinite_loss_check = NonFiniteLossCheck(
            max_consecutive=self.params.get("max_nonfinite_loss_steps", 0),
            grad_scaler_enabled=self.autocast.grad_scaler_enabled,
        )

        # gradient clipping
        self.max_grad_norm = self.params.get("optimizer_max_grad_norm", -1.0)

        # we need this further down
        # Initialize gradient reduction (DDP-like) hooks on the default stream so that
        # AccumulateGrad nodes use the same stream as training forward/backward,
        # avoiding "AccumulateGrad node's stream does not match" warnings and CUDA
        # graph capture issues.
        with Timer() as timer:
            if dist.is_initialized() and not self.params.disable_ddp:
                self.model = init_gradient_reduction_hooks(
                    self.model,
                    device=self.device,
                    reduction_buffer_count=self.params.parameters_reduction_buffer_count,
                    broadcast_buffers=False,
                    find_unused_parameters=self.params["enable_grad_anomaly_detection"],
                    gradient_as_bucket_view=True,
                    static_graph=False,
                )
        self.timers["reduction hooks init"] = timer.time

        # lets get one sample from the dataloader:
        # set to train just to be safe
        self._set_train()
        # get sample and map to gpu
        iterator = iter(self.train_dataloader)
        data = next(iterator)
        batch = batch_to_device(data, self.device)
        inp, tar = batch[INPUT], batch[TARGET]
        # get shapes
        inp_shape = inp.shape
        tar_shape = tar.shape

        with Timer() as timer:
            self._compile_model(inp_shape)
        self.timers["compile model"] = timer.time

        # visualization wrapper:
        with Timer() as timer:
            out_bias, out_scale = self.train_dataloader.get_output_normalization()
            self._setup_visualizer(out_bias, out_scale)
        self.timers["visualizer init"] = timer.time

        # reload checkpoints
        counters = {"iters": 0, "start_epoch": 0}
        if self.params.pretrained and not self.params.resuming:
            if not self.params.is_set("pretrained_checkpoint_path"):
                raise ValueError("Error, please specify a valid pretrained checkpoint path")

            # use specified checkpoint
            checkpoint_path = self.params.pretrained_checkpoint_path

            if self.log_to_screen:
                self.logger.info(
                    f"Loading pretrained checkpoint {checkpoint_path} in {self.params.load_checkpoint} mode"
                )

            with Timer() as timer:
                self.restore_from_checkpoint(
                    checkpoint_path,
                    model=self.model,
                    loss=self.loss_obj if self.params.get("load_loss", True) else None,
                    optimizer=self.model_optimizer if self.params.get("load_optimizer", True) else None,
                    scheduler=self.model_scheduler if self.params.get("load_scheduler", True) else None,
                    counters=counters if self.params.get("load_counters", True) else None,
                    checkpoint_mode=self.params.load_checkpoint,
                    strict=self.params.get("strict_restore", True),
                )
            self.timers["loading checkpoint"] = timer.time

            # override learning rate - useful when restoring optimizer but want to override the LR
            if self.params.get("override_lr", False):
                for param_group in self.model_optimizer.param_groups:
                    param_group["lr"] = self.params.get("lr", 1e-3)

        if self.params.resuming:

            # find latest checkpoint
            checkpoint_path = self.params.checkpoint_path
            self.checkpoint_version_current = get_latest_checkpoint_version(checkpoint_path)
            checkpoint_path = checkpoint_path.format(
                checkpoint_version=self.checkpoint_version_current, mp_rank="{mp_rank}"
            )

            if self.log_to_screen:
                self.logger.info(f"Resuming from checkpoint {checkpoint_path} in {self.params.load_checkpoint} mode")

            with Timer() as timer:
                self.restore_from_checkpoint(
                    checkpoint_path,
                    model=self.model,
                    loss=self.loss_obj if self.params.get("load_loss", True) else None,
                    optimizer=self.model_optimizer if self.params.get("load_optimizer", True) else None,
                    scheduler=self.model_scheduler if self.params.get("load_scheduler", True) else None,
                    counters=counters if self.params.get("load_counters", True) else None,
                    dataloader=self.get_train_dataloader_for_restore(),
                    checkpoint_mode=self.params.load_checkpoint,
                    strict=self.params.get("strict_restore", True),
                    log_to_screen=self.log_to_screen,
                )
            self.timers["loading checkpoint"] = timer.time

        # read out counters correctly
        self.iters = counters["iters"]
        self.start_epoch = counters["start_epoch"]
        self.epoch = self.start_epoch

        # wait till everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # counting runs a reduction so we need to count on all ranks before printing on rank 0
        pcount, _, _ = count_parameters(self.model, self.device)
        if self.log_to_screen:
            self.logger.info("Number of trainable model parameters: {}".format(pcount))

        # allow freezing of model parameters to train only the perturbation
        if self.params.is_set("freeze_model") and self.params.freeze_model:
            if self.log_to_screen:
                self.logger.info("Freezing model weights")
            for param in self.model.parameters():
                param.requires_grad = False

        # print setup times
        self._log_timers()

    def _set_train(self):
        self.model.train()
        self.loss_obj.train()
        self.preprocessor.train()

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()
        self.preprocessor.eval()

    def train(self, training_profiler=None, validation_profiler=None):
        # log parameters
        if self.log_to_screen:
            # log memory usage so far
            all_mem_gb, max_mem_gb = get_memory_usage(self.device)
            self.logger.info(
                f"Scaffolding memory high watermark: {all_mem_gb:.2f} GB ({max_mem_gb:.2f} GB for pytorch)"
            )
            # announce training start
            self.logger.info("Starting Ensemble Training Loop...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        training_start = time.time()
        best_valid_loss = 1.0e6
        for epoch in range(self.start_epoch, self.params.max_epochs):
            if dist.is_initialized():
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                if self.valid_sampler is not None:
                    self.valid_sampler.set_epoch(epoch)

            # start timer
            epoch_start = time.time()

            # train if not to be skipped
            if not self.params.get("skip_training", False):
                train_time, train_data_gb, train_logs = self.train_one_epoch(profiler=training_profiler)
            else:
                train_time = 0
                train_data_gb = 0
                train_logs = {"train_steps": 0, "loss": 0.0}

            if dist.is_initialized():
                dist.barrier(device_ids=[self.device.index])

            # validate if not to be skipped
            if not self.params.get("skip_validation", False):
                valid_time, viz_time, valid_logs = self.validate_one_epoch(epoch, profiler=validation_profiler)
            else:
                valid_time = 0
                viz_time = 0
                valid_logs = {"base": {}, "metrics": {}}

            if self.params.scheduler == "ReduceLROnPlateau":
                self.model_scheduler.step(valid_logs["base"]["validation loss"])
            elif self.model_scheduler is not None:
                self.model_scheduler.step()

            # log learning rate
            if self.log_to_wandb:
                for param_group in self.model_optimizer.param_groups:
                    lr = param_group["lr"]
                wandb.log({"learning rate": lr}, step=self.epoch)

            # the state of the data pipeline is sharded across the data-parallel comm, so it has
            # to be gathered by all ranks before entering the rank-0-only block below
            if (self.params.save_checkpoint != "none") and not self.params.get("skip_training", False):
                dataloader_state = self.get_train_dataloader_state()
            else:
                dataloader_state = None

            # save out checkpoints
            if (
                (self.data_parallel_rank == 0)
                and (self.params.save_checkpoint != "none")
                and not self.params.get("skip_training", False)
            ):
                store_start = time.time()
                checkpoint_mode = self.params["save_checkpoint"]
                counters = {"iters": self.iters, "epoch": self.epoch}

                # increase checkpoint counter
                self.checkpoint_version_current = (
                    self.checkpoint_version_current + 1
                ) % self.params.checkpoint_num_versions
                checkpoint_path = self.params.checkpoint_path.format(
                    checkpoint_version=self.checkpoint_version_current, mp_rank="{mp_rank}"
                )

                # checkpoint at the end of every epoch
                self.save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.loss_obj,
                    self.model_optimizer,
                    self.model_scheduler,
                    counters,
                    dataloader_state=dataloader_state,
                    checkpoint_mode=checkpoint_mode,
                )

                # save best checkpoint
                best_checkpoint_path = self.params.best_checkpoint_path.format(mp_rank=comm.get_rank("model"))
                best_model_checkpoint_saved = os.path.isfile(best_checkpoint_path)
                if (not self.params.get("skip_validation", False)) and (
                    (not best_model_checkpoint_saved) or (valid_logs["base"]["validation loss"] <= best_valid_loss)
                ):
                    self.save_checkpoint(
                        self.params.best_checkpoint_path,
                        self.model,
                        self.loss_obj,
                        self.model_optimizer,
                        self.model_scheduler,
                        counters,
                        dataloader_state=dataloader_state,
                        checkpoint_mode=checkpoint_mode,
                    )
                    best_valid_loss = valid_logs["base"]["validation loss"]

                # time how long it took
                store_stop = time.time()

                if self.log_to_screen:
                    self.logger.info(
                        f"Saving checkpoint ({checkpoint_mode}) took: {(store_stop - store_start):.2f} sec"
                    )

            # wait for everybody
            if dist.is_initialized():
                dist.barrier(device_ids=[self.device.index])

            # end timer
            epoch_end = time.time()

            # create timing logs:
            timing_logs = {
                "epoch time [s]": epoch_end - epoch_start,
                "training time [s]": train_time,
                "validation time [s]": valid_time,
                "visualization time [s]": viz_time,
                "training step time [ms]": train_logs["train_steps"]
                and (train_time / train_logs["train_steps"]) * 10**3
                or 0,
                "minimal IO rate [GB/s]": train_time and train_data_gb / train_time or 0,
            }

            # log metrics:
            self.log_epoch(train_logs, valid_logs, timing_logs)

            # exit here if not training
            if self.params.get("skip_training", False):
                break

        # training done
        training_end = time.time()
        if self.log_to_screen:
            self.logger.info("Total training time is {:.2f} sec".format(training_end - training_start))

        return

    def _ensemble_step(self, batch):
        """
        Single forward of the rollout over the batch, with the ensemble folded into the batch dim.

        The rollout folds the members into the batch dim, (B*E, ...), for one forward per step and
        unfolds the predictions to (B, E, ...) for the loss; the targets stay at (B, ...). The noise
        is drawn at the folded batch, so each (batch, ensemble member) pair sees an independent sample.
        """
        rollout = self._rollout(batch, ensemble_size=self.params.local_ensemble_size)
        pred_folded = self.model_train(rollout)

        # the losses take the time steps folded into the channels, and the ensemble as dim 1
        flatten = self.preprocessor.flatten_history
        pred = rollout.unfold(flatten(pred_folded))

        loss = self.loss_obj(pred, flatten(batch[TARGET]), inp=flatten(batch[INPUT]))

        return pred, loss

    def train_one_epoch(self, profiler=None):
        self.epoch += 1
        total_data_bytes = 0
        self._set_train()

        # we need this for the loss average
        accumulated_loss = torch.zeros((2), dtype=torch.float32, device=self.device)

        if self.max_grad_norm > 0.0:
            accumulated_grad_norm = torch.zeros((2), dtype=torch.float32, device=self.device)
        else:
            accumulated_grad_norm = None

        train_steps = 0
        train_start = time.perf_counter_ns()
        self.model_train.zero_grad(set_to_none=True)
        progress_bar = tqdm(
            self.train_dataloader, desc=f"Training progress epoch {self.epoch}", disable=not self.log_to_screen
        )
        for data in progress_bar:
            train_steps += 1
            self.iters += 1

            torch.cuda.nvtx.range_push(f"train step {train_steps}")
            self.step_timer.begin_step()

            # map to device
            batch = batch_to_device(data, self.device)
            inp, tar = batch[INPUT], batch[TARGET]

            # assuming float32
            total_data_bytes += inp.nbytes + tar.nbytes

            # check if we need to perform an gacc update
            do_update = train_steps % self.params["gradient_accumulation_steps"] == 0
            loss_scaling_fact = 1.0
            if self.params["gradient_accumulation_steps"] > 1:
                loss_scaling_fact = 1.0 / np.float32(self.params["gradient_accumulation_steps"])

            # accumulate loss into this tensor
            with self.autocast():

                if do_update:
                    pred, loss = self._ensemble_step(batch)
                else:
                    with self.model_train.no_sync():
                        pred, loss = self._ensemble_step(batch)
                loss = loss * loss_scaling_fact

            # stop on a non-finite loss before it reaches the weights, on all ranks together
            self.nonfinite_loss_check(loss, step=self.iters, epoch=self.epoch)

            # backward pass
            self.gscaler.scale(loss).backward()

            # increment accumulated loss
            accumulated_loss[0] += loss.detach().clone() * inp.shape[0]
            accumulated_loss[1] += inp.shape[0]

            # log the loss
            pbar_postfix = {"loss": loss.item()}

            # perform weight update if requested
            if do_update:
                if self.max_grad_norm > 0.0:
                    self.gscaler.unscale_(self.model_optimizer)
                    grad_norm = clip_grads(self.model_train, self.max_grad_norm, verbose=self.log_to_screen)
                    accumulated_grad_norm[0] += grad_norm.detach()
                    accumulated_grad_norm[1] += 1.0
                    pbar_postfix["grad norm"] = grad_norm.item()

                self.gscaler.step(self.model_optimizer)
                self.gscaler.update()
                self.model_train.zero_grad(set_to_none=True)

            if (
                (self.params.print_timings_frequency > 0)
                and (self.iters % self.params.print_timings_frequency == 0)
                and self.log_to_screen
            ):
                running_train_time = time.perf_counter_ns() - train_start
                print("\n")
                print(
                    f"Average step time after step {self.iters}: {running_train_time / float(train_steps) * 10**(-6):.1f} ms"
                )
                print(
                    f"Average effective io rate after step {self.iters}: {total_data_bytes * float(comm.get_world_size()) / (float(running_train_time) * 10**(-9) * 1024. * 1024. * 1024.):.2f} GB/s"
                )
                print(f"Current loss {loss.item()}")
                print("\n")

            # if logging of weights and grads during training is enabled, write them out at the first step of each epoch
            if (self.params.dump_weights_and_grads > 0) and (
                (self.iters - 1) % self.params.dump_weights_and_grads == 0
            ):
                weights_and_grads_path = self.params["experiment_dir"]
                if self.log_to_screen:
                    self.logger.info(f"Dumping weights and gradients to {weights_and_grads_path}")
                self.dump_weights_and_grads(
                    weights_and_grads_path,
                    self.model,
                    step=(self.epoch * self.params.num_samples_per_epoch + self.iters),
                )

            # set progress bar prefix
            progress_bar.set_postfix(**pbar_postfix)

            self.step_timer.end_step()
            torch.cuda.nvtx.range_pop()

            # profiler step
            if profiler is not None:
                profiler.step()

        # average the loss over ranks and steps
        if dist.is_initialized():
            dist.all_reduce(accumulated_loss, op=dist.ReduceOp.SUM, group=comm.get_group("data"))

        # add the train loss to logs
        train_loss = accumulated_loss[0] / (accumulated_loss[1] * loss_scaling_fact)
        logs = {"loss": train_loss.item()}

        # add train steps to log
        logs["train_steps"] = train_steps

        # log gradient norm
        if accumulated_grad_norm is not None:
            grad_norm = accumulated_grad_norm[0] / accumulated_grad_norm[1]
            logs["gradient norm"] = grad_norm.item()

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # finalize timers
        train_end = time.perf_counter_ns()
        train_time = (train_end - train_start) * 10 ** (-9)
        total_data_gb = (total_data_bytes / (1024.0 * 1024.0 * 1024.0)) * float(comm.get_world_size())

        return train_time, total_data_gb, logs

    def validate_one_epoch(self, epoch, profiler=None):
        # set to eval
        self._set_eval()

        # clear cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # synchronize
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # initialize metrics buffers
        self.metrics.zero_buffers()

        visualize_data = self.params.log_video and (epoch % self.params.log_video == 0)

        # start the timer
        valid_start = time.time()

        with torch.inference_mode():
            with torch.no_grad():

                eval_steps = 0
                progress_bar = tqdm(
                    self.valid_dataloader,
                    desc=f"Validation progress epoch {self.epoch}",
                    disable=not self.log_to_screen,
                )
                for data in progress_bar:
                    eval_steps += 1

                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_push(f"eval step {eval_steps}")
                    self.step_timer.begin_step()

                    # map to gpu
                    batch = batch_to_device(data, self.device)

                    # the rollout folds the ensemble into the batch dim for a single forward
                    # per step, (B*E, ...); the target stays at the original batch -- the loss
                    # expects (B, E, ...) pred against (B, ...) target. The noise is replaced at
                    # the first step and continued after it.
                    tar = batch[TARGET]
                    rollout = self._rollout(batch, ensemble_size=self.params.local_ensemble_size)

                    # the losses and metrics take the channels of a single step
                    flatten = self.preprocessor.flatten_history

                    # one prediction at a time, so that each lead time is released after its metrics
                    steps = self._eval_steps(rollout)

                    # loop over lead times
                    num_lead_times = tar.shape[1]
                    for idt in range(num_lead_times):

                        targ = flatten(tar[:, idt : idt + 1])

                        with self.autocast():
                            # split the ensemble back out, (B, E, ...), for the loss
                            pred = rollout.unfold(flatten(next(steps)))

                            loss = self.loss_obj(pred, targ, None)

                            # log the loss
                            progress_bar.set_postfix({"loss": loss.item()})

                        # TODO: move all of this into the visualization handler
                        if (eval_steps <= 1) and visualize_data:
                            # create average prediction for deterministic metrics
                            pred_gather = pred[0, 0, ...].clone()
                            targ_gather = targ[0, ...].clone()

                            pred_gather = self.metrics._gather_input(pred_gather)
                            targ_gather = self.metrics._gather_input(targ_gather)

                            self._visualize_step(pred_gather, targ_gather, eval_steps, idt, num_lead_times)

                        # update metrics
                        self.metrics.update(pred, targ, loss, idt)

                    self.step_timer.end_step()
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()

                    # profiler step
                    if profiler is not None:
                        profiler.step()

                # create final logs
                logs = self.metrics.finalize()

        # finalize plotting
        viz_time = time.perf_counter_ns()
        if visualize_data and self.visualizer is not None:
            self.visualizer.finalize()
        viz_time = (time.perf_counter_ns() - viz_time) * 10 ** (-9)

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # timer
        valid_time = time.time() - valid_start

        return valid_time, viz_time, logs

    def log_epoch(self, train_logs, valid_logs, timing_logs):
        # separator
        separator = "".join(["-" for _ in range(50)])
        print_prefix = "    "

        def get_pad(nchar):
            return "".join([" " for x in range(nchar)])

        if self.log_to_screen:
            # header:
            self.logger.info(separator)
            self.logger.info(f"Epoch {self.epoch} summary:")
            self.logger.info("Performance Parameters:")
            self.logger.info(print_prefix + "training steps: {}".format(train_logs["train_steps"]))
            self.logger.info(print_prefix + "validation steps: {}".format(valid_logs["base"]["validation steps"]))
            all_mem_gb, _ = get_memory_usage(self.device)
            self.logger.info(print_prefix + f"memory footprint [GB]: {all_mem_gb:.2f}")
            for key in timing_logs.keys():
                self.logger.info(print_prefix + key + ": {:.2f}".format(timing_logs[key]))

            # compute padding:
            print_list = ["training loss", "validation loss"] + list(valid_logs["metrics"].keys())
            max_len = max([len(x) for x in print_list])
            pad_len = [max_len - len(x) for x in print_list]
            # validation summary
            self.logger.info("Metrics:")
            self.logger.info(print_prefix + "training loss: {}{}".format(get_pad(pad_len[0]), train_logs["loss"]))
            if "gradient norm" in train_logs:
                plen = max_len - len("gradient norm")
                self.logger.info(
                    print_prefix + "gradient norm: {}{}".format(get_pad(plen), train_logs["gradient norm"])
                )
            self.logger.info(
                print_prefix
                + "validation loss: {}{}".format(get_pad(pad_len[1]), valid_logs["base"]["validation loss"])
            )
            for idk, key in enumerate(print_list[3:], start=3):
                value = valid_logs["metrics"][key]
                if np.isscalar(value):
                    self.logger.info(f"{print_prefix}{key}: {get_pad(pad_len[idk])}{value}")
            self.logger.info(separator)

        if self.log_to_wandb:
            wandb.log(train_logs, step=self.epoch)
            wandb.log(valid_logs["base"], step=self.epoch)

            # log metrics
            wandb.log(valid_logs["metrics"], step=self.epoch, commit=True)

        return
