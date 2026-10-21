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
import torch
import logging
from functools import partial

# utilities
from makani.utils import logging_utils
from makani.utils.YParams import YParams

# distributed computing stuff
from makani.utils import comm
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils import argument_parser
from makani.utils.argument_parser import parse_odirect_config
from makani.utils import profiling

# import trainer
from makani import Inferencer


if __name__ == "__main__":
    parser = argument_parser.get_default_argument_parser()
    parser.add_argument("--ensemble_size", default=0, type=int, help="Ensemble size parameter")
    parser.add_argument("--ensemble_parallel_size", default=1, type=int, help="Ensemble parallelization")
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument(
        "--output_channels", default=[], nargs="+", type=str, help="Channels to output. Must be specified as a list."
    )
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        help="Name of the output file. Will be written to the scores folder in the experiment directory.",
    )
    parser.add_argument(
        "--output_memory_buffer_size",
        default=None,
        type=int,
        help="Number of (initial-condition x leadtime) steps to buffer in local memory before flushing to disk. With valid_autoreg_steps=R, one initial condition produces R+1 steps; setting this below R+1 enables mid-rollout flushing, which is useful when a single trajectory does not fit in memory. Bigger values use more memory but reduce I/O overhead. Defaults to the entire run (num_samples * (R+1)). Clamped to at least the local batch size. Pass 0 (or any non-positive value) to disable buffering entirely and stream every step directly to disk.",
    )
    parser.add_argument(
        "--dataset_file_suffix",
        default=None,
        type=str,
        help="Deprecated and ignored: the storage layout is detected from what is at the data path.",
    )
    parser.add_argument(
        "--mask_file",
        default=None,
        type=str,
        help="Masking file in order to weight datapoints geographically. If not specified, uniform weighting is applied.",
    )
    parser.add_argument(
        "--climatology_file",
        default=None,
        type=str,
        help="Time dependent climatology file in order to subtract climatological mean. If not specified, static climatology is applied.",
    )
    parser.add_argument("--metrics_file", default="metrics.h5", type=str, help="Name of the metrics output file.")
    parser.add_argument(
        "--bias_file", default=None, type=str, help="If specified, bias will be computed and saved to this file."
    )
    parser.add_argument(
        "--spectrum_file",
        default=None,
        type=str,
        help="If specified, spectrum will be computed and saved to this file.",
    )
    parser.add_argument(
        "--zonal_spectrum_file",
        default=None,
        type=str,
        help="If specified, zonal spectrum will be computed and saved to this file.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for running inference. It has to be within the dates available in the dataset. If end_date is specified, it has to be smaller than that. Has to be in the format YYYY-MM-DD (ISO 8601 format).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for running inference. It has to be within the dates available in the dataset. If start_date is specified, it has to be bigger than that. Has to be in the format YYY-MM-DD (ISO 8601 format).",
    )
    parser.add_argument(
        "--date_step",
        type=int,
        default=1,
        help="At what interval to sample initial conditions. Needs to be an integer number specifying the step in terms of dhours.",
    )
    parser.add_argument(
        "--wb2_compatible", action="store_true", help="Makes metrics and quadratures compatible with weatherbench2."
    )
    parser.add_argument(
        "--enable_gds",
        action="store_true",
        help="Enable GPUDirect Storage. Opens the output file with the HDF5 GDS driver (requires HDF5 built with the GDS VFD and a cuFile-compatible GPU buffer).",
    )

    # parse
    args = parser.parse_args()

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # distributed
    params["ensemble_parallel_size"] = args.ensemble_parallel_size
    params["matmul_parallel_size"] = args.matmul_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size

    params["data_parallel_sizes"] = [args.ensemble_parallel_size, -1]
    params["data_parallel_names"] = ["ensemble", "batch"]
    params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.matmul_parallel_size]
    params["model_parallel_names"] = ["h", "w", "matmul"]

    # checkpoint format
    params["load_checkpoint"] = args.load_checkpoint

    # make sure to reconfigure logger after the pytorch distributed init
    comm.init(
        model_parallel_sizes=params["model_parallel_sizes"],
        model_parallel_names=params["model_parallel_names"],
        data_parallel_sizes=params["data_parallel_sizes"],
        data_parallel_names=params["data_parallel_names"],
        verbose=False,
    )
    world_rank = comm.get_world_rank()

    # update parameters
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    params["global_batch_size"] = params.batch_size
    if params["global_batch_size"] % comm.get_size("batch") != 0:
        raise ValueError(
            f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('batch')} GPU."
        )
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("batch"))

    # ensemble size
    if args.ensemble_size > 0:
        params["ensemble_size"] = args.ensemble_size
    elif not params.is_set("ensemble_size"):
        params["ensemble_size"] = 1
    if params["ensemble_size"] % comm.get_size("ensemble") != 0:
        raise ValueError(
            f"Error, cannot evenly distribute {params['ensemble_size']} across {comm.get_size('ensemble')} GPU."
        )
    params["local_ensemble_size"] = params["ensemble_size"] // comm.get_size("ensemble")

    # set device
    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        logging.info(f"writing output to {expDir}")
        if not os.path.isdir(expDir):
            os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, "scores"), exist_ok=True)
        os.makedirs(os.path.join(expDir, "scores", "wandb"), exist_ok=True)

    params["experiment_dir"] = os.path.abspath(expDir)

    # output files
    output_channels = args.output_channels
    output_file = (
        os.path.join(params["experiment_dir"], "scores", args.output_file) if args.output_file is not None else None
    )
    metrics_file = os.path.join(params["experiment_dir"], "scores", args.metrics_file)
    bias_file = os.path.join(params["experiment_dir"], "scores", args.bias_file) if args.bias_file is not None else None
    spectrum_file = (
        os.path.join(params["experiment_dir"], "scores", args.spectrum_file) if args.spectrum_file is not None else None
    )
    zonal_spectrum_file = (
        os.path.join(params["experiment_dir"], "scores", args.zonal_spectrum_file)
        if args.zonal_spectrum_file is not None
        else None
    )
    output_memory_buffer_size = args.output_memory_buffer_size

    if args.checkpoint_path is None:
        params["checkpoint_path"] = os.path.join(
            expDir, "training_checkpoints/ckpt_mp{mp_rank}_v{checkpoint_version}.tar"
        )
        params["best_checkpoint_path"] = os.path.join(expDir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")
    else:
        params["checkpoint_path"] = os.path.join(args.checkpoint_path, "ckpt_mp{mp_rank}_v{checkpoint_version}.tar")
        params["best_checkpoint_path"] = os.path.join(args.checkpoint_path, "best_ckpt_mp{mp_rank}.tar")

    # check if all files are there - do not comment out.
    resuming = True
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.checkpoint_path.format(mp_rank=mp_rank, checkpoint_version=0)
        if params["load_checkpoint"] == "legacy" or mp_rank < 1:
            resuming = resuming and os.path.isfile(checkpoint_fname)

    params["resuming"] = False
    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = args.jit_mode
    params["enable_odirect"], params["odirect_alignment"] = parse_odirect_config(args.odirect_config)
    params["enable_s3"] = args.enable_s3
    params["disable_ddp"] = args.disable_ddp
    params["checkpointing_level"] = args.checkpointing_level
    params["enable_synthetic_data"] = args.enable_synthetic_data
    params["split_data_channels"] = args.split_data_channels
    params["n_future"] = 0
    params["mask_file"] = args.mask_file
    params["climatology_file"] = args.climatology_file

    if args.dataset_file_suffix is not None:
        logging.warning(
            "--dataset_file_suffix is deprecated and ignored; the storage layout is detected from the data path."
        )

    # by default we use only the multifiles dataloader for inference
    params["multifiles"] = True

    # wandb configuration
    if params["wandb_name"] is None:
        params["wandb_name"] = args.config + "_inference_" + str(args.run_num)
    if params["wandb_group"] is None:
        params["wandb_group"] = "makani" + args.config
    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = os.path.join(expDir, "scores")

    if world_rank == 0:
        logging_utils.config_logger()
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, "out.log"))
        logging_utils.log_versions()
        params.log(logging.getLogger())

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    # parse dataset metadata
    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError("Error, please specify a dataset descriptor file in json format")

    # instantiate trainer / inference / ensemble object
    inferencer = Inferencer(params, world_rank)

    # run with profiling
    if world_rank in args.capture_ranks:
        if args.capture_type == "torch":
            capture_prefix = f"{args.capture_prefix}_rank{world_rank}" if args.capture_prefix is not None else None
            trace_handler = partial(profiling.trace_handler, print_stats=True, export_trace_prefix=capture_prefix)

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=args.capture_range_start - 1,
                    warmup=1,
                    active=args.capture_range_stop - args.capture_range_start,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                inferencer.score_model(
                    metrics_file=metrics_file,
                    output_channels=output_channels,
                    output_file=output_file,
                    output_memory_buffer_size=output_memory_buffer_size,
                    bias_file=bias_file,
                    spectrum_file=spectrum_file,
                    zonal_spectrum_file=zonal_spectrum_file,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    date_step=args.date_step,
                    wb2_compatible=args.wb2_compatible,
                    enable_odirect=params["enable_odirect"],
                    odirect_alignment=params["odirect_alignment"],
                    enable_gds=args.enable_gds,
                    profiler=profiler,
                )
        elif args.capture_type == "cupti":
            with profiling.CUDAProfiler(
                capture_range_start=args.capture_range_start, capture_range_stop=args.capture_range_stop, enabled=True
            ) as profiler:
                with torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=False):
                    inferencer.score_model(
                        metrics_file=metrics_file,
                        output_channels=output_channels,
                        output_file=output_file,
                        output_memory_buffer_size=output_memory_buffer_size,
                        bias_file=bias_file,
                        spectrum_file=spectrum_file,
                        zonal_spectrum_file=zonal_spectrum_file,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        date_step=args.date_step,
                        wb2_compatible=args.wb2_compatible,
                        enable_odirect=params["enable_odirect"],
                        odirect_alignment=params["odirect_alignment"],
                        enable_gds=args.enable_gds,
                        profiler=profiler,
                    )

    else:
        inferencer.score_model(
            metrics_file=metrics_file,
            output_channels=output_channels,
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            bias_file=bias_file,
            spectrum_file=spectrum_file,
            zonal_spectrum_file=zonal_spectrum_file,
            start_date=args.start_date,
            end_date=args.end_date,
            date_step=args.date_step,
            wb2_compatible=args.wb2_compatible,
            enable_odirect=params["enable_odirect"],
            odirect_alignment=params["odirect_alignment"],
            enable_gds=args.enable_gds,
        )

    # cleanup
    comm.cleanup()
