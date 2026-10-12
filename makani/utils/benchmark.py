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

"""Measurement and bookkeeping helpers for the benchmark driver (``makani/benchmark.py``).

Three separate concerns live here:

1. :class:`StepTimer` -- per-step wall time, recorded with CUDA events so the
   measurement is not distorted by the asynchronous launch queue. It is
   constructed (disabled) by every :class:`~makani.utils.driver.Driver`, so the
   trainers can call ``begin_step``/``end_step`` unconditionally and pay nothing
   outside a benchmark run.

2. Provenance -- what was actually run: the decomposition ``comm`` built, the
   config file and config line, the software/hardware environment.

3. Comparability -- a hash over the parts of the resolved configuration that
   have to match before two step times mean the same thing. Deliberately
   *excludes* the decomposition and the world size, since those are the axes a
   benchmark suite varies.
"""

import os
import re
import json
import time
import datetime
import platform
import socket
import hashlib
import getpass
import subprocess
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import torch
import torch.distributed as dist

from makani.utils import comm


class StepTimer:
    """Records the wall time of individual training/inference steps.

    On CUDA the timing is taken with events recorded into the stream rather than
    with a host-side clock: a host clock around an asynchronous step measures
    launch time, not execution time, and would only become correct by inserting
    a synchronization per step -- which itself perturbs what is being measured.
    Events are read out once, at collection time, after a single synchronize.

    ``warmup_steps`` leading steps are recorded but dropped from the statistics.
    They cover cudnn/cublas autotuning, NCCL buffer allocation, the allocator
    reaching steady state, and (with ``--jit_mode=inductor``) compilation.

    When ``enabled`` is False every method is a no-op, so the calls can sit
    unconditionally in the training loops.
    """

    def __init__(self, enabled: bool = False, warmup_steps: int = 0, label: str = "step"):
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.label = label
        self.use_cuda_events = enabled and torch.cuda.is_available()
        self.reset()

    def reset(self):
        self._events = []
        self._host_times_ms = []
        self._open = None

    @property
    def num_steps(self) -> int:
        return len(self._events) if self.use_cuda_events else len(self._host_times_ms)

    def begin_step(self):
        if not self.enabled:
            return
        if self._open is not None:
            raise RuntimeError("StepTimer.begin_step() called twice without an intervening end_step()")
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._open = start
        else:
            self._open = time.perf_counter_ns()

    def end_step(self):
        if not self.enabled:
            return
        if self._open is None:
            raise RuntimeError("StepTimer.end_step() called without a matching begin_step()")
        if self.use_cuda_events:
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
            self._events.append((self._open, stop))
        else:
            self._host_times_ms.append((time.perf_counter_ns() - self._open) * 1e-6)
        self._open = None

    def timings_ms(self, include_warmup: bool = False) -> List[float]:
        """All recorded step times in milliseconds, warmup dropped by default."""
        if self.use_cuda_events:
            if self._events:
                torch.cuda.synchronize()
            times = [start.elapsed_time(stop) for start, stop in self._events]
        else:
            times = list(self._host_times_ms)

        if not include_warmup:
            times = times[self.warmup_steps :]

        return times


def summarize_timings(times_ms: Sequence[float]) -> Dict[str, Any]:
    """Distribution summary of a list of step times.

    The median is the headline number rather than the mean: a single step
    delayed by a stray allocation or a checkpoint of the host process should not
    move the reported figure.
    """
    if len(times_ms) == 0:
        return {"num_steps": 0}

    arr = np.asarray(times_ms, dtype=np.float64)

    return {
        "num_steps": int(arr.size),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def reduce_step_timings(times_ms: Sequence[float]) -> List[float]:
    """Element-wise max of the per-step times across all ranks.

    A collective step is only finished when its slowest participant is finished,
    so the step time of the run is the max over ranks -- taken per step, not on
    the already-summarized statistics, since which rank straggles can change
    from step to step.

    Ranks are truncated to the smallest common step count first. They should
    agree (the dummy loader shards evenly and the trainers ``drop_last``), but a
    mismatch here would otherwise deadlock rather than report.
    """
    if not dist.is_initialized() or comm.get_world_size() == 1:
        return list(times_ms)

    device = torch.device(f"cuda:{comm.get_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")

    count = torch.tensor([len(times_ms)], dtype=torch.long, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.MIN)
    num_steps = int(count.item())

    if num_steps == 0:
        return []

    local = torch.tensor(list(times_ms)[:num_steps], dtype=torch.float64, device=device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)

    return local.cpu().tolist()


def get_decomposition_record() -> Dict[str, Any]:
    """The decomposition ``comm`` actually built, not the one that was requested.

    The distinction matters: the data dimensions are auto-sized from ``-1`` in
    :func:`makani.utils.comm.init`, so what a launcher asked for and what the run
    used are not the same thing. Reporting the resolved sizes makes it
    impossible for the logged decomposition to disagree with the measured one.
    """
    sizes = {name: comm.get_size(name) for name in ["h", "w", "matmul", "ensemble", "batch"]}

    spec = "w{world}_h{h}w{w}m{matmul}e{ensemble}b{batch}".format(world=comm.get_world_size(), **sizes)

    record = dict(
        world_size=comm.get_world_size(),
        model=comm.get_size("model"),
        spatial=comm.get_size("spatial"),
        data=comm.get_size("data"),
        spec=spec,
        **sizes,
    )

    return record


def check_decomposition():
    """Fail loudly on a decomposition that does not tile the world.

    A mis-prescribed decomposition otherwise produces a run that looks healthy
    and reports a number for a configuration nobody asked for.
    """
    dec = get_decomposition_record()
    product = dec["h"] * dec["w"] * dec["matmul"] * dec["ensemble"] * dec["batch"]

    if product != dec["world_size"]:
        raise ValueError(
            f"decomposition h={dec['h']} w={dec['w']} matmul={dec['matmul']} ensemble={dec['ensemble']} "
            f"batch={dec['batch']} multiplies out to {product} ranks but the world has {dec['world_size']}."
        )

    return dec


def _run_git(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git"] + args,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if out.returncode != 0:
        return None

    return out.stdout.strip()


#: Vendor prefixes stripped for display, matching torch-harmonics' benchmarks/run.py::_fmt_arch
#: so the two suites name the same hardware identically.
_ARCH_STRIP = ("NVIDIA ", "AMD ", "Intel ")


def format_arch(name: Optional[str]) -> Optional[str]:
    """Strip the vendor prefix from a device name, as torch-harmonics' benchmarks do."""
    if not name:
        return name

    for prefix in _ARCH_STRIP:
        if name.startswith(prefix):
            return name[len(prefix) :]

    return name


def get_cpu_name() -> str:
    """A cleaned-up CPU model string, or 'Generic CPU' if it cannot be determined.

    Deliberately identical to ``_get_cpu_name`` in torch-harmonics' ``benchmarks/bench.py``,
    including the probe order and the cleaning rules, so a CPU reported by the makani benchmark
    and by the torch-harmonics benchmark is named the same string and the two sets of results
    can be lined up without a translation table.

    ``/proc/cpuinfo`` carries ``model name`` on x86 but not on aarch64, which is why ``lscpu``
    is tried next -- that is where Grace reports ``Neoverse-V2``.
    """

    def _clean(name: str) -> str:
        name = re.sub(r"\s+@\s+[\d.]+\s*GHz.*", "", name)  # drop "@ 2.00GHz" suffix
        name = name.replace("(R)", "").replace("(TM)", "")  # drop trademark noise
        return re.sub(r" {2,}", " ", name).strip()

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return _clean(line.split(":", 1)[1].strip())
    except OSError:
        pass

    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Model name"):
                return _clean(line.split(":", 1)[1].strip())
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out:
            return _clean(out)
    except Exception:
        pass

    name = platform.processor()

    return _clean(name) if name else "Generic CPU"


#: Names the driver returns for parts it has no entry for -- every pre-release board reports one
#: of these, so "the GPU is unnamed" has to be detected rather than assumed from the string.
_PLACEHOLDER_GPU_NAMES = ("NVIDIA Graphics Device", "Graphics Device", "NVIDIA Device")


def gpu_name_is_placeholder(name: Optional[str]) -> bool:
    """Whether the driver declined to name this part."""
    return (name is None) or (name.strip() in _PLACEHOLDER_GPU_NAMES)


def describe_gpu(props) -> str:
    """A factual fingerprint of a GPU, for parts the driver cannot name.

    Deliberately *not* a marketing name: mapping a compute capability to an architecture would
    be a guess, and a wrong one recorded in a results file is worse than none. Capability, SM
    count and memory are what the driver actually reports, they are stable across runs, and they
    are specific enough to tell two unnamed parts apart. Use ``--gpu_label`` to record the real
    name when a human knows it.
    """
    return f"sm{props.major}{props.minor}-{props.multi_processor_count}sm-{round(props.total_memory / 1024**3)}gb"


def get_environment_record(gpu_label: Optional[str] = None) -> Dict[str, Any]:
    """Software and hardware identity of the run.

    Carried as annotation rather than as part of the comparability hash: a run
    on a different machine is still comparable, it just has to be visible that
    it was one.
    """
    record = {
        "hostname": socket.gethostname(),
        # named exactly as torch-harmonics' benchmarks name it, so results from the two suites
        # can be joined on the hardware without a translation table
        "cpu_name": get_cpu_name(),
        "user": getpass.getuser(),
        # UTC, so runs from machines in different zones sort and compare directly
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_run_git(["status", "--porcelain", "--untracked-files=no"])),
        # CPU affinity, because a run launched under a pinning wrapper (bindpcie and friends)
        # and one launched without it differ by a real margin in step time -- host-side launch
        # overhead, dataloader workers and NUMA locality all move. Without this, the two are
        # indistinguishable in the results file.
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        record.update(
            {
                # Pre-release parts report a placeholder here ("NVIDIA Graphics Device"): the
                # driver's name table only carries shipped boards. The fields below identify the
                # chip well enough to tell such runs apart, and --gpu_label records what it
                # actually was when only a human knows.
                "gpu_name": props.name,
                "gpu_label": gpu_label or os.environ.get("MAKANI_BENCHMARK_GPU_LABEL", None),
                # what to call this part when neither the driver nor a human named it
                "gpu_name_is_placeholder": gpu_name_is_placeholder(props.name),
                "gpu_descriptor": describe_gpu(props),
                "gpu_memory_gb": round(props.total_memory / 1024**3, 1),
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_multiprocessor_count": props.multi_processor_count,
                "gpu_l2_cache_mb": (
                    round(props.L2_cache_size / 1024**2, 1) if hasattr(props, "L2_cache_size") else None
                ),
                "gpus_per_node": torch.cuda.device_count(),
            }
        )
        try:
            nccl_version = torch.cuda.nccl.version()
            record["nccl_version"] = ".".join(str(v) for v in nccl_version)
        except (AttributeError, RuntimeError):
            record["nccl_version"] = None
    else:
        record["gpu_name"] = None

    return record


#: Memory quantities collected per rank, in the order they are packed into the gather buffer.
_MEMORY_FIELDS = ("peak_allocated_gb", "peak_reserved_gb", "allocated_gb", "reserved_gb", "device_used_gb")


def get_memory_record() -> Dict[str, Any]:
    """This rank's device memory, in GiB.

    The ``peak_*`` figures are high-water marks since the last
    ``reset_peak_memory_stats``, which the benchmark issues just before the measured pass -- so
    they cover everything the steps touch, including workspaces allocated lazily on the first
    step (attention, cuDNN, NCCL) rather than at construction. Reading them after the pass is
    what makes that true; reading them earlier would miss whatever the steps allocate.

    ``device_used_gb`` comes from the driver rather than from torch's allocator, so unlike the
    other four it also counts the CUDA context, NCCL buffers and any cuBLAS/cuDNN workspace held
    outside the caching allocator. It is the number that matches nvidia-smi, and the one to look
    at when asking whether a decomposition fits.
    """
    if not torch.cuda.is_available():
        return {}

    free, total = torch.cuda.mem_get_info()

    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "device_used_gb": (total - free) / 1024**3,
    }


def gather_memory_record() -> Dict[str, Any]:
    """The per-GPU memory footprint across the job, as min/max/mean per quantity.

    Ranks do not all carry the same memory: a spatial decomposition leaves the polar ranks with
    different shard shapes, and rank 0 additionally holds whatever the logging and reporting
    path allocates. A single max hides that, so each quantity is reported as its spread over
    ranks -- max answers "does it fit", min and mean together say how evenly the decomposition
    loaded the GPUs.
    """
    local = get_memory_record()

    if not local:
        return {}

    if not dist.is_initialized() or comm.get_world_size() == 1:
        return {field: {"min": local[field], "max": local[field], "mean": local[field]} for field in _MEMORY_FIELDS}

    device = torch.device(f"cuda:{comm.get_local_rank()}")
    values = torch.tensor([local[field] for field in _MEMORY_FIELDS], dtype=torch.float64, device=device)

    gathered = torch.empty((comm.get_world_size(), len(_MEMORY_FIELDS)), dtype=torch.float64, device=device)
    dist.all_gather_into_tensor(gathered, values)
    gathered = gathered.cpu()

    record = {}
    for index, field in enumerate(_MEMORY_FIELDS):
        column = gathered[:, index]
        record[field] = {
            "min": column.min().item(),
            "max": column.max().item(),
            "mean": column.mean().item(),
        }

    return record


# Keys dropped before hashing a configuration for comparability. These are the
# knobs that a benchmark suite is expected to vary between runs (decomposition,
# world size, per-rank batch), plus bookkeeping that says nothing about what was
# computed (paths, logging, checkpoint format, worker counts). Everything else
# -- crucially every architecture key, whatever the nettype calls them -- is
# hashed, so this is a blocklist rather than an allowlist: a new model
# hyperparameter is part of the identity of a run by default.
_COMPARABILITY_EXCLUDED_KEYS = frozenset(
    {
        # decomposition and run geometry
        "batch_size",
        "local_ensemble_size",
        "world_size",
        "h_parallel_size",
        "w_parallel_size",
        "matmul_parallel_size",
        "ensemble_parallel_size",
        "model_parallel_sizes",
        "model_parallel_names",
        "data_parallel_sizes",
        "data_parallel_names",
        "parameters_reduction_buffer_count",
        "io_grid",
        "io_rank",
        "data_shard_id",
        "data_num_shards",
        # run length -- a benchmark caps it, it is not a property of the model
        "max_epochs",
        "n_train_samples",
        "n_train_samples_per_epoch",
        "n_eval_samples",
        "n_eval_samples_per_epoch",
        "num_samples_per_epoch",
        "n_years",
        # paths and dataset bookkeeping
        "exp_dir",
        "experiment_dir",
        "checkpoint_path",
        "best_checkpoint_path",
        "metadata_json_path",
        "train_data_path",
        "valid_data_path",
        "inf_data_path",
        "min_path",
        "max_path",
        "time_means_path",
        "global_means_path",
        "global_stds_path",
        "time_diff_means_path",
        "time_diff_stds_path",
        "invariants_path",
        "maskpath",
        "h5_path",
        "dataset",
        "lat",
        "lon",
        "dhours",
        # io and runtime plumbing
        "enable_synthetic_data",
        "enable_s3",
        "enable_odirect",
        "odirect_alignment",
        "multifiles",
        "dataset_backend",
        "num_data_workers",
        "num_visualization_workers",
        "print_timings_frequency",
        "verbose",
        "resuming",
        "pretrained",
        "load_checkpoint",
        "save_checkpoint",
        "checkpoint_num_versions",
        "skip_validation",
        "skip_training",
        "disable_ddp",
        "enable_grad_anomaly_detection",
        "dump_weights_and_grads",
        "split_data_channels",
        "run_num",
        "save_raw_forecasts",
        "save_channel",
        "load_loss",
    }
)

_COMPARABILITY_EXCLUDED_PREFIXES = ("wandb", "log_", "benchmark_", "capture_", "img_crop_")


def _is_comparability_key(key: str) -> bool:
    if key.startswith("_"):
        return False
    if key in _COMPARABILITY_EXCLUDED_KEYS:
        return False
    if key.startswith(_COMPARABILITY_EXCLUDED_PREFIXES):
        return False
    return True


def get_comparability_keys(params, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The subset of the resolved config that two runs must share to be comparable."""
    keys = {k: v for k, v in params.to_dict().items() if _is_comparability_key(k)}

    if extra is not None:
        keys.update(extra)

    return keys


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def hash_dict(obj: Dict[str, Any], length: int = 16) -> str:
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()[:length]


def hash_file(path: str, length: int = 16) -> Optional[str]:
    if (path is None) or (not os.path.isfile(path)):
        return None

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()[:length]


def generate_metadata_json(params, path: str, dhours: int = 1, grid_type: Optional[str] = None) -> str:
    """Write a dataset descriptor for the synthetic loader, derived from the config.

    A checked-in descriptor would have to pin a channel list, and
    :func:`makani.utils.parse_dataset_metada.parse_dataset_metadata` validates
    every name in the config's ``channel_names`` against ``coords.channel`` --
    so a static file would silently restrict the suite to the models whose
    channel naming it happens to match. Deriving it from the resolved config
    instead means any config benchmarks out of the box.

    Coordinates are deliberately omitted: the parser synthesizes them from
    ``coords.grid_type`` and the config's ``img_shape_x``/``img_shape_y``, which
    is the documented path for dummy-data experiments and keeps the resolution
    defined in exactly one place (the model config).
    """
    # The checks below use ``in`` (item access) rather than ``is_set`` (attribute access) on
    # purpose: ``ParamsBase`` does not mirror attribute assignment into its dict, and
    # ``parse_dataset_metadata`` -- the consumer of this descriptor -- indexes ``params`` by
    # item. Validating through the attribute path would let a params object that satisfies this
    # function still fail with a bare KeyError from inside the parser.
    if ("img_shape_x" not in params) or ("img_shape_y" not in params):
        raise ValueError(
            "Generating a synthetic dataset descriptor requires img_shape_x and img_shape_y in the model "
            "config: without coordinates in the descriptor, the grid is reconstructed from the declared "
            "grid type and those shapes. Add them to the config, or pass --metadata_json_path explicitly."
        )

    if ("channel_names" not in params) or (not params["channel_names"]):
        raise ValueError(
            "Generating a synthetic dataset descriptor requires channel_names in the model config, since "
            "the descriptor's channel list is derived from it. Pass --metadata_json_path explicitly to use "
            "a real dataset descriptor instead."
        )

    channel_names = params["channel_names"]

    if grid_type is None:
        grid_type = params.get("data_grid_type", None) or "equiangular"

    # A config may name the dataset it is standing in for (e.g. ICON_R02B09approx). Without one
    # the shape is the only thing distinguishing two synthetic runs in the results file, which is
    # too implicit to rely on once more than one grid is being benchmarked.
    dataset_name = params.get("dataset_name", None) or "makani-benchmark-synthetic"

    metadata = {
        "dataset_name": dataset_name,
        "attrs": {"description": "synthetic dataset descriptor generated by makani/benchmark.py"},
        "h5_path": "fields",
        "dims": ["time", "channel", "lat", "lon"],
        "dhours": dhours,
        "coords": {
            "grid_type": grid_type,
            "channel": list(channel_names),
        },
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    return path


def append_record(path: str, record: Dict[str, Any]):
    """Append one run to a JSON-lines result file, creating it if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def format_record(record: Dict[str, Any]) -> str:
    """Human-readable one-run summary for the log."""
    dec = record["decomposition"]
    batching = record["batching"]
    timing = record["timing"]
    memory = record.get("memory", {})

    lines = [
        "",
        "=" * 72,
        f"benchmark: {record['mode']}  {record['config']['yaml_path']}:{record['config']['config_name']}",
        "=" * 72,
        f"  model               : {record['model']['nettype']} "
        f"({record['model']['num_parameters'] / 1e6:.1f}M parameters)",
        f"  shape               : {record['model']['img_shape_x']} x {record['model']['img_shape_y']} x "
        f"{record['model']['n_in_channels']} in / {record['model']['n_out_channels']} out",
        f"  decomposition       : h={dec['h']} w={dec['w']} matmul={dec['matmul']} "
        f"ensemble={dec['ensemble']} batch={dec['batch']} (world={dec['world_size']})",
        f"  global batch size   : {batching['global_batch_size']} " f"(local {batching['local_batch_size']})",
        f"  ensemble size       : {batching['ensemble_size']} (local {batching['local_ensemble_size']})",
        f"  steps               : {timing['num_steps']} measured, {record['benchmark']['warmup_steps']} warmup",
        f"  step time (median)  : {timing['median_ms']:.1f} ms",
        f"  step time (p10/p90) : {timing['p10_ms']:.1f} / {timing['p90_ms']:.1f} ms",
        f"  throughput          : {record['throughput']['samples_per_second']:.2f} samples/s",
    ]

    if memory:
        # min/max/mean over GPUs: max says whether it fits, the spread says how evenly the
        # decomposition loaded them
        for field, label in (
            ("peak_allocated_gb", "peak allocated"),
            ("peak_reserved_gb", "peak reserved"),
            ("device_used_gb", "device used"),
        ):
            stats = memory.get(field)
            if stats:
                lines.append(
                    f"  {label:<20}: {stats['max']:.1f} GiB max, {stats['mean']:.1f} GiB mean, "
                    f"{stats['min']:.1f} GiB min  (over {dec['world_size']} GPUs)"
                )

    lines += [
        f"  spec                : {dec['spec']}",
        f"  comparability hash  : {record['config']['comparability_hash']}",
        "=" * 72,
        "",
    ]

    return "\n".join(lines)
