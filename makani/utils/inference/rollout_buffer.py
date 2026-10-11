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

import ctypes
from math import prod
from typing import Optional, List, Tuple, Union
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
import h5py as h5

# distributed computing stuff
from torch import amp
import torch.distributed as dist
from makani.utils import comm
from makani.utils.grids import get_spectral_transform
from makani.models.common import RealFFT1
from makani.mpu.fft import DistributedRealFFT1
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim
from makani.mpu.mappings import reduce_from_parallel_region

# get torch_harmonics for spectra
import torch_harmonics as th
import torch_harmonics.distributed as thd


def _barrier(device):
    """
    Synchronize across ranks if a process group is initialized.

    Wraps ``dist.barrier`` with two safety guards: it's a no-op when no
    process group has been initialized, and it only passes ``device_ids``
    on CUDA (``device.index`` is ``None`` for CPU and would make
    ``dist.barrier(device_ids=[None])`` raise).
    """
    if not dist.is_initialized():
        return
    device_ids = [device.index] if device.type == "cuda" else None
    dist.barrier(device_ids=device_ids)


def _driver_available(name: str) -> bool:
    """
    Report whether this h5py/HDF5 build can open files with the ``name`` VFD.

    Neither optional driver we care about is guaranteed to exist: ``direct`` is
    compiled in only when h5py is built with ``H5PY_DIRECT_VFD=1`` against an
    HDF5 configured with ``--enable-direct-vfd``, and ``gds`` is not upstream at
    all — it ships as an out-of-tree plugin. h5py resolves driver names against a
    registry that the build and any plugins populate at import time, so querying
    it is the honest capability check: it follows whatever image we happen to be
    running in, with nothing for the caller to keep in sync.
    """
    try:
        return name in h5.registered_drivers()
    except Exception:
        # very old h5py without the driver registry — treat as unsupported
        return False


def odirect_available() -> bool:
    """Whether the HDF5 ``direct`` VFD (O_DIRECT writes) is usable here."""
    return _driver_available("direct")


def gds_available() -> bool:
    """Whether the HDF5 ``gds`` VFD (GPUDirect Storage writes) is usable here."""
    return _driver_available("gds")


def _unavailable_driver_message(flag: str, driver: str, how: str) -> str:
    """Actionable message for a VFD the caller asked for but this image lacks."""
    return (
        f"{flag}=True but this HDF5/h5py build has no '{driver}' driver "
        f"(registered drivers: {sorted(h5.registered_drivers())}). {how} "
        f"Drop {flag} to use the default POSIX writer."
    )


def _as_numpy(tensor: torch.Tensor, ctype, nptype):
    """
    Reinterpret a (CUDA) tensor's ``data_ptr()`` as a numpy view without a host copy.

    Used by the GDS write path: h5py's GDS VFD only consumes the pointer for cuFile
    DMA setup and never dereferences it as host memory, so a "numpy view" of GPU
    memory is safe to hand off. Caller must keep the source tensor alive for the
    duration of the write and must have synchronized any kernels that produced
    its data — there is no implicit copy/sync here.
    """
    pointer = ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctype))
    buff = (pointer._type_ * prod(tensor.shape)).from_address(ctypes.addressof(pointer.contents))
    return np.frombuffer(buff, dtype=nptype).reshape(*tensor.shape)


class DataBuffer(object, metaclass=ABCMeta):
    r"""
    DataBuffer class used as base class for online data analysis
    """

    def __init__(
        self,
        num_rollout_steps: int,
        rollout_dt: int,
        channel_names: List[str],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
    ):

        # store members
        self.num_rollout_steps = num_rollout_steps
        self.rollout_dt = rollout_dt
        self.output_channels = output_channels
        self.output_file = output_file

        # get mapping from channels to all channels
        self.channel_mask = [channel_names.index(c) for c in self.output_channels]
        self.num_channels = len(self.output_channels)

        if not self.num_channels > 0:
            raise ValueError(f"Empty channel lists aren't suported. Got {self.output_channels}.")

        # set device
        self.device = device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # deal with output scale and bias
        if scale is not None:
            scale = scale.squeeze()
            self.scale = scale[self.channel_mask].to(self.device)
        else:
            self.scale = torch.ones(self.num_channels, device=self.device)
        if bias is not None:
            bias = bias.squeeze()
            self.bias = bias[self.channel_mask].to(self.device)
        else:
            self.bias = torch.zeros(self.num_channels, device=self.device)

        # instantiate communicator:
        if self.output_file is not None:
            # set up communicator if requested
            if comm.get_world_size() > 1:
                # initialize MPI. This call is collective!
                from mpi4py import MPI

                self.mpi_comm = MPI.COMM_WORLD.Split(color=0, key=comm.get_world_rank())
            else:
                self.mpi_comm = None

        return

    @abstractmethod
    def zero_buffers(self):
        pass

    @abstractmethod
    def update(self, pred, tstamps, idt):
        pass

    @abstractmethod
    def finalize(self):
        pass


class RolloutBuffer(DataBuffer):
    r"""
    RolloutBuffer class handles the recording of selected channels during the rollout and manages associated buffers.

    Parameters
    ============
    num_samples : int
        Total number of initial conditions to be recorded
    batch_size : int
        Maximum local batch size
    num_rollout_steps : int
        Number of total rollout steps
    ensemble_size : int
        Ensemble size
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    output_memory_buffer_size: int, optional
        Number of (initial-condition × leadtime) steps to cache in memory before writing to disk.
        With ``valid_autoreg_steps = R``, each initial condition produces ``R + 1`` steps, so a
        full rollout for one IC fits in ``R + 1`` slots. Setting ``output_memory_buffer_size``
        below ``R + 1`` enables mid-rollout flushing, which is useful when a single trajectory
        does not fit in memory. Clamped to ``[batch_size, num_samples * (R + 1)]``. Defaults
        to ``num_samples * (R + 1)`` (i.e., buffer the entire run). A value of ``0`` (or any
        non-positive integer) is a legacy shorthand for ``streaming_mode=True`` — no in-memory
        buffer is allocated and each ``update()`` writes straight to disk; this preserves the
        behavior of dispatcher scripts that pass ``--output_memory_buffer_size=0``.
    streaming_mode: bool, optional
        When ``True``, every ``update()`` call writes directly to disk and no
        in-memory rollout buffer is allocated. Requires ``output_file`` to be
        set. Default is ``False``.
    buffer_device: str or torch.device, optional
        Device on which the in-memory rollout buffer is allocated. Defaults
        to ``torch.device("cpu")``, which uses pinned memory when ``device``
        is CUDA. Pass a CUDA device to keep the buffer GPU-resident and
        avoid GPU→CPU transfers on every ``update()``; the buffer is copied
        to host once at flush time. Ignored when ``streaming_mode`` is
        ``True``.
    enable_odirect: bool, optional
        When ``True``, open the output file with HDF5's ``direct`` VFD so
        writes bypass the host page cache (O_DIRECT). Requires an HDF5 build
        with the direct VFD compiled in (``--enable-direct-vfd``) and
        ``output_file`` to be set. Incompatible with the MPI-IO driver used
        for distributed writes; use only on single-rank or per-rank file
        layouts. Mutually exclusive with ``enable_gds``. Default is ``False``.
    odirect_alignment: int, optional
        Alignment (in bytes) used for the HDF5 ``direct`` VFD's ``alignment``
        and ``block_size`` parameters. ``0`` uses the HDF5 defaults. Only
        relevant when ``enable_odirect`` is ``True``.
    enable_gds: bool, optional
        When ``True``, open the output file with the HDF5 ``gds`` driver and
        route ``write_direct`` calls through a numpy view of GPU memory
        (zero-copy DMA via cuFile/nvidia-fs) instead of staging through host.
        Requires an HDF5 build with the GDS VFD, a CUDA-resident write source
        (``buffer_device=cuda`` for buffered mode; ``device=cuda`` for
        streaming), and that the source memory be cuFile-compatible
        (classic ``cudaMalloc``, not VMM-backed). Mutually exclusive with
        ``enable_odirect``. Default is ``False``.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        num_rollout_steps: int,
        rollout_dt: int,
        ensemble_size: int,
        img_shape: Tuple[int, int],
        local_shape: Tuple[int, int],
        local_offset: Tuple[int, int],
        channel_names: List[str],
        lat_lon: Tuple[List[float], List[float]],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        streaming_mode: bool = False,
        buffer_device: Union[str, torch.device] = torch.device("cpu"),
        enable_odirect: bool = False,
        odirect_alignment: int = 0,
        enable_gds: bool = False,
    ):
        super().__init__(
            num_rollout_steps, rollout_dt, channel_names, device, scale, bias, output_channels, output_file
        )

        # O_DIRECT (HDF5 ``direct`` VFD): bypasses the host page cache. Requires an
        # output file and a non-MPI single-writer layout — the direct VFD is mutually
        # exclusive with ``mpio``. Validation against the MPI driver happens later
        # in ``_create_output_file`` where we know whether ``mpi_comm`` was created.
        if enable_odirect and output_file is None:
            raise ValueError("enable_odirect=True requires output_file to be set.")
        # Fail here rather than deep inside h5py: asking for a VFD this image was not
        # built with is a setup mistake, and "Unknown driver type" does not say so.
        if enable_odirect and not odirect_available():
            raise RuntimeError(
                _unavailable_driver_message(
                    "enable_odirect",
                    "direct",
                    "Rebuild h5py with H5PY_DIRECT_VFD=1 against an HDF5 configured with --enable-direct-vfd.",
                )
            )
        self.enable_odirect = enable_odirect
        self.odirect_alignment = odirect_alignment

        # GDS path requires an output file (no in-memory equivalent) and a CUDA source
        # for the writes; validation against buffer_device/device is deferred to where
        # the actual write happens since the user may want streaming OR buffered GDS.
        if enable_gds and output_file is None:
            raise ValueError("enable_gds=True requires output_file to be set.")
        # O_DIRECT and GDS open the file with different VFDs; only one can win.
        if enable_odirect and enable_gds:
            raise ValueError("enable_odirect and enable_gds are mutually exclusive — pick one I/O backend.")
        # Fail here rather than deep inside h5py: asking for GDS on an image that
        # lacks the VFD is a setup mistake, and "Unknown driver type" does not say so.
        if enable_gds and not gds_available():
            raise RuntimeError(
                _unavailable_driver_message(
                    "enable_gds",
                    "gds",
                    "The GDS VFD is not upstream yet — it needs an image built with the "
                    "out-of-tree vfd-gds plugin (see docker/Dockerfile.internal.gds).",
                )
            )
        self.enable_gds = enable_gds

        # streaming mode has no in-memory fallback — every update writes to disk.
        # Two ways to opt in:
        #   - explicit streaming_mode=True (preferred for new code)
        #   - output_memory_buffer_size <= 0 (legacy shorthand from tkurth/gds-support;
        #     preserved so existing inference dispatchers that pass
        #     ``--output_memory_buffer_size=0`` keep working without code changes).
        # ``output_memory_buffer_size=None`` (the unset default) does NOT trigger
        # streaming — it still means "buffer the whole run".
        streaming_via_size = (output_memory_buffer_size is not None) and (output_memory_buffer_size <= 0)
        self.streaming_mode = streaming_mode or streaming_via_size
        if self.streaming_mode and output_file is None:
            raise ValueError(
                "streaming mode requires output_file to be set; " "there is no in-memory buffer in streaming mode."
            )

        # resolve buffer device (mirrors DataBuffer's str→torch.device coercion).
        # CUDA buffers avoid GPU→CPU transfers per update() call but pay one
        # bulk copy at flush time.
        self.buffer_device = torch.device(buffer_device) if isinstance(buffer_device, str) else buffer_device

        # GDS hands raw pointers to cuFile, so the write source must live on the GPU:
        # streaming writes straight from ``device``, buffered writes from ``buffer_device``.
        if self.enable_gds:
            gds_source = self.device if self.streaming_mode else self.buffer_device
            if gds_source.type != "cuda":
                which = "device" if self.streaming_mode else "buffer_device"
                raise ValueError(f"enable_gds=True requires {which} to be a CUDA device, got {gds_source}.")

        # store additional members
        self.img_shape = img_shape
        self.local_shape = local_shape
        self.local_offset = local_offset
        self.lat_lon = lat_lon
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.output_channels = output_channels
        # buffer size is in STEP units (one slot = one (IC, leadtime) pair × E ensemble × C × H × W).
        # default: buffer the whole run; clamp floor=batch_size (must fit one update() call),
        # ceiling=num_samples * (R + 1) (no point allocating beyond the full run).
        full_steps = num_samples * (num_rollout_steps + 1)
        self.num_buffered_samples = output_memory_buffer_size if output_memory_buffer_size is not None else full_steps
        self.num_buffered_samples = max(min(self.num_buffered_samples, full_steps), batch_size)

        # little hacky but we use this to compute the range where to write the output to
        if comm.is_distributed("batch") and comm.get_size("batch") > 1:
            num_samples = torch.tensor([self.num_samples], dtype=torch.long, device=self.device)
            num_samples_list = [num_samples.clone() for _ in range(comm.get_size("batch"))]
            dist.all_gather(num_samples_list, num_samples, group=comm.get_group("batch"))
            num_samples = torch.cumsum(torch.cat(num_samples_list, dim=0), dim=0).cpu().numpy()
            self.num_samples_offsets = [
                0,
            ] + num_samples.tolist()
        else:
            self.num_samples_offsets = [0, self.num_samples]
        self.num_samples_total = self.num_samples_offsets[-1]

        # rollout buffer is flat over (IC × leadtime) — one slot per step.
        # shape: num_steps x ensemble_size x num_channels x nlat x nlon
        # per-slot (ic, idt) bookkeeping lives in ``self.chunks`` (a list of batch-group
        # records), not in parallel index tensors — the producer's natural fill pattern is
        # B parallel ICs progressing through their R+1 leadtimes in lockstep, which the
        # chunk records capture compactly and let us coalesce into per-batch slab writes.
        if self.streaming_mode:
            # No in-memory buffer; each update() flushes to disk directly.
            self.rollout_data = None
            self.timestamp_data = None
        else:
            # pin_memory only applies to CPU buffers, and is only useful when the
            # compute device is CUDA (it accelerates async H2D/D2H transfers).
            pin_memory = (self.buffer_device.type == "cpu") and (self.device.type == "cuda")
            local_buffer_size = (self.num_buffered_samples, self.ensemble_size, self.num_channels, *self.local_shape)
            self.rollout_data = torch.zeros(
                local_buffer_size, dtype=torch.float32, device=self.buffer_device, pin_memory=pin_memory
            )
            # worst case (R+1 == 1) every slot is a distinct IC, so the timestamp buffer
            # is sized to the data buffer; in practice only a fraction is used per flush.
            self.timestamp_data = torch.zeros(
                (self.num_buffered_samples,), dtype=torch.float64, device=self.buffer_device, pin_memory=pin_memory
            )

        # open output_file
        self.file_handle = None
        if self.output_file is not None:
            self._create_output_file(self.output_file)

        # set up the local IC file pointer unconditionally — update() consults it to
        # snapshot each chunk's ic_offset and to advance at end-of-rollout, regardless
        # of whether there is an actual file to write to (no-file mode is used by the
        # in-memory unit tests).
        self.file_offset = self.num_samples_offsets[comm.get_rank("batch")]

        # initialize buffer offsets
        self.buffer_offset = 0

        # per-batch chunk records, each a dict with keys:
        #   start_slot  — first data slot in ``rollout_data`` owned by this chunk
        #   ic_offset   — file IC offset for the first IC of this chunk
        #   batch_size  — number of parallel ICs (B) in this chunk
        #   idt_start   — first leadtime index this chunk covers (0 normally; > 0 for a
        #                 mid-rollout carry-over chunk created by a flush)
        #   idt_count   — leadtimes written so far for this chunk
        # slot layout within one chunk: slot[start_slot + j*B + i] == (ic[i], idt_start + j)
        self.chunks = []

        # pending timestamps that have not yet been flushed to disk.
        # within one flush cycle, the buffered timestamps occupy slots [0, ts_buffer_offset)
        # of ``timestamp_data`` and map to file IC offsets
        # [ts_first_ic_offset, ts_first_ic_offset + ts_buffer_offset) — one contiguous
        # slab because each rank processes its ICs sequentially.
        self.ts_buffer_offset = 0
        self.ts_first_ic_offset = None

        # reshape bias and scale
        self.bias = self.bias.reshape(-1, 1, 1)
        self.scale = self.scale.reshape(-1, 1, 1)

    def _create_output_file(self, output_file):
        if self.mpi_comm is not None:
            # MPI-IO is mutually exclusive with the direct VFD; the user picks one.
            if self.enable_odirect:
                raise ValueError(
                    "enable_odirect=True is incompatible with the MPI-IO driver "
                    "(used whenever world_size > 1). Use a per-rank file layout or disable O_DIRECT."
                )
            # Same story for GDS: one driver per file handle, and mpio has already won.
            # Without this the flag would be silently ignored on multi-rank runs.
            if self.enable_gds:
                raise ValueError(
                    "enable_gds=True is incompatible with the MPI-IO driver "
                    "(used whenever world_size > 1). Use a per-rank file layout or disable GDS."
                )
            # initialize MPI. This call is collective!
            self.file_handle = h5.File(output_file, "w", driver="mpio", comm=self.mpi_comm)
        elif self.enable_odirect:
            # HDF5 ``direct`` VFD: writes go through O_DIRECT, bypassing the page cache.
            driver_kwargs = (
                dict(alignment=self.odirect_alignment, block_size=self.odirect_alignment)
                if self.odirect_alignment > 0
                else {}
            )
            self.file_handle = h5.File(output_file, "w", driver="direct", **driver_kwargs)
        elif self.enable_gds:
            # GDS VFD: writes go straight from GPU memory to storage via cuFile/nvidia-fs.
            self.file_handle = h5.File(output_file, "w", driver="gds")
        else:
            self.file_handle = h5.File(output_file, "w")

        # create hdf5 dataset
        total_buffer_size = (
            self.num_samples_total,
            self.num_rollout_steps + 1,
            self.ensemble_size * comm.get_size("ensemble"),
            self.num_channels,
            *self.img_shape,
        )
        self.rollout_buffer_disk = self.file_handle.create_dataset("fields", total_buffer_size, dtype=np.float32)

        # create timestamps for scale
        self.timestamp_buffer_disk = self.file_handle.create_dataset(
            "timestamp", (self.num_samples_total), dtype=np.float64
        )
        self.timestamp_buffer_disk.make_scale("timestamp")
        self.rollout_buffer_disk.dims[0].attach_scale(self.timestamp_buffer_disk)

        # create leadtimes
        leadtime = self.file_handle.create_dataset("lead_time", (self.num_rollout_steps + 1), dtype=np.float64)
        leadtime.make_scale("lead_time")
        self.rollout_buffer_disk.dims[1].attach_scale(leadtime)
        if comm.get_world_rank() == 0:
            dts = np.arange(0, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64)
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = self.file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        self.rollout_buffer_disk.dims[3].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

        # create lon and lat descriptors
        lats = self.file_handle.create_dataset("lat", len(self.lat_lon[0]), dtype=np.float32)
        lats.make_scale("lat")
        self.rollout_buffer_disk.dims[4].attach_scale(lats)
        if comm.get_world_rank() == 0:
            lats[...] = np.array(self.lat_lon[0], dtype=np.float32)
        lons = self.file_handle.create_dataset("lon", len(self.lat_lon[1]), dtype=np.float32)
        lons.make_scale("lon")
        self.rollout_buffer_disk.dims[5].attach_scale(lons)
        if comm.get_world_rank() == 0:
            lons[...] = np.array(self.lat_lon[1], dtype=np.float32)

        # label dimensions for more information
        self.rollout_buffer_disk.dims[0].label = "Timestamp in UTC time zone"
        self.rollout_buffer_disk.dims[1].label = "Lead time relative to timestamp"
        self.rollout_buffer_disk.dims[2].label = "Ensemble index"
        self.rollout_buffer_disk.dims[3].label = "Channel name"
        self.rollout_buffer_disk.dims[4].label = "Latitude in degrees"
        self.rollout_buffer_disk.dims[5].label = "Longitude in degrees"

        return

    # close the output file
    def __del__(self):
        # ``__del__`` may run after a partial-init failure (e.g. constructor
        # validation raised before ``self.file_handle`` was assigned). Use
        # getattr so a missing attribute doesn't shadow the original error.
        file_handle = getattr(self, "file_handle", None)
        if file_handle is not None:
            file_handle.close()
            self.file_handle = None
        return

    def zero_buffers(self):
        """
        Reset all buffer state to a fresh post-construction state. Zeros the
        data tensors AND resets ``buffer_offset`` so callers may invoke this
        mid-rollout without leaving a stale offset pointing into newly-zeroed
        slots. Also clears the chunk records and pending-timestamp bookkeeping.

        Does NOT reset ``file_offset`` — callers calling this mid-flow keep
        their position in the output file.

        In streaming mode there are no in-memory buffers, so this is a no-op.
        """

        if self.streaming_mode:
            return

        with torch.no_grad():
            self.timestamp_data.fill_(0.0)
            self.rollout_data.fill_(0.0)
        self.buffer_offset = 0
        self.chunks = []
        self.ts_buffer_offset = 0
        self.ts_first_ic_offset = None

        return

    _CTYPE_BY_DTYPE = {torch.float32: (ctypes.c_float, np.float32), torch.float64: (ctypes.c_double, np.float64)}

    def _np_view_for_write(self, tensor: torch.Tensor):
        """
        Produce the numpy buffer to hand to ``write_direct``.

        With GDS, returns a zero-copy view of GPU memory (and pre-syncs CUDA so any
        producer kernels have committed). The view does not own the memory, so the
        tensor must already be contiguous and the caller must keep it alive until the
        write completes — making a contiguous copy here would hand cuFile a pointer
        to a temporary that may be freed before the write. Without GDS, returns a
        contiguous host array — bringing the tensor over from a non-CPU device if necessary.
        """
        if self.enable_gds:
            if (tensor.device.type != "cuda") or (not tensor.is_contiguous()):
                raise ValueError(
                    f"GDS writes need a contiguous CUDA tensor, got device={tensor.device}, "
                    f"contiguous={tensor.is_contiguous()}."
                )
            torch.cuda.synchronize(device=tensor.device)
            ctype, nptype = self._CTYPE_BY_DTYPE[tensor.dtype]
            return _as_numpy(tensor, ctype, nptype)
        src = tensor if tensor.device.type == "cpu" else tensor.cpu()
        return np.ascontiguousarray(src.numpy())

    def _write_chunk_to_disk(self, chunk):
        """
        Write one batch-group chunk to the HDF5 ``fields`` dataset.

        The chunk holds ``B`` parallel ICs at leadtimes ``[idt_start, idt_start + idt_count)``.
        Slots in the data buffer follow an (idt-major, ic-minor) layout
        (``slot = start_slot + j*B + i``), so we reshape into
        ``(idt_count, B, E, C, H, W)`` and transpose to ``(B, idt_count, ...)`` to match
        the file's (IC-major, idt-minor) layout. One slab write per chunk.
        """

        idt_count = chunk["idt_count"]
        if idt_count == 0:
            return

        start_slot = chunk["start_slot"]
        B = chunk["batch_size"]
        ic_offset = chunk["ic_offset"]
        idt_start = chunk["idt_start"]

        # gather the chunk's slab from the data buffer
        slab = self.rollout_data[start_slot : start_slot + B * idt_count]
        slab = slab.reshape(idt_count, B, self.ensemble_size, self.num_channels, *self.local_shape)
        # transpose to (B, idt_count, E, C, H, W) to match the file's IC-major layout
        slab = slab.transpose(0, 1).contiguous()

        # GDS path keeps slab on GPU and hands the pointer to cuFile;
        # non-GDS path stages through host.
        slab_arr = self._np_view_for_write(slab)

        # destination slice in the file
        ic_range = slice(ic_offset, ic_offset + B)
        idt_range = slice(idt_start, idt_start + idt_count)
        ens_start = self.ensemble_size * comm.get_rank("ensemble")
        ens_range = slice(ens_start, ens_start + self.ensemble_size)
        lat_range = slice(self.local_offset[0], self.local_offset[0] + self.local_shape[0])
        lon_range = slice(self.local_offset[1], self.local_offset[1] + self.local_shape[1])

        self.rollout_buffer_disk.write_direct(
            slab_arr,
            dest_sel=np.s_[ic_range, idt_range, ens_range, :, lat_range, lon_range],
        )

        return

    def _flush_buffer_to_disk(self):
        """
        Write the accumulated in-memory buffer to disk (non-streaming mode).

        Data is written one batch-group chunk at a time (coalesced slab per chunk).
        Timestamps were buffered contiguously and are written as a single slab.
        If the most recent chunk's rollout is incomplete (i.e. the buffer filled mid-rollout),
        a carry-over chunk is installed so the next ``update()`` for the in-flight batch
        continues writing into the freshly-zeroed buffer at the correct idt offset.
        """

        # wait for everything to complete (CUDA only — synchronize is a CUDA-specific
        # primitive and would raise on CPU-only builds).
        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)

        if (self.file_handle is not None) and (self.buffer_offset > 0):
            # write each batch chunk as one coalesced slab
            for chunk in self.chunks:
                self._write_chunk_to_disk(chunk)

            # write the pending timestamps (one contiguous slab per flush)
            if (self.ts_buffer_offset > 0) and (comm.get_rank("model") == 0) and (comm.get_rank("ensemble") == 0):
                timestamp_arr = self._np_view_for_write(self.timestamp_data)
                ts_buf_range = slice(0, self.ts_buffer_offset)
                ts_file_range = slice(self.ts_first_ic_offset, self.ts_first_ic_offset + self.ts_buffer_offset)
                self.timestamp_buffer_disk.write_direct(
                    timestamp_arr,
                    source_sel=ts_buf_range,
                    dest_sel=ts_file_range,
                )

        # determine whether the last chunk's rollout is incomplete; if so, the next
        # update() for the same batch continues into a freshly-zeroed buffer and we
        # need a carry-over chunk to remember where in the file it goes.
        carry = None
        if self.chunks:
            last = self.chunks[-1]
            completed_idts = last["idt_start"] + last["idt_count"]
            if completed_idts < (self.num_rollout_steps + 1):
                carry = {
                    "start_slot": 0,
                    "ic_offset": last["ic_offset"],
                    "batch_size": last["batch_size"],
                    "idt_start": completed_idts,
                    "idt_count": 0,
                }

        # reset all buffer state (data, timestamps, chunks, offsets); file_offset preserved
        self.zero_buffers()

        if carry is not None:
            self.chunks.append(carry)

        return

    def _write_to_disk(self, tstamps, predp, idt):
        """Write a single (batch, leadtime) slice straight to disk (streaming mode)."""

        if self.file_handle is None:
            return

        current_batch_size = predp.shape[0]

        # batch range in file (uses current file_offset, advanced after the last leadtime)
        batch_range = slice(self.file_offset, self.file_offset + current_batch_size)

        # ensemble range
        ens_start = self.ensemble_size * comm.get_rank("ensemble")
        ens_range = slice(ens_start, ens_start + self.ensemble_size)

        # spatial range
        lat_range = slice(self.local_offset[0], self.local_offset[0] + self.local_shape[0])
        lon_range = slice(self.local_offset[1], self.local_offset[1] + self.local_shape[1])

        # GDS path keeps data on GPU and hands the pointer to cuFile;
        # non-GDS path copies to host first. Bind the contiguous tensors to locals so
        # they outlive the zero-copy GDS views until write_direct returns.
        predp = predp.detach().contiguous()
        tstamps = tstamps.detach().contiguous()
        predp_arr = self._np_view_for_write(predp)

        self.rollout_buffer_disk.write_direct(
            predp_arr,
            dest_sel=np.s_[batch_range, idt, ens_range, :, lat_range, lon_range],
        )

        # write timestamps once per IC (on the first leadtime), single-writer
        if (idt == 0) and (comm.get_rank("model") == 0) and (comm.get_rank("ensemble") == 0):
            tarr = self._np_view_for_write(tstamps)
            self.timestamp_buffer_disk.write_direct(
                tarr,
                dest_sel=batch_range,
            )

        return

    def update(self, pred, tstamps, idt):
        """
        Record one (leadtime × B-IC × E-ensemble) prediction tile.

        ``pred`` has shape ``(B, E, C, H, W)`` and represents B parallel ICs at leadtime
        ``idt``. In non-streaming mode this consumes B slots in the flat data buffer; if
        the buffer would overflow, we flush first. A carry-over chunk left behind by a
        mid-rollout flush is continued transparently.
        """

        # get the current batch size
        current_batch_size = pred.shape[0]

        # check if we can buffer the next element or if we need to flush now
        # (only meaningful in non-streaming mode — streaming has no buffer to fill)
        if (not self.streaming_mode) and (self.buffer_offset + current_batch_size > self.num_buffered_samples):
            self._flush_buffer_to_disk()

        with torch.no_grad():
            predp = self.scale * pred[..., self.channel_mask, :, :] + self.bias

            if self.streaming_mode:
                self._write_to_disk(tstamps, predp, idt)
                # advance file pointer when this IC's last leadtime is done
                if (idt + 1) == (self.num_rollout_steps + 1):
                    self.file_offset += current_batch_size
            else:
                # at idt == 0 we are starting a NEW batch of B initial conditions:
                # open a fresh chunk record and stash this batch's timestamps. A
                # carry-over chunk from a mid-rollout flush keeps idt > 0, so it
                # falls through to the data-write branch and reuses the existing
                # chunks[-1] entry installed at flush time.
                if idt == 0:
                    chunk = {
                        "start_slot": self.buffer_offset,
                        "ic_offset": self.file_offset,
                        "batch_size": current_batch_size,
                        "idt_start": 0,
                        "idt_count": 0,
                    }
                    self.chunks.append(chunk)

                    # buffer this batch's timestamps; they form one contiguous slab with
                    # earlier pending batches in the current flush cycle.
                    if self.ts_first_ic_offset is None:
                        self.ts_first_ic_offset = self.file_offset
                    ts_slot = self.ts_buffer_offset
                    self.timestamp_data[ts_slot : ts_slot + current_batch_size].copy_(tstamps, non_blocking=True)
                    self.ts_buffer_offset += current_batch_size

                # write the (B, E, C, H, W) tile into the flat buffer
                batch_start = self.buffer_offset
                batch_end = batch_start + current_batch_size
                self.rollout_data[batch_start:batch_end].copy_(predp, non_blocking=True)
                self.buffer_offset = batch_end

                # advance the active chunk's leadtime counter
                self.chunks[-1]["idt_count"] += 1

                # advance the file IC pointer once this batch's rollout completes
                if (idt + 1) == (self.num_rollout_steps + 1):
                    self.file_offset += current_batch_size

        return

    def finalize(self):

        _barrier(self.device)

        # Flush remaining buffered data when there's both a file to write to and
        # an in-memory buffer to flush. ``_flush_buffer_to_disk`` doubles as a
        # destructive reset; in the no-file path it just zeros the buffer.
        # Calling it here without a file would silently destroy in-memory data
        # the caller may want to read. In streaming mode there's nothing to
        # flush — every ``update()`` already wrote directly to disk.
        if (not self.streaming_mode) and (self.file_handle is not None):
            self._flush_buffer_to_disk()

        _barrier(self.device)

        # Close the output file and null the handle so a subsequent finalize()
        # or __del__ doesn't try to operate on an already-closed h5py object.
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        return


class MeanStdBuffer(DataBuffer):
    r"""
    MeanStdBuffer class handles the recording of bias during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    variable_shape: Union[Tuple[int], Tuple[int, int]],
        shape of the variable tensor to be averaged for each rollout step and channel
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """

    def __init__(
        self,
        num_rollout_steps: int,
        rollout_dt: int,
        variable_shape: Union[Tuple[int], Tuple[int, int]],
        channel_names: List[str],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
    ):
        super().__init__(
            num_rollout_steps, rollout_dt, channel_names, device, scale, bias, output_channels, output_file
        )

        # rollout buffer on CPU has dimensions num_rollout_steps x num_channels x nlat x nlon
        pin_memory = self.device.type == "cuda"
        local_buffer_size = (self.num_rollout_steps, self.num_channels, *variable_shape)
        self.running_mean = torch.zeros(local_buffer_size, dtype=torch.float32, device=self.device)
        self.running_var = torch.zeros(local_buffer_size, dtype=torch.float32, device=self.device)

        # recall how many variable dims we have
        self.variable_dims = len(variable_shape)

        # number of samples seen
        self.num_samples_tracked = torch.zeros(
            (self.num_rollout_steps, 1, *(1 for _ in range(self.variable_dims))), dtype=torch.int64, device=self.device
        )

    def zero_buffers(self):
        """
        set buffers to zero
        """
        with torch.no_grad():
            self.running_mean.fill_(0.0)
            self.running_var.fill_(0.0)
            self.num_samples_tracked.fill_(0)

        return

    def _compute_stats(self, data, dim=0):
        count = torch.tensor(data.shape[dim], dtype=torch.int64, device=self.device).reshape(
            1, *(1 for _ in range(self.variable_dims))
        )
        var, mean = torch.var_mean(data, dim=dim, correction=0, keepdim=False)
        m2 = var * count

        return mean, m2, count

    def _welford_combine(self, mean, m2, count, idt) -> torch.Tensor:
        count_new = self.num_samples_tracked[idt] + count
        delta = mean - self.running_mean[idt]
        self.running_mean[idt, ...] += delta * float(count) / float(count_new)
        self.running_var[idt, ...] += m2 + torch.square(delta) * float(self.num_samples_tracked[idt] * count) / float(
            count_new
        )
        self.num_samples_tracked[idt, ...] = count_new

        return

    def _aggregate_stats(self, group_name="data"):
        if comm.get_size(group_name) > 1:
            with torch.no_grad():
                # extract tensors: we need to be SUPER CAREFUL:
                # reduce_from_parallel-region is seemingly inplace when called from within
                # no_grad region, so we need to create copies here
                counts_old = self.num_samples_tracked.clone()
                means_old = self.running_mean.clone()
                m2s_old = self.running_var.clone()

                # counts are: n = sum_k n_k
                counts_agg = reduce_from_parallel_region(self.num_samples_tracked, group_name)
                # means are: mu = sum_i n_i * mu_i / n
                # normalize before reducing to avoid overflows
                meansc = counts_old * means_old / counts_agg.to(dtype=torch.float32)
                means_agg = reduce_from_parallel_region(meansc, group_name)

                # m2s are: sum_i m2_i + sum_i n_i * (mu_i - mu)^2
                m2s_agg = reduce_from_parallel_region(m2s_old, group_name)
                deltas = torch.square(means_old - means_agg) * counts_old
                deltas_agg = reduce_from_parallel_region(deltas, group_name)
                m2s_agg += deltas_agg

                # write back
                self.num_samples_tracked[...] = counts_agg[...]
                self.running_mean[...] = means_agg[...]
                self.running_var[...] = m2s_agg[...]

        return


class TemporalAverageBuffer(MeanStdBuffer):
    r"""
    temporalAverageBuffer class handles the recording of spatial data during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """

    def __init__(
        self,
        num_rollout_steps: int,
        rollout_dt: int,
        img_shape: Tuple[int, int],
        local_shape: Tuple[int, int],
        local_offset: Tuple[int, int],
        channel_names: List[str],
        lat_lon: Tuple[List[float], List[float]],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
    ):

        super().__init__(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            variable_shape=local_shape,
            channel_names=channel_names,
            device=device,
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=output_file,
        )

        self.img_shape = img_shape
        self.local_shape = local_shape
        self.local_offset = local_offset
        self.lat_lon = lat_lon

        # reshape bias and scale
        self.bias = self.bias.reshape(1, -1, 1, 1)
        self.scale = self.scale.reshape(1, -1, 1, 1)

    def _write_output_data(self, mean, std):

        if self.mpi_comm is not None:
            # initialize MPI. This call is collective!
            file_handle = h5.File(self.output_file, "w", driver="mpio", comm=self.mpi_comm)
        else:
            file_handle = h5.File(self.output_file, "w")

        # create hdf5 dataset and write the data
        data_shape = (self.num_rollout_steps, self.num_channels, self.img_shape[0], self.img_shape[1])
        mean_data = file_handle.create_dataset("mean", data_shape, dtype=np.float64)
        std_data = file_handle.create_dataset("std", data_shape, dtype=np.float64)

        # spatial range
        lat_range = slice(self.local_offset[0], self.local_offset[0] + self.local_shape[0])
        lon_range = slice(self.local_offset[1], self.local_offset[1] + self.local_shape[1])

        if comm.get_rank("data") == 0:
            mean_data[..., lat_range, lon_range] = mean[...]
            std_data[..., lat_range, lon_range] = std[...]

        # create leadtimes
        leadtime = file_handle.create_dataset("lead_time", self.num_rollout_steps, dtype=np.float64)
        leadtime.make_scale("lead_time")
        mean_data.dims[0].attach_scale(leadtime)
        std_data.dims[0].attach_scale(leadtime)
        dts = np.arange(
            self.rollout_dt, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64
        )
        if comm.get_world_rank() == 0:
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        mean_data.dims[1].attach_scale(chans)
        std_data.dims[1].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

        # create lon and lat descriptors
        lats = file_handle.create_dataset("lat", len(self.lat_lon[0]), dtype=np.float32)
        lats.make_scale("lat")
        mean_data.dims[2].attach_scale(lats)
        std_data.dims[2].attach_scale(lats)
        if comm.get_world_rank() == 0:
            lats[...] = np.array(self.lat_lon[0], dtype=np.float32)
        lons = file_handle.create_dataset("lon", len(self.lat_lon[1]), dtype=np.float32)
        lons.make_scale("lon")
        mean_data.dims[3].attach_scale(lons)
        std_data.dims[3].attach_scale(lons)
        if comm.get_world_rank() == 0:
            lons[...] = np.array(self.lat_lon[1], dtype=np.float32)

        # label dimensions for more information
        mean_data.dims[0].label = "Lead time"
        mean_data.dims[1].label = "Channel name"
        mean_data.dims[2].label = "Latitude in degrees"
        mean_data.dims[3].label = "Longitude in degrees"
        std_data.dims[0].label = "Lead time"
        std_data.dims[1].label = "Channel name"
        std_data.dims[2].label = "Latitude in degrees"
        std_data.dims[3].label = "Longitude in degrees"

        # close file
        file_handle.close()

        _barrier(self.device)

        return

    def update(self, data, idt):
        """update local buffers"""

        with torch.no_grad():
            data_projected = self.scale * data[..., self.channel_mask, :, :] + self.bias

            # compute the local variance and mean over the local batch dimension
            mean, m2, count = self._compute_stats(data_projected, dim=0)

            # do distributed welford
            self._welford_combine(mean, m2, count, idt)

        return

    def finalize(self):

        _barrier(self.device)

        # compute common mean
        self._aggregate_stats("data")

        std = torch.sqrt(self.running_var / (self.num_samples_tracked.to(dtype=torch.float32) - 1.0))
        mean = self.running_mean

        # write out bias/var
        if self.output_file is not None:
            self._write_output_data(mean.cpu().numpy(), std.cpu().numpy())

        _barrier(self.device)

        return


class SpectrumAverageBuffer(MeanStdBuffer):
    r"""
    SpectrumAverageBuffer class handles the recording of spatial data during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """

    def __init__(
        self,
        num_rollout_steps: int,
        rollout_dt: int,
        img_shape: Tuple[int, int],
        ensemble_size: int,
        grid_type: str,
        channel_names: List[str],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        spatial_distributed: Optional[bool] = False,
    ):

        # instantiate SHT
        self.spatial_distributed = spatial_distributed
        if self.spatial_distributed and (comm.get_size("spatial") > 1):
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
            self.sht = get_spectral_transform("sht", *img_shape, grid_type, distributed=True)
            self.lmax = self.sht.lmax
            self.lmax_local = self.sht.l_shapes[comm.get_rank("h")]
            # compute loffset
            offsets = [0] + np.cumsum(self.sht.l_shapes).tolist()
            self.lmax_offset = offsets[comm.get_rank("h")]
        else:
            self.sht = get_spectral_transform("sht", *img_shape, grid_type)
            self.lmax = self.sht.lmax
            self.lmax_local = self.lmax
            self.lmax_offset = 0

        # determine parameter size
        self.ensemble_size = ensemble_size
        enssize = self.ensemble_size
        if comm.get_rank("ensemble") == 0:
            enssize += 1
        variable_shape = (enssize, self.lmax_local)

        super().__init__(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            variable_shape=variable_shape,
            channel_names=channel_names,
            device=device,
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=output_file,
        )

        # move sht to device
        self.sht = self.sht.to(self.device)

        # reshape bias and scale
        self.bias = self.bias.reshape(1, 1, -1, 1, 1)
        self.scale = self.scale.reshape(1, 1, -1, 1, 1)

    def update(self, data, targ, idt):
        """update local buffers"""

        with torch.no_grad():
            if comm.get_rank("ensemble") == 0:
                data = torch.cat([targ, data], dim=1)

            # rescale data and project channels
            datap = self.scale * data[..., self.channel_mask, :, :] + self.bias

            # perform SHT in FP32 to be safe
            dtype = datap.dtype
            with amp.autocast(device_type=datap.device.type, enabled=False):
                sdatap = self.sht(datap.to(torch.float32))

                # compute power spectrum:
                sdatap = torch.square(torch.abs(sdatap))

                # be w-distributed aware:
                if comm.get_rank("w") == 0:
                    sdatap[..., 1:] *= 2.0
                else:
                    sdatap[...] *= 2.0

                # local sum over w
                spect = torch.sum(sdatap, dim=-1)

                # do distributed sum over w
                spect = reduce_from_parallel_region(spect, "w")

            # convert back precision
            spect = spect.to(dtype=dtype)

            # swap E and C
            spect = spect.permute(0, 2, 1, 3).contiguous()

            # compute the local variance and mean over the local batch dimension
            mean, m2, count = self._compute_stats(spect, dim=0)

            # do distributed welford
            self._welford_combine(mean, m2, count, idt)

        return

    def finalize(self):

        _barrier(self.device)

        # compute common mean
        self._aggregate_stats("batch")

        std = torch.sqrt(self.running_var / (self.num_samples_tracked.to(dtype=torch.float32) - 1.0))
        mean = self.running_mean

        # write out bias/var
        if self.output_file is not None:
            self._write_output_data(mean.cpu().numpy(), std.cpu().numpy())

        _barrier(self.device)

        return

    def _write_output_data(self, mean, std):

        if self.mpi_comm is not None:
            # initialize MPI. This call is collective!
            file_handle = h5.File(self.output_file, "w", driver="mpio", comm=self.mpi_comm)
        else:
            file_handle = h5.File(self.output_file, "w")

        # create hdf5 dataset and write the data
        data_shape = (
            self.num_rollout_steps,
            self.num_channels,
            self.ensemble_size * comm.get_size("ensemble") + 1,
            self.lmax,
        )
        mean_data = file_handle.create_dataset("mean", data_shape, dtype=np.float64)
        std_data = file_handle.create_dataset("std", data_shape, dtype=np.float64)

        # l range
        l_start = self.lmax_offset
        l_end = l_start + self.lmax_local
        l_range = slice(l_start, l_end)

        # ensemble range
        if comm.get_size("ensemble") > 1:
            if comm.get_rank("ensemble") == 0:
                ens_start = 0
                ens_end = self.ensemble_size + 1
            else:
                ens_start = comm.get_rank("ensemble") * self.ensemble_size + 1
                ens_end = ens_start + self.ensemble_size
            ens_range = slice(ens_start, ens_end)
        else:
            ens_range = slice(0, self.ensemble_size + 1)

        if (comm.get_rank("batch") == 0) and (comm.get_rank("w") == 0):
            mean_data[..., ens_range, l_range] = mean[...]
            std_data[..., ens_range, l_range] = std[...]

        # create leadtimes
        leadtime = file_handle.create_dataset("lead_time", self.num_rollout_steps, dtype=np.float64)
        leadtime.make_scale("lead_time")
        mean_data.dims[0].attach_scale(leadtime)
        std_data.dims[0].attach_scale(leadtime)
        dts = np.arange(
            self.rollout_dt, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64
        )
        if comm.get_world_rank() == 0:
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        mean_data.dims[1].attach_scale(chans)
        std_data.dims[1].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

        # label dimensions for more information
        mean_data.dims[0].label = "Lead time"
        std_data.dims[0].label = "Lead time"
        mean_data.dims[1].label = "Channel name"
        std_data.dims[1].label = "Channel name"
        mean_data.dims[2].label = "Ensemble index"
        std_data.dims[2].label = "Ensemble index"
        mean_data.dims[3].label = "Spectral mode"
        std_data.dims[3].label = "Spectral mode"

        # close file
        file_handle.close()

        _barrier(self.device)

        return


class ZonalSpectrumAverageBuffer(MeanStdBuffer):
    r"""
    SpectrumAverageBuffer class handles the recording of spatial data during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    lat_lon: Tuple[List[float], List[float]]
        latitude and longitude coordinates
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """

    def __init__(
        self,
        num_rollout_steps: int,
        rollout_dt: int,
        img_shape: Tuple[int, int],
        ensemble_size: int,
        channel_names: List[str],
        lat_lon: Tuple[List[float], List[float]],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        spatial_distributed: Optional[bool] = False,
    ):

        # instantiate SHT
        self.spatial_distributed = spatial_distributed and comm.get_size("spatial") > 1
        self.nlat = img_shape[0]
        if self.spatial_distributed:
            if comm.get_size("w") > 1:
                self.rfft = DistributedRealFFT1(nlon=img_shape[1])
                m_shapes = self.rfft.m_shapes
                self.mmax_local = m_shapes[comm.get_rank("w")]
                # compute loffset
                offsets = [0] + np.cumsum(m_shapes).tolist()
                self.mmax_offset = offsets[comm.get_rank("w")]
            else:
                self.rfft = RealFFT1(nlon=img_shape[1])
                self.mmax_local = self.rfft.mmax
                self.mmax_offset = 0
            nlat_shapes = compute_split_shapes(self.nlat, comm.get_size("h"))
            self.nlat_local = nlat_shapes[comm.get_rank("h")]
            # compute loffset
            offsets = [0] + np.cumsum(nlat_shapes).tolist()
            self.nlat_offset = offsets[comm.get_rank("h")]
        else:
            self.rfft = RealFFT1(nlon=img_shape[1])
            self.mmax_local = self.rfft.mmax
            self.mmax_offset = 0
            self.nlat_local = self.nlat
            self.nlat_offset = 0

        # store lat/lon data
        self.lat_lon = lat_lon
        self.mmax = self.rfft.mmax

        # determine parameter size
        self.ensemble_size = ensemble_size
        enssize = self.ensemble_size
        if comm.get_rank("ensemble") == 0:
            enssize += 1
        variable_shape = (enssize, self.nlat_local, self.mmax_local)

        super().__init__(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            variable_shape=variable_shape,
            channel_names=channel_names,
            device=device,
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=output_file,
        )

        # reshape bias and scale
        self.bias = self.bias.reshape(1, 1, -1, 1, 1)
        self.scale = self.scale.reshape(1, 1, -1, 1, 1)

        # we need to compute the latitude weighting factors
        lats = torch.tensor(self.lat_lon[0], dtype=torch.float32, device=self.device)
        # we need to use cos since we use latitude
        self.lat_weights = torch.cos(torch.deg2rad(lats))
        # split the tensor in lat dim
        if self.spatial_distributed and comm.get_size("h") > 1:
            self.lat_weights = split_tensor_along_dim(self.lat_weights, dim=0, num_chunks=comm.get_size("h"))[
                comm.get_rank("h")
            ]
        # reshape:
        self.lat_weights = torch.reshape(self.lat_weights, (1, 1, 1, -1, 1))

    def update(self, data, targ, idt):
        """update local buffers"""

        with torch.no_grad():
            if comm.get_rank("ensemble") == 0:
                data = torch.cat([targ, data], dim=1)

            # rescale data and project channels:
            datap = self.scale * data[..., self.channel_mask, :, :] + self.bias

            # distributed FFT
            dtype = datap.dtype
            with amp.autocast(device_type=datap.device.type, enabled=False):
                sdatap = self.rfft(datap.to(torch.float32), norm="forward")

                # compute power spectrum:
                spect = self.lat_weights * torch.square(torch.abs(sdatap))

                # be w-distributed aware:
                if comm.get_rank("w") == 0:
                    spect[..., 1:] *= 2.0
                else:
                    spect[...] *= 2.0

            # convert back precision
            spect = spect.to(dtype=dtype)

            # swap E and C
            spect = spect.permute(0, 2, 1, 3, 4).contiguous()

            # compute the local variance and mean over the local batch dimension
            mean, m2, count = self._compute_stats(spect, dim=0)

            # do distributed welford
            self._welford_combine(mean, m2, count, idt)

        return

    def finalize(self):

        _barrier(self.device)

        # compute common mean
        self._aggregate_stats("batch")

        std = torch.sqrt(self.running_var / (self.num_samples_tracked.to(dtype=torch.float32) - 1.0))
        mean = self.running_mean

        # write out bias/var
        if self.output_file is not None:
            self._write_output_data(mean.cpu().numpy(), std.cpu().numpy())

        _barrier(self.device)

        return

    def _write_output_data(self, mean, std):

        if self.mpi_comm is not None:
            # initialize MPI. This call is collective!
            file_handle = h5.File(self.output_file, "w", driver="mpio", comm=self.mpi_comm)
        else:
            file_handle = h5.File(self.output_file, "w")

        # create hdf5 dataset and write the data
        data_shape = (
            self.num_rollout_steps,
            self.num_channels,
            self.ensemble_size * comm.get_size("ensemble") + 1,
            self.nlat,
            self.mmax,
        )
        mean_data = file_handle.create_dataset("mean", data_shape, dtype=np.float64)
        std_data = file_handle.create_dataset("std", data_shape, dtype=np.float64)

        # lat range
        lat_range = slice(self.nlat_offset, self.nlat_offset + self.nlat_local)
        # m-range
        m_range = slice(self.mmax_offset, self.mmax_offset + self.mmax_local)

        # ensemble range
        if comm.get_size("ensemble") > 1:
            if comm.get_rank("ensemble") == 0:
                ens_start = 0
                ens_end = self.ensemble_size + 1
            else:
                ens_start = comm.get_rank("ensemble") * self.ensemble_size + 1
                ens_end = ens_start + self.ensemble_size
            ens_range = slice(ens_start, ens_end)
        else:
            ens_range = slice(0, self.ensemble_size + 1)

        if comm.get_rank("batch") == 0:
            mean_data[..., ens_range, lat_range, m_range] = mean[...]
            std_data[..., ens_range, lat_range, m_range] = std[...]

        # create leadtimes
        leadtime = file_handle.create_dataset("lead_time", self.num_rollout_steps, dtype=np.float64)
        leadtime.make_scale("lead_time")
        mean_data.dims[0].attach_scale(leadtime)
        std_data.dims[0].attach_scale(leadtime)
        dts = np.arange(
            self.rollout_dt, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64
        )
        if comm.get_world_rank() == 0:
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        mean_data.dims[1].attach_scale(chans)
        std_data.dims[1].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

        # create lon and lat descriptors
        lats = file_handle.create_dataset("lat", len(self.lat_lon[0]), dtype=np.float32)
        lats.make_scale("lat")
        mean_data.dims[3].attach_scale(lats)
        std_data.dims[3].attach_scale(lats)
        if comm.get_world_rank() == 0:
            lats[...] = np.array(self.lat_lon[0], dtype=np.float32)

        # label dimensions for more information
        mean_data.dims[0].label = "Lead time"
        std_data.dims[0].label = "Lead time"
        mean_data.dims[1].label = "Channel name"
        std_data.dims[1].label = "Channel name"
        mean_data.dims[2].label = "Ensemble index"
        std_data.dims[2].label = "Ensemble index"
        mean_data.dims[3].label = "Latitude"
        std_data.dims[3].label = "Latitude"
        mean_data.dims[4].label = "Spectral mode"
        std_data.dims[4].label = "Spectral mode"

        # close file
        file_handle.close()

        _barrier(self.device)

        return
