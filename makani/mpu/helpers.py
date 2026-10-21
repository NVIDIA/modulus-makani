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

import torch
import torch.distributed as dist
from torch_harmonics.distributed.primitives import _gather
from makani.utils import comm


@torch.compiler.disable
def gather_uneven(tensor, dim, comm_name):
    # nothing to gather, and inside comm.serial() the group is None, which would gather over the world
    if comm.get_size(comm_name) == 1:
        return tensor

    # the shards may differ in size along dim; torch-harmonics' gather exchanges the sizes first
    # and pads the shards to the largest on gloo, which only gathers tensors of equal shape
    return _gather(tensor.contiguous(), dim, None, group=comm.get_group(comm_name))


def all_reduce_mean(tensor, group):
    """In-place all-reduce to the mean over ``group``. gloo has no ``ReduceOp.AVG``, so it sums and divides there."""
    if dist.get_backend(group) == dist.Backend.GLOO:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
        tensor.div_(dist.get_world_size(group=group))
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=group, async_op=False)
    return tensor


def barrier(device):
    """Barrier across all ranks, if torch.distributed is initialized.

    NCCL is given the device explicitly rather than inferring it from the ambient CUDA state;
    on CPU there is no device index to give.
    """
    if not dist.is_initialized():
        return
    device_ids = [device.index] if device.type == "cuda" else None
    dist.barrier(device_ids=device_ids)


def sync_params(model, mode="broadcast"):
    """Helper routine to ensure shared weights are the same after initialization"""

    def _sync_param(param, comm_group, mode):
        if comm.get_size(comm_group) > 1:
            if mode == "broadcast":
                is_complex = param.is_complex()
                if is_complex:
                    param_real = torch.view_as_real(param).clone()
                else:
                    param_real = param.clone()
                # tlist = [torch.empty_like(param_real) for x in range(comm.get_size(comm_group))]
                # tlist[comm.get_rank(comm_group)] = param_real
                # gather all weights in the comm group
                dist.broadcast(
                    param_real, src=comm.get_root(comm_group), group=comm.get_group(comm_group), async_op=False
                )
                # use weight of rank 0
                # important to use copy here otherwise the handle gets detaches from the optimizer
                if is_complex:
                    param.copy_(torch.view_as_complex(param_real))
                else:
                    param.copy_(param_real)
            elif mode == "mean":
                param_real = torch.view_as_real(param) if param.is_complex() else param
                all_reduce_mean(param_real, comm.get_group(comm_group))
            else:
                raise ValueError(f"Unknown weight synchronization mode {mode}")

        return

    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():
            # share along data dim
            _sync_param(param, "data", mode)

            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]

            for comm_group in param.is_shared_mp:
                _sync_param(param, comm_group, mode)

    # synchronize the device to make sure all copies have finished
    barrier(next(model.parameters()).device)

    return
