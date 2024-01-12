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

import types
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from makani.utils import comm

# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


# handler for additional gradient reductions
# helper for gradient reduction across channel parallel ranks
def init_gradient_reduction_hooks(
    model, device_ids, output_device, bucket_cap_mb=25, broadcast_buffers=True, find_unused_parameters=False, gradient_as_bucket_view=True, static_graph=False
):
    # early exit if we are not in a distributed setting:
    if not dist.is_initialized():
        return model

    # set this to false in init and then find out if we can use it:
    need_hooks = False
    ddp_group = comm.get_group("data")

    # this is the trivial case
    if comm.get_size("model") == 1:
        # the simple case, we can just continue then
        ddp_group = None
    else:
        # count parameters and reduction groups
        num_parameters_total = 0
        num_parameters_shared_model = 0
        for param in model.parameters():
            # if it does not have any annotation, we assume it is shared between all model ranks
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]
                # non_singleton_group_names.copy()

            # add the sharing type to the dict
            num_parameters_total += 1
            if "model" in param.is_shared_mp:
                num_parameters_shared_model += 1

        # if all parameters are shared between all model ranks, then the situation is easy
        if num_parameters_shared_model == num_parameters_total:
            # we can always use DDP
            ddp_group = None

            # register some pre-multiply reduction hooks
            print("Setting up gradient hooks to account for shared parameter multiplicity")
            for param in model.parameters():
                param.register_hook(lambda grad: grad * float(comm.get_size("model")))
        else:
            ddp_group = comm.get_group("data")
            broadcast_buffers = False
            need_hooks = True

    # WAR: this is a workaround required for complex tensors
    # it is used as a workaround for bad complex tensors
    # DDP support in torch
    if not need_hooks:
        for param in model.parameters():
            if param.is_complex():
                need_hooks = True
                broadcast_buffers = False
                break

    # we can set up DDP and exit here
    print("Setting up DDP communication hooks")
    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        bucket_cap_mb=bucket_cap_mb,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=ddp_group,
    )

    if not need_hooks:
        return model

    print("Setting up custom communication hooks")

    # define comm hook:
    def reduction_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        # allreduce everything first:
        buff = bucket.buffer()
        params = bucket.parameters()

        # define the grad reduction function
        def grad_reduction(fut, grads, group, reduction="sum"):
            # check if grads are complex
            is_complex = [g.is_complex() for g in grads]
            grads_real = [torch.view_as_real(g) if g.is_complex() else g for g in grads]

            # flatten
            coalesced = _flatten_dense_tensors(grads_real)

            # reduce
            if reduction == "sum":
                dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=comm.get_group(group), async_op=False)
            elif reduction == "mean":
                dist.all_reduce(coalesced, op=dist.ReduceOp.AVG, group=comm.get_group(group), async_op=False)
            else:
                raise NotImplementedError(f"Error, reduction {reduction} not supported.")

            # copy back
            for buf, synced_real, is_comp in zip(grads, _unflatten_dense_tensors(coalesced, grads_real), is_complex):
                if is_comp:
                    synced = torch.view_as_complex(synced_real)
                else:
                    synced = synced_real
                buf.copy_(synced)

            return bucket.buffer()

        # WAR: we need to add a workaround for complex gradients here, therefore we need to hack the allreduce step a little bit.
        # once this is fixed, the below line can be uncommented and we can remove the hack
        # get future for allreduce
        # fut = dist.all_reduce(buff, op=dist.ReduceOp.AVG, group=comm.get_group("data"), async_op=True).get_future()

        # get future
        fut = torch.futures.Future()
        fut.set_result(bucket.buffer())

        # get the data gradients first:
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.data)

        if grads:
            fut = fut.then(lambda x: grad_reduction(x, grads=grads, group="data", reduction="mean"))

        # now go through the groups
        for group in comm.get_names():
            if group == "data":
                continue

            # real first
            grads = []
            for p in params:
                if (p.grad is not None) and (group in p.is_shared_mp):
                    grads.append(p.grad.data)

            # append the new reduction functions
            if grads:
                fut = fut.then(lambda x: grad_reduction(x, grads=grads, group=group, reduction="sum"))

        return fut

    # register model comm hook
    model.register_comm_hook(state=None, hook=reduction_comm_hook)

    return model
