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
from torch.nn.parallel import DistributedDataParallel
from torch_harmonics.distributed import compute_split_shapes
from torch_harmonics.distributed.primitives import (
    _gather,
    _split,
    _reduce,
    _transpose,
)

# bridge so new-style autograd.Function (separate setup_context) works with
# torch.amp.custom_fwd/custom_bwd; see makani/mpu/_amp_utils.py (pytorch#132388).
from makani.mpu._amp_utils import _custom_setup_context

# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# import config
from makani.mpu.config import config


class _DistributedTranspose(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(x, dims, dim1_split_sizes, comm_id):
        # WAR for a potential contig check torch bug for channels last contig tensors
        x = x.contiguous()
        xlist, _, _ = _transpose(x, dims[0], dims[1], dim1_split_sizes, group=comm.get_group(comm_id))
        x = torch.cat(xlist, dim=dims[1]).contiguous()
        return x

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, dims, dim1_split_sizes, comm_id = inputs
        ctx.dims = dims
        ctx.comm_id = comm_id
        # dim0_split_sizes is what _transpose would compute from splitting x along dims[0]
        ctx.dim0_split_sizes = compute_split_shapes(x.shape[dims[0]], comm.get_size(comm_id))

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, go):
        dims = ctx.dims
        dim0_split_sizes = ctx.dim0_split_sizes
        # WAR for a potential contig check torch bug for channels last contig tensors
        go = go.contiguous()
        gilist, _, _ = _transpose(go, dims[1], dims[0], dim0_split_sizes, group=comm.get_group(ctx.comm_id))
        gi = torch.cat(gilist, dim=dims[0]).contiguous()
        return gi, None, None, None


class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region"""

    @staticmethod
    def symbolic(graph, input_, comm_id_):
        return input_

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(input_, comm_id_):
        return input_

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        _, comm_id_ = inputs
        ctx.comm_id = comm_id_

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=comm.get_group(ctx.comm_id)), None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region"""

    @staticmethod
    def symbolic(graph, input_, comm_id_):
        return _reduce(input_, group=comm.get_group(comm_id_))

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(input_, comm_id_):
        return _reduce(input_, group=comm.get_group(comm_id_))

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        pass  # nothing needed for backward

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the chunk corresponding to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_):
        return _split(input_, dim_, group=comm.get_group(comm_id_))

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(input_, dim_, comm_id_):
        return _split(input_, dim_, group=comm.get_group(comm_id_))

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        input_, dim_, comm_id_ = inputs
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        ctx.split_shapes = compute_split_shapes(input_.shape[dim_], comm.get_size(comm_id_))

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return (
            _gather(
                grad_output,
                ctx.dim,
                ctx.split_shapes,
                group=comm.get_group(ctx.comm_id),
            ),
            None,
            None,
        )


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_, shapes_):
        return _gather(input_, dim_, shapes_, group=comm.get_group(comm_id_))

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(input_, dim_, shapes_, comm_id_):
        return _gather(input_, dim_, shapes_, group=comm.get_group(comm_id_))

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        input_, dim_, shapes_, comm_id_ = inputs
        ctx.dim = dim_
        ctx.comm_id = comm_id_

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return (
            _split(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)),
            None,
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------
@torch.compiler.disable
def copy_to_parallel_region(input, comm_id):
    """Copy input"""
    # a single rank has nothing to communicate; this also keeps the backward from reducing
    # without a process group, or over the whole world inside comm.serial()
    if comm.get_size(comm_id) == 1:
        return input
    return _CopyToParallelRegion.apply(input, comm_id)


@torch.compiler.disable
def reduce_from_parallel_region(input, comm_id):
    """All-reduce the input from the matmul parallel region."""
    if comm.get_size(comm_id) == 1:
        return input
    return _ReduceFromParallelRegion.apply(input, comm_id)


@torch.compiler.disable
def scatter_to_parallel_region(input, dim, comm_id):
    """Split the input and keep only the corresponding chuck to the rank."""
    if comm.get_size(comm_id) == 1:
        return input
    return _ScatterToParallelRegion.apply(input, dim, comm_id)


@torch.compiler.disable
def gather_from_parallel_region(input, dim, shapes, comm_id):
    """Gather the input from matmul parallel region and concatenate."""
    if comm.get_size(comm_id) == 1:
        return input
    return _GatherFromParallelRegion.apply(input, dim, shapes, comm_id)


@torch.compiler.disable
def distributed_transpose(x, dims, shapes, comm_id):
    r"""
    Redistribute a tensor by swapping which dimension is sharded.

    Performs an all-to-all so that the dimension currently split across the
    group becomes local and a currently local dimension becomes split. This is
    the primitive behind distributed transforms: an operation that needs a full
    axis (an FFT along longitude, say) first transposes so that axis is local,
    trading away locality of another one.

    Differentiable -- the backward pass performs the inverse redistribution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, sharded along ``dims[0]``.
    dims : tuple of int
        ``(src, dst)``: the dimension currently sharded and the one to shard
        instead.
    shapes : list of int
        Per-rank sizes along the dimension being gathered, needed because the
        split may be uneven.
    comm_id : str
        Name of the communicator group to redistribute over.

    Returns
    -------
    torch.Tensor
        The redistributed tensor, now sharded along ``dims[1]``.
    """
    if comm.get_size(comm_id) == 1:
        return x
    return _DistributedTranspose.apply(x, dims, shapes, comm_id)


class gradient_print_wrapper(torch.autograd.Function):
    r"""
    Identity in the forward pass that prints gradient statistics in the backward.

    Debugging aid: insert it anywhere in a model to see the min, max and mean of
    the gradient flowing through that point, without altering the computation.
    Useful for locating where gradients vanish or blow up in a deep or
    distributed stack.

    Call via ``gradient_print_wrapper.apply(x, msg)``.
    """

    @staticmethod
    @torch.compiler.disable
    def forward(x, msg=""):
        r"""
        Return the input unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to pass through.
        msg : str, optional
            Label included in the backward-pass printout.

        Returns
        -------
        torch.Tensor
            ``x``, unchanged.
        """
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        r"""
        Stash the label for use in the backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context.
        inputs : tuple
            The forward arguments ``(x, msg)``.
        output : torch.Tensor
            The forward result; unused.
        """
        _, msg = inputs
        ctx.msg = msg

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, go):
        r"""
        Print gradient statistics and pass the gradient through unchanged.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context carrying the label.
        go : torch.Tensor
            Incoming gradient.

        Returns
        -------
        tuple
            ``(go, None)`` -- the gradient unchanged, and ``None`` for the
            non-tensor ``msg`` argument.
        """
        msg = ctx.msg
        print(f"Gradient stats for {msg}: min: {go.min()}, max: {go.max()}, mean: {go.mean()}")
        return go, None


# weight gradient reduction helpers


# handler for additional gradient reductions
# helper for gradient reduction across channel parallel ranks
def init_gradient_reduction_hooks(
    model,
    device,
    reduction_buffer_count=1,
    broadcast_buffers=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=False,
    verbose=None,
):
    r"""
    Wrap a model in DDP with the gradient reductions model parallelism requires.

    Plain DDP reduces every gradient over the data-parallel group, which is
    wrong once model parallelism is in play: a parameter replicated across model
    ranks accumulates a gradient contribution on each of them, so it must also
    be reduced over the group it is shared across, while a sharded parameter
    must not be. This function inspects the ``is_shared_mp`` annotation each
    layer stamps on its parameters and installs a communication hook that
    performs the right reduction per parameter.

    Two cases avoid the hook entirely: with no model parallelism, or when *all*
    parameters are shared across all model ranks. In the latter case the correct
    result is obtained by scaling gradients by the model group size and letting
    ordinary DDP handle the rest, which is cheaper than a custom hook.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap. Parameters without an ``is_shared_mp`` annotation are
        assumed to be shared across all model ranks.
    device : torch.device
        Device the model lives on.
    reduction_buffer_count : int, optional
        Number of DDP gradient buckets, by default ``1``. Forced to ``1`` when
        custom hooks are needed, since the hook reduces per parameter.
    broadcast_buffers : bool, optional
        Broadcast buffers from rank 0 at each forward, by default ``True``.
        Disabled automatically when custom hooks are needed, as sharded buffers
        legitimately differ across ranks.
    find_unused_parameters : bool, optional
        Passed to DDP, by default ``False``.
    gradient_as_bucket_view : bool, optional
        Passed to DDP, by default ``True``.
    static_graph : bool, optional
        Passed to DDP, by default ``False``.
    verbose : bool, optional
        Log the sharing modes and hook setup. Defaults to the debug setting in
        the mpu config.

    Returns
    -------
    torch.nn.Module
        The wrapped model, or the original model unchanged if torch.distributed
        is not initialized.
    """
    # early exit if we are not in a distributed setting:
    if not dist.is_initialized():
        return model

    if verbose is None:
        verbose = config.debug

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
                if verbose:
                    print(f"Parameter {param.name} has no sharing mode specified, settting to globally shared.")
                param.is_shared_mp = ["model"]

            # add the sharing type to the dict
            num_parameters_total += 1
            if "model" in param.is_shared_mp:
                num_parameters_shared_model += 1

        # if all parameters are shared between all model ranks, then the situation is easy
        if num_parameters_shared_model == num_parameters_total:
            # we can always use DDP
            ddp_group = None

            # register some pre-multiply reduction hooks
            if verbose:
                print("Setting up gradient hooks to account for shared parameter multiplicity")
            for param in model.parameters():
                param.register_hook(lambda grad: grad * float(comm.get_size("model")))
        else:
            ddp_group = comm.get_group("data")
            broadcast_buffers = False
            need_hooks = True

    # compute bucket cap in MB:
    if need_hooks:
        # if we need hooks, we can only use a single reduction buffer:
        reduction_buffer_count = 1

    # determine size of model. Only local number of parameters is relevant:
    _, _, local_parameter_size_bytes = count_parameters(model, device)

    # compute reduction buffer size
    reduction_size_bytes = (local_parameter_size_bytes + reduction_buffer_count - 1) // reduction_buffer_count
    reduction_size_mb = (reduction_size_bytes + 1048575) // 1048576

    # we should fuse the first bucket with the others
    dist._DEFAULT_FIRST_BUCKET_BYTES = reduction_size_mb * 1048576

    # we can set up DDP and exit here
    if verbose:
        print("Setting up DDP communication hooks")
    model = DistributedDataParallel(
        model,
        # DDP takes no device for a model on the CPU
        device_ids=[device] if device.type == "cuda" else None,
        output_device=device if device.type == "cuda" else None,
        bucket_cap_mb=reduction_size_mb,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=ddp_group,
    )

    if not need_hooks:
        return model

    if verbose:
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
                all_reduce_mean(coalesced, comm.get_group(group))
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
        for group in comm.get_comm_names():
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


# deferred to avoid circular imports with makani.utils
from makani.utils import comm
from makani.models.helpers import count_parameters
from makani.mpu.helpers import all_reduce_mean
