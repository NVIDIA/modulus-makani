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

import contextlib
import math

import torch.distributed as dist

# we are using the distributed manager from physicsnemo
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.distributed.config import ProcessGroupNode, ProcessGroupConfig

# we need this
_DM = None
_COMM_ROOTS = {}

# nesting depth of serial(); while positive, the group queries report a single process
_SERIAL_DEPTH = 0


def _group_queries_active() -> bool:
    return (_DM is not None) and (_SERIAL_DEPTH == 0)


@contextlib.contextmanager
def serial():
    """
    Make the group queries report a single process while the context is active.

    Inside the context, ``get_size`` returns 1, ``get_rank`` and ``get_root``
    return 0, ``get_group`` returns ``None``, ``is_distributed`` returns
    ``False`` and ``get_comm_names`` returns no names -- what they return when
    the comms are not initialized. The world queries (``get_world_size``,
    ``get_world_rank``, ``get_local_rank``) are not affected, and the process
    groups themselves stay alive.

    This builds a serial model, loss or metric next to distributed ones in the
    same process, e.g. as the reference in a distributed test. Models choose
    their parallel layout when they are constructed, so only the construction
    has to happen inside the context; the forward and backward passes of such a
    model run serially outside of it as well. Code that queries the comms when it
    is called, such as checkpoint saving, has to be called inside the context.

    Contexts can be nested. ``init`` refuses to run inside one.
    """
    global _SERIAL_DEPTH
    _SERIAL_DEPTH += 1
    try:
        yield
    finally:
        _SERIAL_DEPTH -= 1


def is_serial() -> bool:
    """Whether a :func:`serial` context is active."""
    return _SERIAL_DEPTH > 0


def get_size(name: str) -> int:
    global _DM
    if _group_queries_active() and (_DM.world_size > 1):
        return _DM.group_size(name)
    else:
        return 1


def get_rank(name: str) -> int:
    global _DM
    if _group_queries_active() and (_DM.world_size > 1):
        return _DM.group_rank(name)
    else:
        return 0


def get_group(name: str):
    global _DM
    if _group_queries_active():
        return _DM.group(name)
    else:
        return None


def get_root(name: str) -> int:
    global _DM
    global _COMM_ROOTS
    if _group_queries_active() and (name in _COMM_ROOTS) and (_DM.world_size > 1):
        return _COMM_ROOTS[name]
    else:
        return 0


# specialized routines for world comms
def get_world_size():
    global _DM
    if _DM is not None:
        return _DM.world_size
    else:
        return 1


def get_world_rank():
    global _DM
    if _DM is not None:
        return _DM.rank
    else:
        return 0


def get_local_rank():
    global _DM
    if _DM is not None:
        return _DM.local_rank
    else:
        return 0


def get_comm_names():
    global _DM
    if _group_queries_active():
        return [name for name in _DM.group_names if (not name.startswith("__orthogonal_to"))]
    else:
        return []


def get_model_comm_names():
    return [x for x in get_comm_names() if x not in ["world", "data", "ensemble", "batch"]]


def is_distributed(name: str):
    global _DM
    if _group_queries_active():
        return name in _DM.group_names
    else:
        return False


def cleanup():
    """Tear down every process group this module registered, then the world group.

    ``DistributedManager.cleanup()`` only calls ``dist.destroy_process_group()``
    with no argument (the default/world group) -- physicsnemo's own comment on
    that call claims "destroying group.WORLD is enough for all process groups
    to get destroyed", but nothing there actually verifies it, and this module
    (via ``create_groups_from_config``) registers a whole tree of subgroups
    under ``_DM._groups`` -- ``h``, ``w``, ``spatial``, ``model``, ``matmul``,
    ``data``, ``ensemble``, ``batch``, plus one ``__orthogonal_to_<name>``
    group per node (``DistributedManager.create_group_from_node`` creates one
    for every node, and ``DistributedManager`` tracks it the same way as any
    other named group, so ``_DM.group_names`` already includes them -- nothing
    here has to enumerate them separately).

    Other code holds direct references to some of these -- e.g.
    ``torch_harmonics.distributed``'s own cached polar/azimuth group handles
    (the exact ``h``/``w`` group objects, if a caller passed them to
    ``torch_harmonics.distributed.init``) -- and may have *already* destroyed
    a given group through its own explicit API (``torch_harmonics.distributed.finalize()``,
    for the polar/azimuth pair) by the time this runs. This module has no
    general way to know which names that covers, so each destroy call is
    wrapped: a group that is already gone raises ``ValueError`` from
    ``dist.destroy_process_group``, treated here as success (someone else
    already did this one) rather than a real failure -- silently relying on
    Python reference counting to eventually destroy the rest is not
    deterministic either way, so every group still gets an explicit destroy
    attempt, guaranteeing its underlying NCCL communicator is gone before the
    next ``init()``.
    """
    global _DM
    if _DM is not None:
        for name in list(_DM.group_names):
            group = _DM.group(name)
            if group is not None:
                try:
                    dist.destroy_process_group(group)
                except ValueError:
                    pass
        _DM.cleanup()
        _DM = None
    return


# initialization routine
def init(
    model_parallel_sizes=[1, 1, 1],
    model_parallel_names=["h", "w", "matmul"],
    data_parallel_sizes=[1, -1],
    data_parallel_names=["ensemble", "batch"],
    verbose=False,
):

    # the groups would be registered against the serial view of the queries
    if is_serial():
        raise RuntimeError("comm.init() cannot be called inside comm.serial()")

    # call basic init first
    DistributedManager.initialize()

    # extract manager object
    global _DM
    _DM = DistributedManager()

    # create process group config:
    world = ProcessGroupNode("world", size=_DM.world_size)
    pconfig = ProcessGroupConfig(world)

    # add nodes:
    # model
    pconfig.add_node(ProcessGroupNode("model"), parent="world")
    # spatial and matmul. matmul is a single (1D) feature-parallel leaf group;
    # the legacy 2D fin/fout decomposition has been retired in favor of a
    # Megatron-style column/row fork-join over the single "matmul" group.
    pconfig.add_node(ProcessGroupNode("spatial"), parent="model")
    pconfig.add_node(ProcessGroupNode("matmul"), parent="model")
    # subgroups for spatial
    pconfig.add_node(ProcessGroupNode("w"), parent="spatial")
    pconfig.add_node(ProcessGroupNode("h"), parent="spatial")
    # add data node last
    pconfig.add_node(ProcessGroupNode("data"), parent="world")
    # other data parallel dims
    for dgname in data_parallel_names:
        pconfig.add_node(ProcessGroupNode(dgname), parent="data")

    # set up leaf sizes
    # model
    model_leaf_config = {}
    for k, v in zip(model_parallel_names, model_parallel_sizes):
        model_leaf_config[k] = v
    # data
    data_group_size = _DM.world_size // math.prod(model_leaf_config.values())
    data_leaf_config = {}
    for k, v in zip(data_parallel_names, data_parallel_sizes):
        data_leaf_config[k] = v
    # determine some automatic shapes: only one is supported, the others will
    # default to 1
    ndata = 1
    for k in data_leaf_config:
        v = data_leaf_config[k]
        if v > 0:
            ndata *= v
    for k in data_leaf_config:
        v = data_leaf_config[k]
        if v <= 0:
            data_leaf_config[k] = data_group_size // ndata
            # the others will automatically be sized 1
            ndata = data_group_size
    # fuse leaf configs
    leaf_config = model_leaf_config
    for k in data_leaf_config:
        leaf_config[k] = data_leaf_config[k]
    # update sizes
    pconfig.set_leaf_group_sizes(leaf_config, update_parent_sizes=True)

    # create remaining process groups
    _DM.create_groups_from_config(pconfig, verbose=(verbose and (_DM.rank == 0)))

    # get comm roots:
    global _COMM_ROOTS
    for gname in get_comm_names():
        rank = _DM.rank
        for grp in _DM._group_ranks[gname]:
            if rank in grp:
                _COMM_ROOTS[gname] = min(grp)

    if verbose:
        import torch

        for rank in range(_DM.world_size):
            if rank == _DM.rank:
                print(f"{rank}: groups:")
                for gname in get_comm_names():
                    print(f"\t{gname}: {_DM._group_ranks[gname]}, root={_COMM_ROOTS[gname]}")
            torch.distributed.barrier(device_ids=[_DM.local_rank] if torch.cuda.is_available() else None)

    return get_size("model")
