# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Distributed-test pytest config.

Suppresses pytest's console output on every rank except rank 0, so the combined
mpirun/srun stdout shows a single coherent test summary instead of N interleaved
copies. The silencing unregisters the terminal reporter in ``pytest_configure``
with ``@pytest.hookimpl(trylast=True)`` -- see that hook for why ``trylast`` is
required and why it must happen at configure time.

It also surfaces the full H x W x M x E x B decomposition (spatial h/w, matmul m,
ensemble e and the auto-filled batch remainder b): ``pytest_report_header`` prints
it once at the top of the session and ``pytest_collection_modifyitems`` appends
``[h=H,w=W,...]`` to each distributed test's node id (m/e/b shown only when > 1).

Escape hatch: setting ``MAKANI_TEST_NO_SILENCE=1`` in the environment disables the
silencing on every rank -- useful when debugging a per-rank failure (e.g. a
distributed-bootstrap error in ``setUpClass``) that rank 0 doesn't reproduce.
"""

import os

import pytest
import torch
import torch.distributed as dist


def _get_world_rank():
    """Return the world rank from the env: ``WORLD_RANK`` if set, else ``RANK``.

    Read directly from the env, not via ``comm``: ``pytest_configure`` runs before
    ``comm.init()`` (which happens in ``setUpClass`` / ``setUpModule``), and
    ``comm.get_world_rank()`` returns 0 until then -- so trusting comm here would make
    every rank look like rank 0 and nothing would get silenced. ``WORLD_RANK`` is
    checked first as the explicit convention; ``RANK`` is the torch.distributed /
    torchrun (and mpirun) variable that is actually set in our launches.
    """
    for key in ("WORLD_RANK", "RANK"):
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    return 0


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Silence pytest's console output on every rank except rank 0.

    ``trylast=True`` is REQUIRED: conftest ``pytest_configure`` hooks otherwise run
    *before* pytest's builtin terminal plugin registers the ``terminalreporter``
    (LIFO plugin order), so ``get_plugin("terminalreporter")`` returns ``None`` and
    the unregister silently no-ops -- the duplicate-output bug. Running last ensures
    the reporter exists by the time we unregister it.

    Silencing at configure time (rather than in ``pytest_runtest_setup``) is also what
    suppresses the session header, the collected-items line and the summary on non-zero
    ranks: a per-test hook fires too late, after those have already been printed.
    """
    # Honor an env-var escape hatch: forces full output on every rank.
    if os.environ.get("MAKANI_TEST_NO_SILENCE"):
        return

    # rank comes from the env (comm isn't initialized at configure time)
    if _get_world_rank() == 0:
        return

    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)


def _world_size_from_env():
    """World size from the launcher env, or ``None`` if not discoverable.

    Needed only to derive the batch (data-parallel remainder) dim, which is not
    passed as a GRID_* variable. Covers torchrun (``WORLD_SIZE``), OpenMPI
    (``OMPI_COMM_WORLD_SIZE``), MPICH/PMI (``PMI_SIZE``) and Slurm (``SLURM_NTASKS``).
    """
    for key in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    return None


def _grid():
    """The full distributed decomposition ``(h, w, m, e, b)`` from the launcher env.

    ``h, w, m, e`` come straight from the ``GRID_*`` variables. The batch dim ``b``
    is the data-parallel remainder ``world_size / (h*w*m*e)`` (matching
    ``comm.init``'s auto-fill of the ``-1`` batch size); it defaults to 1 when the
    world size cannot be read from the env or does not divide evenly.
    """
    h = int(os.environ.get("GRID_H", 1))
    w = int(os.environ.get("GRID_W", 1))
    m = int(os.environ.get("GRID_M", 1))
    e = int(os.environ.get("GRID_E", 1))
    world = _world_size_from_env()
    model_and_ensemble = h * w * m * e
    if world and model_and_ensemble and (world % model_and_ensemble == 0):
        b = world // model_and_ensemble
    else:
        b = 1
    return h, w, m, e, b


def _is_distributed(grid):
    """True once any dimension of the decomposition is > 1."""
    return any(d > 1 for d in grid)


def _grid_suffix(grid):
    """``[h=..,w=..]`` with optional ``m``/``e``/``b`` components when those are active.

    h and w are always shown (they are the canonical spatial dims); matmul, ensemble
    and batch are only appended when > 1 to keep node ids compact for the common case.
    """
    h, w, m, e, b = grid
    parts = [f"h={h}", f"w={w}"]
    if m > 1:
        parts.append(f"m={m}")
    if e > 1:
        parts.append(f"e={e}")
    if b > 1:
        parts.append(f"b={b}")
    return "[" + ",".join(parts) + "]"


def pytest_report_header(config):
    """Surface the H x W x M x E x B decomposition at the top of the session.

    Non-zero ranks have already had their terminal reporter unregistered in
    ``pytest_configure``, so this prints on rank 0 only. Returns ``None`` for a plain
    non-distributed run.
    """
    grid = _grid()
    if not _is_distributed(grid):
        return None
    h, w, m, e, b = grid
    return f"distributed grid: H x W x M x E x B = {h} x {w} x {m} x {e} x {b}"


def pytest_collection_modifyitems(config, items):
    """Append the grid to each distributed test's node id (e.g. ``...[h=2,w=4,e=2,b=2]``)
    so the decomposition is visible alongside every test name in ``-v`` output. The
    ``m``/``e``/``b`` components are only shown when that dimension is enabled (> 1).
    """
    grid = _grid()
    if not _is_distributed(grid):
        return
    suffix = _grid_suffix(grid)
    for item in items:
        if item.fspath.basename.startswith(("test_distributed_", "tests_distributed_")):
            item._nodeid = item.nodeid + suffix


@pytest.fixture(autouse=True)
def drained_device_and_matched_ranks():
    """Start every test with matched ranks and a drained device, and end it with a drained device.

    Modules own receive buffers that NCCL writes asynchronously, e.g. the halo of a
    distributed DISCO convolution. When a test ends while such a write is still in
    flight, the caching allocator hands the memory to the next test, and NCCL writes
    into its tensors: a test then fails or passes depending on which tests ran before
    it. torch-harmonics guards its distributed tests the same way.

    The barrier relies on every rank running the same tests, which reduce_success
    already requires. autouse fixtures apply to unittest.TestCase methods as well, and
    run after setUpClass, i.e. after comm.init().
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
