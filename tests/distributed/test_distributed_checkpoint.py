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

import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.distributed as dist

from makani.utils import comm
from makani.utils.driver import Driver
from makani.utils.checkpoint_helpers import gather_model_state_dict

from .distributed_helpers import _init_grid, reduce_success, sync_and_barrier, teardown_comms


class _ShardedTestModel(nn.Module):
    """
    Tiny model with one parameter sharded across (h, w) groups and one replicated
    parameter, mirroring the annotation pattern used by real makani modules
    (e.g. spectral weights). Each rank constructs its own local slice of the
    sharded parameter from a globally-deterministic source so we can verify
    round-trips per rank.
    """

    def __init__(self, local_weight: torch.Tensor, num_features: int):
        super().__init__()

        # Sharded along dim 2 ("h") and dim 3 ("w") — this is the spectral-weight
        # sharding pattern in fourcastnet3 / fourcastnatt3.
        self.weight = nn.Parameter(local_weight.clone())
        self.weight.is_shared_mp = ["matmul"]
        self.weight.sharded_dims_mp = [None, None, "h", "w"]

        # Replicated 1-D bias.
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.bias.is_shared_mp = ["spatial"]
        self.bias.sharded_dims_mp = [None]

    def forward(self, x):
        return x  # not used; tests touch state_dict only


class _DivergentPlanModel(_ShardedTestModel):
    """
    Model whose gather *plan* differs across the model group: only the model-root
    rank registers an extra sharded parameter, so its ordered sequence of
    gather_uneven calls has one more entry than every other rank's.

    This is the artificial analog of the real failure mode behind the checkpoint
    hang -- ``model.named_parameters()`` (and hence the per-param all_gather
    sequence) not matching across the model-parallel ranks. The crc precondition
    in ``gather_model_state_dict`` must catch it as a clean ``RuntimeError`` rather
    than letting it deadlock the ``all_gather`` inside ``gather_uneven`` (the
    600s NCCL timeout we are guarding against).
    """

    def __init__(self, local_weight: torch.Tensor, num_features: int):
        super().__init__(local_weight, num_features)
        if comm.get_rank("model") == 0:
            # extra parameter sharded over "w" -> adds one entry to this rank's
            # gather plan only. (Never actually gathered: the crc check fires first.)
            self.extra = nn.Parameter(torch.zeros(num_features))
            self.extra.is_shared_mp = ["spatial"]
            self.extra.sharded_dims_mp = ["w"]


class TestDistributedCheckpoint(unittest.TestCase):
    """
    Round-trip tests for the legacy and flexible checkpoint formats under
    spatial model parallelism (h × w groups), with optional ensemble (E) and
    batch (B) data parallelism layered on top.

    Run with e.g.::

        # pure spatial model parallel (H x W)
        GRID_H=2 GRID_W=2 mpirun -n 4 pytest tests/distributed/test_distributed_checkpoint.py

        # spatial + ensemble + batch: model=H*W*M=4, data=E*B; with -n 16 and
        # GRID_E=2 the remaining factor (16/4/2 = 2) auto-fills the batch group,
        # so this exercises H x W x E x B = 2 x 2 x 2 x 2.
        GRID_H=2 GRID_W=2 GRID_E=2 mpirun -n 16 pytest tests/distributed/test_distributed_checkpoint.py

    The flexible save gathers sharded params over the *model* group only and
    replicates the result across the data-parallel (ensemble/batch) groups, so
    running with E,B > 1 is what exercises the desync the crc precondition guards
    against. The tests pass trivially when world_size=1 (no sharding actually
    happens), so meaningful coverage requires multi-process launch.
    """

    @classmethod
    def setUpClass(cls):
        # Standard distributed-test bootstrap. Reads GRID_H/GRID_W/GRID_E from
        # env, calls comm.init, sets up cls.device and cls.{w,h,e}rank etc.
        _init_grid(cls)

        # Ensure a power-of-2 multiple of the grid for clean splitting. Using
        # 8 per direction × grid dim gives at least 8 elements per rank.
        cls.global_h = 8 * cls.grid_size_h
        cls.global_w = 8 * cls.grid_size_w
        cls.num_features = 4

        # Shared scratch directory: rank 0 picks a path under /tmp, broadcasts
        # to everyone so all ranks read/write the same checkpoint files.
        if cls.world_rank == 0:
            tmpdir = tempfile.mkdtemp(prefix="makani_ckpt_test_")
        else:
            tmpdir = None
        # broadcast a list of 1 string from rank 0 to everyone
        bcast_list = [tmpdir]
        if dist.is_initialized():
            dist.broadcast_object_list(bcast_list, src=0)
        cls.tmpdir = bcast_list[0]

    @classmethod
    def tearDownClass(cls):
        # Only rank 0 cleans up; sync + barrier first to ensure no in-flight reads.
        sync_and_barrier()
        if cls.world_rank == 0 and cls.tmpdir is not None and os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
        # barrier again (every rank waits for rank 0's cleanup above) then
        # explicitly destroy the process group -- see teardown_comms's docstring
        teardown_comms()

    def _build_local_slice(self, seed: int = 12345) -> torch.Tensor:
        """
        Build the per-rank local slice of a globally-deterministic weight tensor.

        All ranks generate the same global tensor (same seed), then each rank
        slices out its own (h, w) chunk. This guarantees every rank knows what
        its slice should be without any communication, which is what we compare
        against after the restore.
        """
        torch.manual_seed(seed)
        global_weight = torch.randn(1, self.num_features, self.global_h, self.global_w, device=self.device)
        # split along h
        h_chunks = list(global_weight.chunk(self.grid_size_h, dim=2))
        h_chunk = h_chunks[self.hrank]
        # split along w
        w_chunks = list(h_chunk.chunk(self.grid_size_w, dim=3))
        return w_chunks[self.wrank].contiguous()

    def _build_model(self) -> nn.Module:
        local_weight = self._build_local_slice()
        return _ShardedTestModel(local_weight, num_features=self.num_features).to(self.device)

    def _snapshot_weights(self, model: nn.Module) -> dict:
        """Return a per-name dict of cloned per-rank tensors for later comparison."""
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    def _zero_weights(self, model: nn.Module) -> None:
        """Mutate live weights to zero so a no-op restore would be detected."""
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()

    def _verify_match(self, model: nn.Module, snapshot: dict, label: str, verbose: bool = False) -> None:
        """Per-rank shard equality check after restore."""
        for name, param in model.named_parameters():
            ref = snapshot[name]
            self.assertTrue(
                reduce_success(torch.equal(param.detach().cpu(), ref.detach().cpu()), self.device),
                msg=f"[{label}] rank {self.world_rank} param '{name}' mismatch after restore",
            )

    # ----------------------------------------------------------------------
    # Legacy format: each rank writes its OWN file (with mp_rank in the name)
    # containing only its local slice. On restore each rank reads its own file.
    # No gather/scatter — the per-rank shape is preserved on disk verbatim.
    # ----------------------------------------------------------------------
    def test_legacy_save_restore_roundtrip(self, verbose=False):
        ckpt_path = os.path.join(self.tmpdir, "legacy_ckpt_mp{mp_rank}_v0.tar")

        model = self._build_model()
        snapshot = self._snapshot_weights(model)

        # 1. Save in legacy mode: each rank writes its own slice to its own file.
        Driver.save_checkpoint(
            ckpt_path,
            model,
            loss=None,
            optimizer=None,
            scheduler=None,
            counters=None,
            checkpoint_mode="legacy",
        )
        if dist.is_initialized():
            dist.barrier()

        # Sanity: the file for THIS rank exists.
        my_file = ckpt_path.format(mp_rank=comm.get_rank("model"))
        with self.subTest(desc=f"legacy file exists for rank {self.world_rank}"):
            self.assertTrue(os.path.isfile(my_file))

        # 2. Mutate live weights so a botched restore would be detectable.
        self._zero_weights(model)

        # 3. Restore. Each rank reads its own file.
        Driver._restore_checkpoint_legacy(ckpt_path, model, strict=True, validate_comms=True)

        # 4. Per-rank shard must match the pre-save snapshot bit-for-bit.
        with self.subTest(desc="weights match after legacy restore"):
            self._verify_match(model, snapshot, label="legacy", verbose=verbose)

    # ----------------------------------------------------------------------
    # Flexible format: rank 0 GATHERS all sharded params into a single global
    # state_dict and writes one file. On restore the file is read on every rank
    # and SCATTERED back to local shards. The on-disk file is independent of
    # the model-parallel topology, so a checkpoint saved with H×W=2×2 can be
    # reloaded with H×W=4×1 — but here we just exercise the same topology.
    # ----------------------------------------------------------------------
    def test_flexible_save_restore_roundtrip(self, verbose=False):
        ckpt_path = os.path.join(self.tmpdir, "flexible_ckpt_mp{mp_rank}_v0.tar")

        model = self._build_model()
        snapshot = self._snapshot_weights(model)

        # 1. Save in flexible mode: rank 0 gathers and writes; other ranks no-op write.
        Driver.save_checkpoint(
            ckpt_path,
            model,
            loss=None,
            optimizer=None,
            scheduler=None,
            counters=None,
            checkpoint_mode="flexible",
        )
        if dist.is_initialized():
            dist.barrier()

        # Sanity: only ONE file exists (mp_rank=0), regardless of world size.
        rank0_file = ckpt_path.format(mp_rank=0)
        with self.subTest(desc="flexible writes a single global file"):
            self.assertTrue(os.path.isfile(rank0_file))
        # The per-rank legacy-style files should NOT be present in flexible mode.
        if comm.get_rank("model") != 0:
            other_file = ckpt_path.format(mp_rank=comm.get_rank("model"))
            with self.subTest(desc=f"no per-rank file for non-zero rank {comm.get_rank('model')}"):
                self.assertFalse(os.path.isfile(other_file))

        # 2. Mutate.
        self._zero_weights(model)

        # 3. Restore. Each rank reads the same file; scatter_model_state_dict
        # gives each rank its local shard.
        Driver._restore_checkpoint_flexible(ckpt_path, model, strict=True)

        # 4. Per-rank shard must match the pre-save snapshot.
        with self.subTest(desc="weights match after flexible restore"):
            self._verify_match(model, snapshot, label="flexible", verbose=verbose)

    # ----------------------------------------------------------------------
    # Cross-format compatibility: a flexible-saved file can be restored on
    # a model that's running under the same topology. Loading legacy checkpoints
    # in flexible mode is NOT supported (legacy files have per-rank shards;
    # flexible expects a single global file at mp_rank=0).
    # ----------------------------------------------------------------------
    def test_flexible_restore_after_save_independent_of_run(self, verbose=False):
        """
        Save flexible, fully tear down the model, build a fresh one, then
        restore flexible. Verifies the on-disk file is self-contained.
        """
        ckpt_path = os.path.join(self.tmpdir, "flexible_repeat_mp{mp_rank}_v0.tar")

        original = self._build_model()
        snapshot = self._snapshot_weights(original)

        Driver.save_checkpoint(
            ckpt_path,
            original,
            checkpoint_mode="flexible",
        )
        if dist.is_initialized():
            dist.barrier()

        # Build a fresh model with DIFFERENT initial weights (different seed).
        torch.manual_seed(99999 + self.world_rank)
        replacement_local = torch.randn(
            1,
            self.num_features,
            self.global_h // self.grid_size_h,
            self.global_w // self.grid_size_w,
            device=self.device,
        )
        fresh = _ShardedTestModel(replacement_local, num_features=self.num_features).to(self.device)

        # Confirm the fresh model is NOT already equal to the snapshot
        with self.subTest(desc="fresh model differs from snapshot before restore"):
            with torch.no_grad():
                self.assertFalse(
                    torch.equal(
                        fresh.weight.detach().cpu(),
                        snapshot["weight"].detach().cpu(),
                    )
                )

        # Restore from disk into the fresh model.
        Driver._restore_checkpoint_flexible(ckpt_path, fresh, strict=True)

        with self.subTest(desc="fresh model matches snapshot after restore"):
            self._verify_match(fresh, snapshot, label="flexible-fresh", verbose=verbose)

    # ----------------------------------------------------------------------
    # Gather-plan consistency precondition (the checkpoint-hang guard).
    #
    # The flexible gather deadlocks if the ranks of a model group do not issue
    # the SAME ordered sequence of all_gather calls (e.g. named_parameters drifts
    # across ranks). gather_model_state_dict crc-checks the plan up front and
    # raises instead of hanging. These two tests cover both verdicts under
    # H x W x E x B: a matching plan passes, a divergent plan raises cleanly.
    # ----------------------------------------------------------------------
    def test_flexible_gather_plan_consistent(self, verbose=False):
        """Positive: identical plan across the model group -> gather completes."""
        if comm.get_size("model") == 1:
            self.skipTest("requires model parallelism (set GRID_H/GRID_W/GRID_M > 1)")

        model = self._build_model()

        # Must not raise and must not hang: every model-group rank runs the same
        # crc all_gather and agrees. Data-parallel (E/B) replicas each run the
        # gather over their own model group independently.
        state_dict = gather_model_state_dict(model)

        n_params = len(list(model.named_parameters()))
        with self.subTest(desc="gather returns one entry per parameter"):
            self.assertTrue(
                reduce_success(len(state_dict) == n_params, self.device),
                msg=f"rank {self.world_rank}: gathered {len(state_dict)} entries, expected {n_params}",
            )

    def test_flexible_gather_plan_mismatch_raises(self, verbose=False):
        """Negative: a divergent plan must raise RuntimeError, not deadlock."""
        if comm.get_size("model") == 1:
            self.skipTest("requires model parallelism (set GRID_H/GRID_W/GRID_M > 1)")

        local_weight = self._build_local_slice()
        model = _DivergentPlanModel(local_weight, num_features=self.num_features).to(self.device)

        # Every model group has exactly one diverging rank (model-rank 0), so the
        # crc check raises on ALL ranks symmetrically -- no rank is left waiting in
        # a collective. assertRaises therefore fires consistently everywhere.
        with self.assertRaises(RuntimeError):
            gather_model_state_dict(model)

    # ----------------------------------------------------------------------
    # Optimizer state, flexible format, with MULTIPLE parameter groups.
    #
    # torch.optim packs optimizer-state indices group-by-group (in
    # optimizer.param_groups order), which does NOT match model.parameters()
    # order once parameters are split across >1 group. gather/scatter of the
    # optimizer state must therefore pair each state entry with the parameter
    # that actually owns it. We build the groups in REVERSE order (no-decay/bias
    # first, then decay/weight) so the packing order differs from model order --
    # the configuration that exposed the original index-vs-param mismatch. With
    # the old enumerate(model.parameters()) indexing this even crashes, because
    # the (h,w)-sharded weight's gather is applied to the 1-D bias state.
    # ----------------------------------------------------------------------
    def _build_reordered_param_groups(self, model):
        decay, no_decay = [], []
        for p in model.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        # no-decay group FIRST -> optimizer packing order != model.parameters() order
        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": 0.1},
        ]

    def _make_stepped_optimizer(self, model):
        optimizer = torch.optim.AdamW(self._build_reordered_param_groups(model), lr=1e-3)
        # populate exp_avg / exp_avg_sq with one deterministic step
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return optimizer

    def _snapshot_opt_state(self, optimizer):
        state = optimizer.state_dict()["state"]
        return {i: {k: v.detach().clone() for k, v in s.items() if torch.is_tensor(v)} for i, s in state.items()}

    def test_flexible_optimizer_state_roundtrip_multigroup(self, verbose=False):
        ckpt_path = os.path.join(self.tmpdir, "flexible_optstate_mp{mp_rank}_v0.tar")

        model = self._build_model()
        optimizer = self._make_stepped_optimizer(model)
        opt_snapshot = self._snapshot_opt_state(optimizer)

        # 1. Save flexible: rank 0 gathers the (sharded) optimizer moments into a global dict.
        Driver.save_checkpoint(
            ckpt_path,
            model,
            optimizer=optimizer,
            checkpoint_mode="flexible",
        )
        if dist.is_initialized():
            dist.barrier()

        # 2. Corrupt the live optimizer moments so a no-op restore would be detected.
        with torch.no_grad():
            for s in optimizer.state.values():
                if "exp_avg" in s:
                    s["exp_avg"].zero_()
                    s["exp_avg_sq"].zero_()

        # 3. Restore flexible: each rank reads the global file and scatters back to its shard.
        Driver._restore_checkpoint_flexible(ckpt_path, model, optimizer=optimizer)

        # 4. Per-rank optimizer moments must match the pre-save snapshot bit-for-bit.
        restored = optimizer.state_dict()["state"]
        for i, ref in opt_snapshot.items():
            for k in ("exp_avg", "exp_avg_sq"):
                with self.subTest(desc=f"opt state[{i}][{k}] rank {self.world_rank}"):
                    self.assertTrue(
                        reduce_success(torch.equal(restored[i][k].cpu(), ref[k].cpu()), self.device),
                        msg=f"rank {self.world_rank}: optimizer state[{i}][{k}] mismatch after multigroup restore",
                    )


if __name__ == "__main__":
    unittest.main()
