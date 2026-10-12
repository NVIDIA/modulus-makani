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

import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from makani.utils import comm
from makani.utils.driver import Driver
from makani.utils.dataloader import get_dataloader

from ..testutils import init_hdf5_dataset, compare_arrays
from ..test_dataloader import init_dataset_params, get_sample
from .distributed_helpers import _init_grid, _gather_helper, reduce_success, sync_and_barrier, teardown_comms


_have_dali = importlib.util.find_spec("nvidia.dali") is not None


class _TinyModel(nn.Module):
    """Minimal replicated model, only needed because checkpoints always carry model state."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.weight.is_shared_mp = ["spatial"]
        self.weight.sharded_dims_mp = [None]

    def forward(self, x):
        return x  # not used; tests touch state_dict only


@unittest.skipUnless(_have_dali, "nvidia.dali is not installed")
class TestDistributedDataloaderCheckpoint(unittest.TestCase):
    """
    Round-trip tests for the state of the DALI training data pipeline under data
    parallelism, in isolation from the rest of the training state.

    The data pipeline is sharded over the ``batch`` comm, so every data-parallel
    rank walks a different subset of the samples and therefore holds a different
    state, while the checkpoint file is written by rank 0 of the ``data`` comm
    only. That asymmetry is what these tests cover: the state is gathered over
    the ``data`` comm on save and indexed back per rank on restore, so a rank
    must resume its OWN sample sequence and not some other rank's.

    Run with e.g.::

        # pure data parallel (B = 4)
        mpirun -n 4 pytest tests/distributed/tests_distributed_dataloader_checkpoint.py

        # ensemble x batch data parallelism on top of a 2x2 spatial grid
        GRID_H=2 GRID_W=2 GRID_E=2 mpirun -n 16 pytest tests/distributed/tests_distributed_dataloader_checkpoint.py

    With world_size=1 the tests still exercise the gather/scatter path, but the
    per-rank distinctness they are really after only shows up with B > 1.
    """

    @classmethod
    def setUpClass(cls):
        # standard distributed-test bootstrap: reads GRID_H/GRID_W/GRID_E from env,
        # calls comm.init, sets cls.device and cls.{w,h,e}rank
        _init_grid(cls)

        # shared scratch directory: rank 0 creates the dataset and broadcasts the
        # path, so all ranks read the same files
        if cls.world_rank == 0:
            tmpdir = tempfile.mkdtemp(prefix="makani_dl_ckpt_test_")
            init_hdf5_dataset(tmpdir)
        else:
            tmpdir = None
        bcast_list = [tmpdir]
        if dist.is_initialized():
            dist.broadcast_object_list(bcast_list, src=0)
        cls.tmpdir = bcast_list[0]

        cls.train_path = os.path.join(cls.tmpdir, "train")
        cls.valid_path = os.path.join(cls.tmpdir, "test")
        cls.stats_path = os.path.join(cls.tmpdir, "stats")

        # every rank has to see the dataset before the first test opens it
        sync_and_barrier()

        cls.dali_device = "gpu" if torch.cuda.is_available() else "cpu"

    @classmethod
    def tearDownClass(cls):
        # only rank 0 cleans up; sync + barrier first to ensure no in-flight reads
        sync_and_barrier()
        if cls.world_rank == 0 and cls.tmpdir is not None and os.path.isdir(cls.tmpdir):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)
        # barrier again (every rank waits for rank 0's cleanup above) then
        # explicitly destroy the process group -- see teardown_comms's docstring
        teardown_comms()

    def _make_loader(self):
        params = init_dataset_params(
            self.train_path,
            self.valid_path,
            self.stats_path,
            batch_size=1,
            n_history=0,
            n_future=0,
            normalization="zscore",
            num_data_workers=1,
        )
        params.multifiles = False
        # samples per shard stay constant as the grid grows, so the epoch is long
        # enough for the consume/compare steps below on any decomposition
        params.n_train_samples = 16 * comm.get_size("batch")

        loader, _, _ = get_dataloader(
            params,
            params.train_data_path,
            mode="train",
            device=self.device,
            dali_device=self.dali_device,
        )
        # shut the pipeline down when the test ends instead of leaving it to the garbage collector
        self.addCleanup(loader.close)

        return loader

    @staticmethod
    def _collect(iterator, num_steps):
        """Fingerprint the next ``num_steps`` batches of the loader."""
        fps = []
        for _ in range(num_steps):
            inp = next(iterator)[0]
            for b in range(inp.shape[0]):
                fps.append(inp[b].cpu().numpy().tobytes())
        return fps

    def test_gather_scatter_state_roundtrip(self, verbose=False):
        """
        Gather the state on all ranks, hand it back through the scatter path and
        check that every rank resumes its own interrupted sample sequence.
        """
        num_consumed = 2
        num_compared = 2

        def fp_summary(fps):
            # short, printable stand-in for the raw fingerprint bytes: first 8 hex
            # digits of each entry's hash, so a mismatch is legible in test output
            import hashlib

            return [hashlib.sha1(fp).hexdigest()[:8] for fp in fps]

        loader = self._make_loader()
        iterator = iter(loader)
        self._collect(iterator, num_consumed)

        # collective over the data comm: one entry per data-parallel rank
        state_dicts = Driver.gather_dataloader_state(loader)

        with self.subTest(desc="one state per data-parallel rank"):
            ok = (state_dicts is not None) and (len(state_dicts) == comm.get_size("data"))
            if verbose and not ok:
                print(
                    f"[rank {self.world_rank}] data comm: rank={comm.get_rank('data')} size={comm.get_size('data')} "
                    f"state_dicts={state_dicts}",
                    flush=True,
                )
            self.assertIsNotNone(state_dicts)
            self.assertEqual(len(state_dicts), comm.get_size("data"))

        # the continuation of the interrupted epoch is this rank's ground truth
        reference = self._collect(iterator, num_compared)

        # restore into a fresh pipeline, which picks this rank's entry out of the list
        restored_loader = self._make_loader()
        Driver._restore_dataloader_state({"dataloader_state": state_dicts}, restored_loader, strict=True)
        resumed = self._collect(iter(restored_loader), num_compared)

        # note that this comparison alone does not discriminate between the ranks:
        # the counters in the state are identical on all of them (they consume in
        # lockstep), so a wrong pick from the list would still yield this rank's
        # samples. test_state_from_a_different_shard_is_rejected covers that.
        with self.subTest(desc="restored loader resumes this rank's sequence"):
            ok = resumed == reference
            if verbose and not ok:
                print(
                    f"[rank {self.world_rank}] data comm: rank={comm.get_rank('data')} size={comm.get_size('data')}\n"
                    f"[rank {self.world_rank}] gathered state_dicts: {state_dicts}\n"
                    f"[rank {self.world_rank}] my scattered entry: {state_dicts[comm.get_rank('data')]}\n"
                    f"[rank {self.world_rank}] reference fingerprints: {fp_summary(reference)}\n"
                    f"[rank {self.world_rank}] resumed    fingerprints: {fp_summary(resumed)}",
                    flush=True,
                )
            self.assertTrue(
                reduce_success(ok, self.device),
                msg=f"rank {self.world_rank} did not resume its own sample sequence",
            )

        # without a restore the pipeline starts at the beginning of the epoch, so a
        # match here would mean the comparison above is not actually sensitive to
        # the stored position
        fresh = self._collect(iter(self._make_loader()), num_compared)

        with self.subTest(desc="loader without restore does not resume"):
            ok = fresh != reference
            if verbose and not ok:
                print(
                    f"[rank {self.world_rank}] data comm: rank={comm.get_rank('data')} size={comm.get_size('data')}\n"
                    f"[rank {self.world_rank}] reference fingerprints: {fp_summary(reference)}\n"
                    f"[rank {self.world_rank}] fresh      fingerprints: {fp_summary(fresh)}",
                    flush=True,
                )
            self.assertTrue(
                reduce_success(ok, self.device),
                msg=f"rank {self.world_rank} resumed without restoring a state",
            )

    def test_state_survives_checkpoint_file_roundtrip(self, verbose=False):
        """
        Same round-trip, but through an actual checkpoint file: the state is written
        by rank 0 of the data comm only and read back by every rank, which is how
        the trainers use it. This also covers that the serialized DALI state
        survives the restricted (``weights_only=True``) unpickler.
        """
        ckpt_path = os.path.join(self.tmpdir, "dataloader_ckpt_mp{mp_rank}_v0.tar")

        num_consumed = 3
        num_compared = 2

        loader = self._make_loader()
        iterator = iter(loader)
        self._collect(iterator, num_consumed)

        model = _TinyModel().to(self.device)

        # gather on all ranks, write on the checkpointing rank only - the same
        # split the trainers implement around their rank-0-only save block
        state_dicts = Driver.gather_dataloader_state(loader)
        if comm.get_rank("data") == 0:
            Driver.save_checkpoint(
                ckpt_path,
                model,
                counters={"iters": 0, "epoch": 0},
                dataloader_state=state_dicts,
                checkpoint_mode="legacy",
            )
        if dist.is_initialized():
            dist.barrier()

        reference = self._collect(iterator, num_compared)

        # every rank restores from the file written for its model rank
        restored_loader = self._make_loader()
        Driver._restore_checkpoint_legacy(
            ckpt_path,
            model,
            dataloader=restored_loader,
            strict=True,
            validate_comms=True,
        )
        resumed = self._collect(iter(restored_loader), num_compared)

        with self.subTest(desc="checkpointed state resumes this rank's sequence"):
            self.assertTrue(
                reduce_success(resumed == reference, self.device),
                msg=f"rank {self.world_rank} did not resume its own sample sequence from the checkpoint",
            )

    @staticmethod
    def _comparable_state(loader):
        """The loader state in a form that can be compared across ranks.

        The serialized DALI blob is a tensor, which does not compare by value inside
        a plain ``==`` on the dict, so it is turned into bytes first.
        """
        return {
            key: (value.numpy().tobytes() if torch.is_tensor(value) else value)
            for key, value in loader.state_dict().items()
        }

    def _all_gather_in_group(self, value, group_name):
        """All-gather a picklable value within one comm, or None if that comm is trivial."""
        size = comm.get_size(group_name)
        if size < 2:
            return None
        gathered = [None for _ in range(size)]
        dist.all_gather_object(gathered, value, group=comm.get_group(group_name))
        return gathered

    def test_restored_state_consistent_across_model_and_ensemble_ranks(self, verbose=False):
        """
        All ranks sharing a data shard must be restored to the SAME position.

        Sharding happens over the ``batch`` comm only, so ranks that differ solely in
        their h / w / matmul / ensemble coordinate feed on the same sample sequence,
        each reading its own spatial piece of it. If a restore put them at different
        positions, the model would silently consume mismatched pieces from different
        samples -- a failure that no single-rank test can see. Every rank within the
        h, w, matmul and ensemble comms must therefore hold a byte-identical state
        after the restore, since that state (counters, shard, seed) is exactly what
        determines which sample comes next.

        Where the IO decomposition is also identical within a comm -- matmul and
        ensemble do not enter ``io_rank``, only h and w do -- the delivered samples
        themselves are compared too, not just the state that produces them: ensemble
        ranks are meant to draw the very same sample and only differ in how they
        perturb it downstream.

        Along h and w the local pieces cannot be compared to each other, since each
        rank holds a different sub-domain by construction. They are instead
        reassembled into the global image and matched against the samples on disk:
        if the ranks had resumed at different positions, the pieces would compose an
        image that exists nowhere in the dataset.
        """
        num_consumed = 2
        num_compared = 2

        loader = self._make_loader()
        iterator = iter(loader)
        self._collect(iterator, num_consumed)

        state_dicts = Driver.gather_dataloader_state(loader)

        restored_loader = self._make_loader()
        Driver._restore_dataloader_state({"dataloader_state": state_dicts}, restored_loader, strict=True)

        restored_state = self._comparable_state(restored_loader)

        # keep the local tensors around, the h/w check below needs to reassemble them.
        # the samples are compared numerically rather than by bytes: they run through
        # the normalization in the pipeline, so demanding bitwise equality would tie
        # the test to that being reproducible down to the last ulp.
        iterator = iter(restored_loader)
        samples = [next(iterator)[0] for _ in range(num_compared)]
        arrays = [np.squeeze(sample.cpu().numpy()) for sample in samples]

        # the shard is what the ranks are supposed to disagree on; without this the
        # equality checks below could pass simply because every rank is identical
        gathered = self._all_gather_in_group(restored_state["shard_id"], "batch")
        if gathered is not None:
            with self.subTest(desc="batch ranks hold distinct shards"):
                self.assertTrue(
                    reduce_success(len(set(gathered)) == len(gathered), self.device),
                    msg="data shards are not distinct across the batch comm",
                )

        for group_name in ("h", "w", "matmul", "ensemble"):
            gathered = self._all_gather_in_group(restored_state, group_name)
            if gathered is None:
                continue
            with self.subTest(desc=f"identical restored state across {group_name} ranks"):
                self.assertTrue(
                    reduce_success(all(state == gathered[0] for state in gathered), self.device),
                    msg=f"restored dataloader state differs across the {group_name} comm",
                )

        # h and w shard the image itself, so only matmul / ensemble ranks read the
        # very same bytes and can be compared on the delivered samples directly
        for group_name in ("matmul", "ensemble"):
            gathered = self._all_gather_in_group(arrays, group_name)
            if gathered is None:
                continue
            identical = all(
                compare_arrays(f"sample {step} across {group_name}", theirs, mine)
                for other in gathered
                for step, (theirs, mine) in enumerate(zip(other, gathered[0]))
            )
            with self.subTest(desc=f"identical samples across {group_name} ranks"):
                self.assertTrue(
                    reduce_success(identical, self.device),
                    msg=f"restored loaders deliver different samples across the {group_name} comm",
                )

        # along h and w the pieces must reassemble into one of the samples on disk.
        # the test stats fixture is zero-mean / unit-std, so the zscore normalization
        # leaves the values alone and the loader output can be matched against the raw
        # file contents (same baseline as _test_dali_parallel_workers_full_epoch_coverage,
        # but numerically, so the check survives a stats fixture that is not exactly
        # neutral or a normalization that is not exactly a pass-through).
        on_disk = [get_sample(self.train_path, int(idx)) for idx in restored_loader.extsource.indices_select]
        for step, sample in enumerate(samples):
            assembled = _gather_helper(sample, dim=-2, group=self.h_group)
            assembled = _gather_helper(assembled, dim=-1, group=self.w_group)
            assembled = np.squeeze(assembled.cpu().numpy())
            found = any(
                compare_arrays(f"assembled sample {step}", assembled, candidate, atol=1e-5, rtol=1e-4)
                for candidate in on_disk
            )
            with self.subTest(desc=f"h/w pieces belong to the same image at step {step}"):
                self.assertTrue(
                    reduce_success(found, self.device),
                    msg=f"the h/w sub-domains at step {step} do not compose a sample of the dataset",
                )

    def test_state_from_a_different_shard_is_rejected(self, verbose=False):
        """
        Picking up another shard's state has to be rejected.

        This is what actually pins down the per-rank indexing on restore. The
        serialized DALI state is only a set of counters (consumed iterations and
        epoch index), and data-parallel ranks consume batches in lockstep, so those
        counters are identical on every rank -- the shard is a property of the
        pipeline, not of the blob. Handing rank A's blob to rank B would therefore
        reproduce B's own samples and go unnoticed. The ``shard_id`` stored next to
        the blob is what makes the mix-up detectable, so it gets its own test.
        """
        if comm.get_size("batch") < 2:
            self.skipTest("needs at least two data shards (B > 1)")

        loader = self._make_loader()
        state_dict = dict(loader.state_dict())

        # the same state as a neighboring shard would have written it
        state_dict["shard_id"] = (state_dict["shard_id"] + 1) % comm.get_size("batch")

        with self.subTest(desc="strict restore raises"):
            with self.assertRaises(ValueError):
                loader.load_state_dict(state_dict, strict=True)

        with self.subTest(desc="non-strict restore skips"):
            self.assertTrue(
                reduce_success(not loader.load_state_dict(state_dict, strict=False), self.device),
                msg=f"rank {self.world_rank} restored a state belonging to another shard",
            )

    def test_restore_rejects_changed_data_parallel_size(self, verbose=False):
        """
        A state list whose length does not match the number of data-parallel ranks
        cannot be assigned to ranks, so it has to be rejected rather than silently
        resuming somebody else's position.
        """
        loader = self._make_loader()
        state_dicts = Driver.gather_dataloader_state(loader)

        # pretend the run was checkpointed with one more data-parallel rank
        too_many = list(state_dicts) + [state_dicts[0]]

        with self.subTest(desc="strict restore raises"):
            with self.assertRaises(ValueError):
                Driver._restore_dataloader_state({"dataloader_state": too_many}, loader, strict=True)

        # non-strict restore skips instead, leaving the pipeline where it was
        with self.subTest(desc="non-strict restore skips"):
            Driver._restore_dataloader_state({"dataloader_state": too_many}, loader, strict=False)


if __name__ == "__main__":
    unittest.main()
