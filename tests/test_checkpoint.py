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
Non-distributed tests for checkpointing: the pure-Python helpers in
``makani.utils.checkpoint_helpers``, and saving and restoring a model through
the ``Driver`` end to end.

The distributed gather/scatter round-trip is covered separately by
``tests/distributed/test_distributed_checkpoint.py``.
"""

import os
import sys
import unittest
import tempfile
from collections import OrderedDict
from unittest import mock
import torch
import torch.nn as nn
import time
from parameterized import parameterized

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.checkpoint_helpers import (
    get_latest_checkpoint_version,
    get_model_state_dict_prefix,
    prepend_prefix_to_state_dict,
    load_checkpoint,
    supports_dataloader_state,
    gather_dataloader_state_dict,
    scatter_dataloader_state_dict,
    UNSAFE_LOAD_ENV_VAR,
)
from makani.models.common import MLP
from makani.utils.driver import Driver
from .testutils import disable_tf32, set_seed, get_default_parameters, compare_arrays


# ---------------------------------------------------------------------------
# Checkpoint helpers: makani.utils.checkpoint_helpers
# ---------------------------------------------------------------------------
#
# Non-distributed unit tests for the pure-Python helpers in
# ``makani.utils.checkpoint_helpers``:
#
#   * ``get_latest_checkpoint_version``
#   * ``get_model_state_dict_prefix``
#   * ``prepend_prefix_to_state_dict``
#   * ``load_checkpoint``
#
# The distributed gather/scatter round-trip is covered separately by
# ``tests/distributed/test_distributed_checkpoint.py``.


class UnsupportedPayload:
    """Stand-in for an arbitrary object smuggled into a checkpoint.

    Must live at module scope so that pickle can resolve it by qualified name.
    """

    def __init__(self, value="payload"):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, UnsupportedPayload) and other.value == self.value


class TestGetLatestCheckpointVersion(unittest.TestCase):
    """
    ``get_latest_checkpoint_version`` formats the path template with
    ``mp_rank=0, checkpoint_version="*"``, globs the result, and parses
    the version from the basename via ``_v(\\d+)`` at end-of-name.
    """

    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        # Standard makani checkpoint pattern: mp_rank in name, version after _v.
        self.path_template = os.path.join(self.tmpdir, "ckpt_mp{mp_rank}_v{checkpoint_version}.tar")

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    def _touch(self, name, mtime=None):
        """Create an empty file at name, optionally overriding its mtime."""
        path = os.path.join(self.tmpdir, name)
        with open(path, "w"):
            pass
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_returns_zero_when_no_files_match(self):
        # No files exist → bare-except branch returns 0.
        self.assertEqual(get_latest_checkpoint_version(self.path_template, verbose=False), 0)

    def test_extracts_version_from_single_file(self):
        self._touch("ckpt_mp0_v3.tar")
        self.assertEqual(get_latest_checkpoint_version(self.path_template, verbose=False), 3)

    def test_picks_latest_by_mtime_not_version_number(self):
        # The implementation uses os.path.getmtime to pick the "latest" — NOT the
        # numeric max. Pin this behavior: write v5 at t=100, v2 at t=200, expect 2.
        self._touch("ckpt_mp0_v5.tar", mtime=100)
        self._touch("ckpt_mp0_v2.tar", mtime=200)
        self.assertEqual(get_latest_checkpoint_version(self.path_template, verbose=False), 2)

    def test_picks_highest_version_when_mtimes_strictly_increase(self):
        # Common case: each successive checkpoint has a later mtime AND a larger
        # version. The function returns the last-written one.
        self._touch("ckpt_mp0_v0.tar", mtime=100)
        self._touch("ckpt_mp0_v1.tar", mtime=200)
        self._touch("ckpt_mp0_v7.tar", mtime=300)
        self.assertEqual(get_latest_checkpoint_version(self.path_template, verbose=False), 7)

    def test_only_inspects_mp_rank_0_files(self):
        # The template is formatted with mp_rank=0, so files at other ranks
        # are not seen. With only an mp_rank=1 file present, function returns 0.
        self._touch("ckpt_mp1_v9.tar")
        self.assertEqual(get_latest_checkpoint_version(self.path_template, verbose=False), 0)

    def test_returns_zero_on_unparseable_filename(self):
        # File matches the glob (mp_rank=0) but lacks ``_v<digits>``: the regex
        # match fails, the bare except swallows it, function returns 0.
        # We construct a glob-matching file by faking the wildcard portion.
        weird_path = os.path.join(self.tmpdir, "ckpt_mp0_vNOTANUMBER.tar")
        with open(weird_path, "w"):
            pass
        self.assertEqual(get_latest_checkpoint_version(self.path_template, verbose=False), 0)


class _OrigModWrapper:
    """Bare wrapper exposing ``_orig_mod`` — mimics torch.compile's OptimizedModule
    for prefix-walk testing without invoking the actual compiler."""

    def __init__(self, inner):
        self._orig_mod = inner


class _FakeDDP(nn.parallel.DistributedDataParallel):
    """Subclass of DDP that bypasses ``__init__`` so it doesn't require an
    initialized process group. ``isinstance`` still returns True, which is
    all ``get_model_state_dict_prefix`` checks."""

    def __init__(self, inner):
        nn.Module.__init__(self)
        self.module = inner


class TestGetModelStateDictPrefix(unittest.TestCase):
    """
    Walks the wrapper chain accumulating ``_orig_mod.`` for torch.compile and
    ``module.`` for DDP. Order of accumulation reflects the wrapper nesting
    order from outside in.
    """

    def test_plain_module_no_prefix(self):
        m = nn.Linear(2, 2)
        self.assertEqual(get_model_state_dict_prefix(m), "")

    def test_compile_wrapper_only(self):
        m = _OrigModWrapper(nn.Linear(2, 2))
        self.assertEqual(get_model_state_dict_prefix(m), "_orig_mod.")

    def test_ddp_wrapper_only(self):
        m = _FakeDDP(nn.Linear(2, 2))
        self.assertEqual(get_model_state_dict_prefix(m), "module.")

    def test_ddp_wraps_compile_outer_first(self):
        # Outer wrapper is DDP, inner is compile → prefix is "module._orig_mod."
        m = _FakeDDP(_OrigModWrapper(nn.Linear(2, 2)))
        self.assertEqual(get_model_state_dict_prefix(m), "module._orig_mod.")

    def test_compile_wraps_ddp_outer_first(self):
        # Outer wrapper is compile, inner is DDP → prefix is "_orig_mod.module."
        m = _OrigModWrapper(_FakeDDP(nn.Linear(2, 2)))
        self.assertEqual(get_model_state_dict_prefix(m), "_orig_mod.module.")

    def test_three_level_nesting(self):
        # compile(DDP(compile(model))) → "_orig_mod.module._orig_mod."
        m = _OrigModWrapper(_FakeDDP(_OrigModWrapper(nn.Linear(2, 2))))
        self.assertEqual(get_model_state_dict_prefix(m), "_orig_mod.module._orig_mod.")


class TestPrependPrefixToStateDict(unittest.TestCase):
    """
    In-place key-rename: every key gets ``prefix`` prepended. ``_metadata``
    (when present) gets the same treatment. Pinning the slightly surprising
    behavior of the metadata loop's empty-key handling.
    """

    def test_empty_dict_no_op(self):
        d = OrderedDict()
        prepend_prefix_to_state_dict(d, "module.")
        self.assertEqual(dict(d), {})

    def test_empty_prefix_keys_unchanged(self):
        # Empty prefix means newkey == key. The pop-then-reinsert sequence
        # iterates in original order and reinserts in the same order, so the
        # mapping is unchanged.
        d = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        prepend_prefix_to_state_dict(d, "")
        self.assertEqual(list(d.keys()), ["a", "b", "c"])
        self.assertEqual(list(d.values()), [1, 2, 3])

    def test_keys_get_prefix_prepended(self):
        d = OrderedDict([("layer.weight", 1), ("layer.bias", 2)])
        prepend_prefix_to_state_dict(d, "module.")
        self.assertEqual(set(d.keys()), {"module.layer.weight", "module.layer.bias"})
        self.assertEqual(d["module.layer.weight"], 1)
        self.assertEqual(d["module.layer.bias"], 2)

    def test_in_place_same_object(self):
        # The function mutates the passed-in dict; doesn't return a new one.
        d = OrderedDict([("a", 1)])
        before_id = id(d)
        result = prepend_prefix_to_state_dict(d, "x.")
        self.assertIsNone(result)  # function returns None
        self.assertEqual(id(d), before_id)  # same object
        self.assertEqual(list(d.keys()), ["x.a"])

    def test_metadata_keys_also_get_prefix(self):
        # state_dict from torch normally has a _metadata attribute (an OrderedDict);
        # the function also rewrites those keys. Construct a state-dict-like dict
        # and attach _metadata via a subclass (plain dict can't take new attrs).
        class _DictWithMetadata(OrderedDict):
            pass

        d = _DictWithMetadata([("a", 1), ("b.c", 2)])
        d._metadata = OrderedDict([("", {"version": 1}), ("b", {"version": 1})])

        prepend_prefix_to_state_dict(d, "module.")

        with self.subTest(desc="data keys"):
            self.assertEqual(set(d.keys()), {"module.a", "module.b.c"})
        with self.subTest(desc="metadata keys"):
            # The empty-string metadata key becomes the prefix verbatim ("module.");
            # the "b" key becomes "module.b".
            self.assertEqual(set(d._metadata.keys()), {"module.", "module.b"})
        with self.subTest(desc="metadata values preserved"):
            self.assertEqual(d._metadata["module."], {"version": 1})
            self.assertEqual(d._metadata["module.b"], {"version": 1})


class TestLoadCheckpoint(unittest.TestCase):
    """
    ``load_checkpoint`` wraps ``torch.load`` with the restricted (``weights_only=True``)
    unpickler so that a hostile checkpoint cannot execute code on load. These tests pin
    both halves of that contract: legitimate checkpoint content still round-trips, and
    anything outside the allowlist is refused with an actionable error.
    """

    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        self.path = os.path.join(self.tmpdir, "ckpt.tar")
        # make sure the env escape hatch is off regardless of the ambient environment
        self.env_ctx = mock.patch.dict(os.environ, {UNSAFE_LOAD_ENV_VAR: "0"})
        self.env_ctx.start()

    def tearDown(self):
        self.env_ctx.stop()
        self.tmpdir_ctx.cleanup()

    def test_loads_plain_checkpoint_contents(self):
        # the shape of what Driver.save_checkpoint writes, without any exotic objects
        store_dict = {
            "model_state": OrderedDict([("w", torch.randn(2, 3))]),
            "comm_grid": OrderedDict([("matmul", {"size": 1, "rank": 0})]),
            "iters": 42,
            "epoch": 2,
        }
        torch.save(store_dict, self.path)

        checkpoint = load_checkpoint(self.path)

        self.assertEqual(checkpoint["iters"], 42)
        self.assertEqual(checkpoint["comm_grid"]["matmul"], {"size": 1, "rank": 0})
        self.assertTrue(torch.allclose(checkpoint["model_state"]["w"], store_dict["model_state"]["w"]))

    def test_preserves_sharding_metadata_on_tensors(self):
        # makani attaches sharded_dims_mp directly to the checkpoint tensors. Tensors with
        # extra attributes serialize through torch._tensor._rebuild_from_type_v2, which the
        # restricted unpickler permits -- convert_checkpoint depends on this surviving.
        tensor = torch.randn(2, 3)
        tensor.sharded_dims_mp = ["matmul", None]
        torch.save({"model_state": OrderedDict([("w", tensor)])}, self.path)

        restored = load_checkpoint(self.path)["model_state"]["w"]

        self.assertTrue(hasattr(restored, "sharded_dims_mp"))
        self.assertEqual(restored.sharded_dims_mp, ["matmul", None])

    def test_mmap_load(self):
        torch.save({"model_state": OrderedDict([("w", torch.randn(2, 3))])}, self.path)

        # convert_checkpoint opens the shards with mmap=True; that must work under the
        # restricted unpickler as well
        checkpoint = load_checkpoint(self.path, mmap=True)

        self.assertEqual(tuple(checkpoint["model_state"]["w"].shape), (2, 3))

    def test_loads_allowlisted_params_struct(self):
        """Legacy checkpoints may carry a params struct, which the allowlist covers.

        Reconstructing an allowlisted class from its ``__dict__`` needs the restricted
        unpickler to support the BUILD opcode for user-registered globals. If this test
        fails, the ParamsBase/YParams entries in ``_register_checkpoint_safe_globals`` do
        not actually help on this torch version and such checkpoints need the env escape
        hatch instead.
        """
        from makani.utils.YParams import ParamsBase

        params = ParamsBase()
        params.update_params({"N_in_channels": 3, "nettype": "sfno"})
        torch.save({"model_state": OrderedDict(), "params": params}, self.path)

        checkpoint = load_checkpoint(self.path)

        self.assertEqual(checkpoint["params"].N_in_channels, 3)
        self.assertEqual(checkpoint["params"].nettype, "sfno")

    def test_rejects_object_outside_allowlist(self):
        torch.save({"model_state": OrderedDict(), "payload": UnsupportedPayload()}, self.path)

        with self.assertRaises(RuntimeError) as ctx:
            load_checkpoint(self.path)

        # the error has to tell the user what to do about it
        message = str(ctx.exception)
        self.assertIn(UNSAFE_LOAD_ENV_VAR, message)
        self.assertIn("allowlist", message)
        # and the underlying unpickler error, naming the offending type, must be chained
        self.assertIsNotNone(ctx.exception.__cause__)
        self.assertIn("UnsupportedPayload", str(ctx.exception.__cause__))

    def test_rejects_object_nested_in_model_state(self):
        # the interesting attack position is inside the state dict itself, not next to it
        torch.save({"model_state": OrderedDict([("w", UnsupportedPayload())])}, self.path)

        with self.assertRaises(RuntimeError):
            load_checkpoint(self.path)

    def test_env_var_escape_hatch_allows_unsafe_load(self):
        torch.save({"model_state": OrderedDict(), "payload": UnsupportedPayload("legacy")}, self.path)

        with mock.patch.dict(os.environ, {UNSAFE_LOAD_ENV_VAR: "1"}):
            # the escape hatch must be loud about what it is doing
            with self.assertWarns(RuntimeWarning):
                checkpoint = load_checkpoint(self.path)

        self.assertEqual(checkpoint["payload"], UnsupportedPayload("legacy"))

    def test_env_var_escape_hatch_off_by_default(self):
        # anything other than an explicit opt-in keeps the safe path
        torch.save({"model_state": OrderedDict(), "payload": UnsupportedPayload()}, self.path)

        for value in ["0", "false", "no", ""]:
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {UNSAFE_LOAD_ENV_VAR: value}):
                    with self.assertRaises(RuntimeError):
                        load_checkpoint(self.path)


class _FakeDataloader:
    """Minimal stand-in for the DALI dataloader's checkpointing interface."""

    def __init__(self, state, enable_checkpointing=True):
        self.state = state
        self.enable_checkpointing = enable_checkpointing

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict, strict=True):
        self.state = state_dict
        return True


class TestDataloaderStateHelpers(unittest.TestCase):
    """Gather/scatter of the dataloader state in the single-rank (non-distributed) case."""

    def test_unsupported_dataloaders_are_detected(self):
        # loaders which cannot checkpoint (multifiles, dummy) have no such interface
        self.assertFalse(supports_dataloader_state(None))
        self.assertFalse(supports_dataloader_state(object()))
        # DALI loaders with checkpointing turned off are treated the same way
        self.assertFalse(supports_dataloader_state(_FakeDataloader({}, enable_checkpointing=False)))
        self.assertTrue(supports_dataloader_state(_FakeDataloader({})))

    def test_gather_returns_none_for_unsupported_dataloader(self):
        self.assertIsNone(gather_dataloader_state_dict(None))
        self.assertIsNone(gather_dataloader_state_dict(_FakeDataloader({}, enable_checkpointing=False)))

    def test_gather_scatter_roundtrip(self):
        state = {"format_version": 1, "pipeline_checkpoint": torch.arange(8, dtype=torch.uint8)}

        state_dicts = gather_dataloader_state_dict(_FakeDataloader(state))

        # one entry per data-parallel rank, which is a single one here
        self.assertEqual(len(state_dicts), 1)
        self.assertIs(scatter_dataloader_state_dict(state_dicts), state)

    def test_scatter_rejects_mismatched_rank_count(self):
        # resuming into a different data-parallel decomposition cannot be honored
        with self.assertRaises(ValueError):
            scatter_dataloader_state_dict([{}, {}])


# ---------------------------------------------------------------------------
# Saving and restoring through the Driver
# ---------------------------------------------------------------------------


class TestSaveRestore(unittest.TestCase):

    def setUp(self):

        disable_tf32()
        set_seed(333)

        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 36
        self.params.img_shape_y = 72
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

    def test_get_latest_checkpoint_version(self):

        def create_empty(filename):
            with open(filename, "w") as fp:
                pass

        with tempfile.TemporaryDirectory() as tempdir:
            create_empty(os.path.join(tempdir, "checkpoint_mp1_v2.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp0_v2.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp1_v0.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp0_v0.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp1_v1.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp0_v1.tar"))

            version = get_latest_checkpoint_version(
                os.path.join(tempdir, "checkpoint_mp0_v{checkpoint_version}.tar"), verbose=False
            )

        self.assertTrue(version == 1)

    def test_get_latest_checkpoint_version_default(self):
        version = get_latest_checkpoint_version("checkpoint_mp0.tar", verbose=False)
        self.assertTrue(version == 0)

    @parameterized.expand(["legacy", "flexible"])
    def test_save_restore(self, checkpoint_mode, verbose=False):
        """
        Tests initialization of all the models and the forward and backward pass
        """

        model = MLP(
            self.params.N_in_channels,
            hidden_features=2 * self.params.N_in_channels,
            out_features=self.params.N_out_channels,
            act_layer=nn.GELU,
            output_bias=True,
            input_format="nchw",
            drop_rate=0.0,
        )

        inp_shape = (
            self.params.batch_size,
            self.params.N_in_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )

        # prepare some dummy data
        inp = torch.randn(*inp_shape)
        inp.requires_grad = True

        # do forward pass:
        out_before = model(inp).detach().cpu().numpy()

        with tempfile.TemporaryDirectory() as tempdir:
            # checkpoint path
            checkpoint_path = os.path.join(tempdir, "ckpt.tar")

            # store checkpoint
            Driver.save_checkpoint(checkpoint_path, model=model, checkpoint_mode=checkpoint_mode)

            # scramble model
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()

            # reload checkpoint
            Driver.restore_from_checkpoint(
                checkpoint_path,
                model=model,
                loss=None,
                optimizer=None,
                scheduler=None,
                counters=None,
                checkpoint_mode=checkpoint_mode,
            )

        # do forward pass
        out_after = model(inp).detach().cpu().numpy()

        # compare
        self.assertTrue(compare_arrays("output", out_before, out_after, rtol=1e-6, atol=1e-6, verbose=verbose))

    @parameterized.expand(["legacy", "flexible"])
    def test_saved_checkpoint_loads_with_safe_unpickler(self, checkpoint_mode):
        """
        Everything ``Driver.save_checkpoint`` writes must be readable by the restricted
        (``weights_only=True``) unpickler used in ``load_checkpoint``, otherwise checkpoints
        written by makani could not be read back by makani.

        This exercises the full store dict -- model state carrying the ``sharded_dims_mp``
        sharding metadata, comm grid, loss/optimizer/scheduler state and the counters.
        """

        model = MLP(
            self.params.N_in_channels,
            hidden_features=2 * self.params.N_in_channels,
            out_features=self.params.N_out_channels,
            act_layer=nn.GELU,
            output_bias=True,
            input_format="nchw",
            drop_rate=0.0,
        )

        # tag a parameter as sharded. In a single-rank test no layer sets this by itself, but the
        # attribute is what convert_checkpoint keys off, so the round-trip has to preserve it.
        sharded_name, sharded_param = next(iter(model.named_parameters()))
        sharded_param.sharded_dims_mp = ["matmul", None]

        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        counters = {"iters": 7, "epoch": 3}

        # take one step so the optimizer actually carries state tensors
        inp = torch.randn(
            self.params.batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y
        )
        loss(model(inp), torch.zeros_like(model(inp))).backward()
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = os.path.join(tempdir, "ckpt_mp{mp_rank}.tar")

            Driver.save_checkpoint(
                checkpoint_path,
                model=model,
                loss=loss,
                optimizer=optimizer,
                scheduler=scheduler,
                counters=counters,
                checkpoint_mode=checkpoint_mode,
            )

            # load it back through the safe loader. A raise here means the safe globals
            # allowlist is out of sync with what save_checkpoint writes.
            checkpoint = load_checkpoint(checkpoint_path.format(mp_rank=0))

        self.assertIn("model_state", checkpoint)
        self.assertEqual(checkpoint["iters"], 7)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertIn("scheduler_state_dict", checkpoint)

        # the sharding metadata must survive the round-trip, otherwise convert_checkpoint
        # would silently treat a sharded tensor as unsharded
        restored = checkpoint["model_state"][sharded_name]
        self.assertTrue(
            hasattr(restored, "sharded_dims_mp"), f"sharding metadata lost for {sharded_name} in {checkpoint_mode} mode"
        )
        self.assertEqual(restored.sharded_dims_mp, ["matmul", None])


if __name__ == "__main__":
    unittest.main()
