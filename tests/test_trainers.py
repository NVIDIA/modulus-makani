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

import os
import sys
import shutil

import tempfile
from typing import Optional

import unittest
from parameterized import parameterized

import torch

from makani import Trainer, EnsembleTrainer, StochasticTrainer, AutoencoderTrainer
from makani.utils import comm
from makani.utils.dataloaders.batch import TARGET, batch_to_device

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, init_hdf5_dataset
from .testutils import H5_PATH, compare_tensors, single_step_rollout


def init_params(
    exp_path: str,
    train_path: str,
    valid_path: str,
    stats_path: str,
    batch_size: int,
    ensemble_size: int,
    stochastic_size: int,
    n_history: int,
    n_future: int,
    normalization: str,
    num_data_workers: int,
):

    # instantiate params base
    params = get_default_parameters()

    # init paths
    params.train_data_path = train_path
    params.valid_data_path = valid_path
    params.min_path = os.path.join(stats_path, "mins.npy")
    params.max_path = os.path.join(stats_path, "maxs.npy")
    params.time_means_path = os.path.join(stats_path, "time_means.npy")
    params.global_means_path = os.path.join(stats_path, "global_means.npy")
    params.global_stds_path = os.path.join(stats_path, "global_stds.npy")
    params.time_diff_means_path = os.path.join(stats_path, "time_diff_means.npy")
    params.time_diff_stds_path = os.path.join(stats_path, "time_diff_stds.npy")

    # experiment direction
    params.experiment_dir = exp_path

    # checkpoint locations
    params.checkpoint_path = os.path.join(exp_path, "training_checkpoints/ckpt_mp{mp_rank}_v0.tar")
    params.best_checkpoint_path = os.path.join(exp_path, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")

    # general parameters
    params.dhours = 24
    params.h5_path = H5_PATH
    params.n_history = n_history
    params.n_future = n_future
    params.batch_size = batch_size
    params.ensemble_size = ensemble_size
    params.local_ensemble_size = ensemble_size
    params.stochastic_size = stochastic_size
    params.stochastic_interpolation_steps = stochastic_size
    params.normalization = normalization

    # performance parameters
    params.num_data_workers = num_data_workers

    # logging options
    params.log_to_screen = False
    params.log_to_wandb = False
    params.log_video = 0

    # test architecture
    params.nettype = "SFNO"
    params.num_layers = 2

    # losss
    params.loss = "l2"
    params.lr = 5e-4
    params.weight_decay = 0.0

    # optimizer
    params.optimizer_type = "AdamW"
    params.optimizer_beta1 = 0.9
    params.optimizer_beta2 = 0.95
    params.optimizer_max_grad_norm = 32

    # job size
    params.max_epochs = 1
    params.n_train_samples = 4
    params.n_eval_samples = 2

    # scheduler
    params.scheduler = "CosineAnnealingLR"
    params.scheduler_T_max = 10
    params.lr_warmup_steps = 0

    # other
    params.pretrained = False

    return params


class _TrainerTestCase(unittest.TestCase):
    """The data, the experiment directory and the parameters the trainer tests share."""

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        # create temporary directory
        cls.datadir = tempfile.TemporaryDirectory(dir=path)
        data_path = cls.datadir.name

        # init datasets and stats
        (
            cls.train_path,
            cls.n_train_samples,
            cls.valid_path,
            cls.n_eval_samples,
            cls.stats_path,
            cls.metadata_path,
            _,
        ) = init_hdf5_dataset(data_path)

    def setUp(self, path: Optional[str] = "/tmp"):

        disable_tf32()
        set_seed(333)

        # create temporary directory
        self.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = self.tmpdir.name

        exp_path = os.path.join(tmp_path, "experiment_dir")
        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, "training_checkpoints"))

        self.params = init_params(
            exp_path,
            self.train_path,
            self.valid_path,
            self.stats_path,
            batch_size=2,
            ensemble_size=3,
            stochastic_size=5,
            n_history=0,
            n_future=0,
            normalization="zscore",
            num_data_workers=0,
        )

        self.params.multifiles = True
        self.params.n_train_samples = self.n_train_samples
        self.params.n_eval_samples = self.n_eval_samples

        # set up data grid
        self.params.io_grid = [1, 1, 1]
        self.params.io_rank = [0, 0, 0]

        # self.num_steps = 5
        self.params.print_timings_frequency = 0

    @classmethod
    def tearDownClass(cls):
        cls.datadir.cleanup()

    def tearDown(self):
        self.tmpdir.cleanup()


class TestTrainer(_TrainerTestCase):

    test_parameters = [
        (Trainer, True, True, "cpu"),
        (EnsembleTrainer, True, True, "cpu"),
        (StochasticTrainer, False, False, "cpu"),
        (AutoencoderTrainer, False, False, "cpu"),
    ]
    if torch.cuda.is_available():
        test_parameters += [
            (Trainer, True, True, "cuda"),
            (EnsembleTrainer, True, True, "cuda"),
            (StochasticTrainer, False, False, "cuda"),
            (AutoencoderTrainer, False, False, "cuda"),
        ]

    @staticmethod
    def _forward(trainer, inp):
        """The trained model on a flat input, with a flat output for the comparisons.

        The StepWrapper takes a one-step rollout and returns the time axis
        explicitly; the stochastic interpolant takes the time steps folded into
        the channels.
        """
        if isinstance(trainer, StochasticTrainer):
            return trainer.model(inp)
        return trainer.model(single_step_rollout(inp.unsqueeze(1))).flatten(1, 2)

    @parameterized.expand(test_parameters, skip_on_empty=True)
    def test_trainer(self, trainer_handle, test_train, test_eval, devstring, verbose=False):

        # create device
        device = torch.device(devstring)

        # instantiate Trainer and run training
        self.trainer = trainer_handle(self.params, 0, device=device)
        self.trainer.train()

        # check that a checkpoint file exists
        with self.subTest(desc="checkpoint files"):
            self.assertTrue(os.path.isfile(self.params.checkpoint_path.format(mp_rank=comm.get_rank("model"))))
            self.assertTrue(os.path.isfile(self.params.best_checkpoint_path.format(mp_rank=comm.get_rank("model"))))

        # check that the number of epochs and iterations is right
        with self.subTest(desc="epoch counter vs max epoch counter"):
            self.assertEqual(self.trainer.epoch, self.params.max_epochs)

        # setup some dummy and remember the output of the model
        inp_shape = (
            self.params.batch_size,
            self.params.N_in_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )
        out_shape = (
            self.params.batch_size,
            self.params.N_out_channels,
            self.params.img_shape_x,
            self.params.img_shape_y,
        )

        # we do not need to remember gradients
        self.trainer._set_eval()

        # prepare some dummy data
        inp = torch.randn(*inp_shape).to(device)
        inp.requires_grad = True

        # forward pass and remember the output
        out_ref = self._forward(self.trainer, inp)
        with self.subTest(desc="output shape"):
            self.assertEqual(out_ref.shape, out_shape)

        # restore a new trainer from checkpoint
        self.params.resuming = True
        self.trainer_restored = trainer_handle(self.params, 0, device=device)

        # check that counters are still the same
        with self.subTest(desc="epoch and iteration counters"):
            self.assertEqual(self.trainer_restored.epoch, self.trainer.epoch)
            self.assertEqual(self.trainer_restored.iters, self.trainer.iters)

        # test that the models produce the same results
        if test_eval:
            self.trainer_restored._set_eval()
            out = self._forward(self.trainer_restored, inp)
            out_ref = self._forward(self.trainer, inp)
            with self.subTest(desc="test output vs reference"):
                self.assertTrue(compare_tensors("test output vs reference", out, out_ref, verbose=verbose))

        # redo this but in train mode
        if test_train:
            self.trainer._set_train()
            out = self._forward(self.trainer, inp)
            with self.subTest(desc="train output vs reference roundtrip 1"):
                self.assertTrue(compare_tensors("train output vs reference roundtrip 1", out, out_ref, verbose=verbose))
            out_ref = out

            self.trainer_restored._set_train()
            out = self._forward(self.trainer_restored, inp)
            with self.subTest(desc="train output vs reference roundtrip 2"):
                self.assertTrue(compare_tensors("train output vs reference roundtrip 2", out, out_ref, verbose=verbose))


def _optimizer(trainer):
    # the ensemble trainer names its optimizer and scheduler after the model
    return trainer.model_optimizer if hasattr(trainer, "model_optimizer") else trainer.optimizer


def _scheduler(trainer):
    return trainer.model_scheduler if hasattr(trainer, "model_scheduler") else trainer.scheduler


def _states_equal(msg, state1, state2, verbose=False):
    """Bitwise comparison of two (nested) state dicts, such as those of a model, an optimizer or a scheduler."""
    if isinstance(state1, torch.Tensor) or isinstance(state2, torch.Tensor):
        if not (isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor)):
            if verbose:
                print(f"{msg}: a tensor compared to a {type(state1 if isinstance(state2, torch.Tensor) else state2)}")
            return False
        return compare_tensors(msg, state1, state2, atol=0.0, rtol=0.0, verbose=verbose, shape_check=True)
    if isinstance(state1, dict):
        if not (isinstance(state2, dict) and (state1.keys() == state2.keys())):
            if verbose:
                print(f"{msg}: the keys differ")
            return False
        return all(_states_equal(f"{msg}.{key}", state1[key], state2[key], verbose=verbose) for key in state1)
    if isinstance(state1, (list, tuple)):
        if not (isinstance(state2, (list, tuple)) and (len(state1) == len(state2))):
            if verbose:
                print(f"{msg}: the lengths differ")
            return False
        return all(
            _states_equal(f"{msg}[{i}]", s1, s2, verbose=verbose) for i, (s1, s2) in enumerate(zip(state1, state2))
        )
    if state1 != state2:
        if verbose:
            print(f"{msg}: {state1} vs {state2}")
        return False
    return True


class TestTrainerCheckpointing(_TrainerTestCase):
    """
    Continuing a training run from a checkpoint, and starting a new one from the weights of another.

    Everything runs on the CPU, where the training is deterministic, so a resumed run has to
    reproduce the uninterrupted one bit for bit: whatever state the checkpoint misses, such as
    that of the optimizer, the scheduler or the counters, shows up as a difference.
    """

    _SEED = 4321

    @parameterized.expand([(Trainer,), (EnsembleTrainer,)], skip_on_empty=True)
    def test_resume_continues_training(self, trainer_handle, verbose=False):
        device = torch.device("cpu")

        # the first epoch, which writes the checkpoint to resume from. The checkpoint path has no
        # version in it, so the second epoch overwrites it, and a copy is kept aside
        self.params.max_epochs = 1
        trainer = trainer_handle(self.params, 0, device=device)
        trainer.train()
        checkpoint_path = self.params.checkpoint_path.format(mp_rank=comm.get_rank("model"))
        shutil.copyfile(checkpoint_path, checkpoint_path + ".epoch1")
        iters_epoch1 = trainer.iters

        # the reference continues in memory with the second epoch. train() starts at the epoch the
        # trainer was constructed with, so it is moved on to the one the first call ended at.
        # The data order and the dropout are drawn from the global generator, which is seeded
        # alike before the second epoch of both runs
        self.params.max_epochs = trainer.params.max_epochs = 2
        trainer.start_epoch = trainer.epoch
        set_seed(self._SEED)
        trainer.train()

        # a new trainer resumes from the checkpoint of the first epoch and trains the second
        shutil.copyfile(checkpoint_path + ".epoch1", checkpoint_path)
        self.params.resuming = True
        trainer_resumed = trainer_handle(self.params, 0, device=device)

        with self.subTest(desc="counters after restore"):
            self.assertEqual(trainer_resumed.epoch, 1)
            self.assertEqual(trainer_resumed.iters, iters_epoch1)

        set_seed(self._SEED)
        trainer_resumed.train()

        with self.subTest(desc="counters after training"):
            self.assertEqual(trainer_resumed.epoch, trainer.epoch)
            self.assertEqual(trainer_resumed.iters, trainer.iters)

        with self.subTest(desc="model"):
            self.assertTrue(
                _states_equal("model", trainer_resumed.model.state_dict(), trainer.model.state_dict(), verbose=verbose)
            )

        with self.subTest(desc="optimizer"):
            self.assertTrue(
                _states_equal(
                    "optimizer",
                    _optimizer(trainer_resumed).state_dict(),
                    _optimizer(trainer).state_dict(),
                    verbose=verbose,
                )
            )

        with self.subTest(desc="scheduler"):
            self.assertTrue(
                _states_equal(
                    "scheduler",
                    _scheduler(trainer_resumed).state_dict(),
                    _scheduler(trainer).state_dict(),
                    verbose=verbose,
                )
            )

    @parameterized.expand(
        [(Trainer,), (EnsembleTrainer,), (StochasticTrainer,), (AutoencoderTrainer,)], skip_on_empty=True
    )
    def test_pretrained(self, trainer_handle, verbose=False):
        device = torch.device("cpu")

        # the run whose checkpoint the new ones start from
        self.params.max_epochs = 1
        source = trainer_handle(self.params, 0, device=device)
        source.train()
        source_lrs = [group["lr"] for group in _optimizer(source).param_groups]

        self.params.pretrained = True
        self.params.resuming = False
        self.params.pretrained_checkpoint_path = self.params.checkpoint_path

        def _model_matches(trainer):
            return _states_equal("model", trainer.model.state_dict(), source.model.state_dict(), verbose=verbose)

        # by default everything is taken over from the checkpoint
        trainer = trainer_handle(self.params, 0, device=device)
        with self.subTest(desc="defaults: model"):
            self.assertTrue(_model_matches(trainer))
        with self.subTest(desc="defaults: counters"):
            self.assertEqual(trainer.epoch, source.epoch)
            self.assertEqual(trainer.iters, source.iters)
        with self.subTest(desc="defaults: optimizer"):
            self.assertTrue(
                _states_equal(
                    "optimizer", _optimizer(trainer).state_dict(), _optimizer(source).state_dict(), verbose=verbose
                )
            )
        with self.subTest(desc="defaults: scheduler"):
            self.assertTrue(
                _states_equal(
                    "scheduler", _scheduler(trainer).state_dict(), _scheduler(source).state_dict(), verbose=verbose
                )
            )

        # finetuning: only the weights are taken over, the optimizer, the scheduler and the
        # counters start afresh
        self.params.load_optimizer = False
        self.params.load_scheduler = False
        self.params.load_counters = False
        trainer = trainer_handle(self.params, 0, device=device)
        with self.subTest(desc="finetune: model"):
            self.assertTrue(_model_matches(trainer))
        with self.subTest(desc="finetune: counters"):
            self.assertEqual(trainer.epoch, 0)
            self.assertEqual(trainer.iters, 0)
        with self.subTest(desc="finetune: optimizer"):
            self.assertEqual(len(_optimizer(trainer).state_dict()["state"]), 0)
            self.assertTrue(all(group["lr"] == self.params.lr for group in _optimizer(trainer).param_groups))
        with self.subTest(desc="finetune: scheduler"):
            self.assertEqual(_scheduler(trainer).last_epoch, 0)

        # the optimizer is restored, with its learning rate replaced by the configured one
        self.params.load_optimizer = True
        self.params.load_scheduler = True
        self.params.load_counters = True
        self.params.override_lr = True
        self.params.lr = 1e-3
        self.assertTrue(all(lr != self.params.lr for lr in source_lrs))
        trainer = trainer_handle(self.params, 0, device=device)
        with self.subTest(desc="override lr: learning rate"):
            self.assertTrue(all(group["lr"] == self.params.lr for group in _optimizer(trainer).param_groups))
        with self.subTest(desc="override lr: optimizer state"):
            self.assertTrue(
                _states_equal(
                    "optimizer state",
                    _optimizer(trainer).state_dict()["state"],
                    _optimizer(source).state_dict()["state"],
                    verbose=verbose,
                )
            )


class _RecordingMetrics:
    """Stands in for the metrics handler of a trainer, and records what validation hands it."""

    def __init__(self):
        self.updates = []

    def zero_buffers(self):
        pass

    def finalize(self):
        return {}

    def update(self, pred, targ, loss, idt):
        self.updates.append((pred.clone(), targ.clone(), idt))


class TestTrainerRollouts(_TrainerTestCase):
    """
    Training and validation over several steps.

    Training runs end to end over a rollout of ``n_future + 1`` steps. The
    validation over ``valid_autoreg_steps + 1`` lead times is then compared
    with the rollout of the same batches through the trained model, with the
    noise drawn from the same seed: the predictions handed to the metrics at
    each lead time have to be the ones of that step, split into their ensemble
    members, and the targets those of that step.
    """

    _NOISE = {"type": "diffusion", "mode": "concatenate", "n_channels": 1}
    _VALID_STEPS = 2
    _SEED = 1234

    test_parameters = [
        # trainer, n_future, push forward, input noise, ensemble size, zenith
        (Trainer, 1, False, None, 1, False),
        (Trainer, 1, True, None, 1, True),
        (EnsembleTrainer, 1, False, _NOISE, 2, True),
    ]

    def _trainer(self, trainer_handle, n_future, push_forward, noise, ensemble_size, zenith):
        params = self.params
        params.n_future = n_future
        params.valid_autoreg_steps = self._VALID_STEPS
        params.multistep = {"push_forward": push_forward}
        params.ensemble_size = params.local_ensemble_size = ensemble_size
        params.add_zenith = zenith
        if noise is not None:
            params.input_noise = noise
        return trainer_handle(params, 0, device=torch.device("cpu"))

    @parameterized.expand(test_parameters, skip_on_empty=True)
    def test_validation_matches_the_rollout(self, trainer_handle, n_future, push_forward, noise, E, zenith):
        trainer = self._trainer(trainer_handle, n_future, push_forward, noise, E, zenith)
        trainer.train()

        pp = trainer.preprocessor
        trainer.metrics = _RecordingMetrics()
        pp.set_rng(reset=True, seed=self._SEED)
        trainer.validate_one_epoch(trainer.epoch)
        updates = trainer.metrics.updates

        pp.set_rng(reset=True, seed=self._SEED)
        trainer.model.eval()
        n_steps = self._VALID_STEPS + 1
        n_batches = 0
        with torch.no_grad():
            for data in trainer.valid_dataloader:
                batch = batch_to_device(data, trainer.device)
                rollout = trainer._rollout(batch, ensemble_size=E)
                preds = trainer.model(rollout)

                for k in range(n_steps):
                    pred, targ, idt = updates[n_batches * n_steps + k]
                    with self.subTest(batch=n_batches, step=k):
                        self.assertEqual(idt, k)
                        expected = pp.flatten_history(preds[:, k : k + 1])
                        if trainer_handle is EnsembleTrainer:
                            expected = rollout.unfold(expected)
                        self.assertTrue(torch.equal(pred, expected), "predictions")
                        self.assertTrue(torch.equal(targ, pp.flatten_history(batch[TARGET][:, k : k + 1])), "targets")
                n_batches += 1

        self.assertGreater(n_batches, 0)
        self.assertEqual(len(updates), n_batches * n_steps)

    def test_stochastic_validation_rolls_out(self):
        # the interpolant trains on single steps, but validates over several, with the zenith angle
        params = self.params
        params.valid_autoreg_steps = self._VALID_STEPS
        params.add_zenith = True
        trainer = StochasticTrainer(params, 0, device=torch.device("cpu"))
        trainer.train()

        trainer.metrics = _RecordingMetrics()
        trainer.validate_one_epoch(trainer.epoch)
        updates = trainer.metrics.updates

        n_steps = self._VALID_STEPS + 1
        self.assertGreater(len(updates), 0)
        self.assertEqual([idt for *_, idt in updates], list(range(n_steps)) * (len(updates) // n_steps))
        for pred, targ, idt in updates:
            with self.subTest(step=idt):
                self.assertEqual(pred.shape[0], targ.shape[0])
                self.assertEqual(pred.shape[1], params.local_ensemble_size)
                self.assertEqual(tuple(pred.shape[2:]), tuple(targ.shape[1:]))
                self.assertTrue(torch.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
