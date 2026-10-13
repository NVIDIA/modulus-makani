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

import tempfile
from typing import Optional

import unittest
from parameterized import parameterized

import torch

from makani import Trainer, EnsembleTrainer, StochasticTrainer, AutoencoderTrainer
from makani.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, init_hdf5_dataset
from .testutils import H5_PATH, compare_tensors


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


class TestTrainer(unittest.TestCase):

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

        The step wrappers take and return the time axis explicitly; the stochastic
        interpolant takes the time steps folded into the channels.
        """
        if isinstance(trainer, StochasticTrainer):
            return trainer.model(inp)
        return trainer.model(inp.unsqueeze(1)).flatten(1, 2)

    @parameterized.expand(test_parameters, skip_on_empty=True)
    def test_trainer(self, trainer_handle, test_train, test_eval, devstring, verbose=False):

        # create device
        device = torch.device(devstring)

        # instantiate Trainer and run training
        self.trainer = trainer_handle(self.params, 0, device=device)
        self.trainer.train()

        # check that a checkpoint file exists
        checkpoint_dir = os.path.join(self.params.experiment_dir, "training_checkpoints")
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


if __name__ == "__main__":
    unittest.main()
