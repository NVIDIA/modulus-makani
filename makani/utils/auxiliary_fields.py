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

import numpy as np
import torch
import h5py


def get_bias_correction(bias_correction_path, output_channels):
    """returns the bias correction from"""

    with h5py.File(bias_correction_path, "r") as f:
        bias = f["mean"][0, :]

    return bias


def get_copernicus_emb(copernicus_emb_path):
    """returns the copernicus embedding for each grid point. Values are floats, dimension 8."""

    # open npy
    emb = np.load(copernicus_emb_path)
    emb = torch.tensor(emb, dtype=torch.float32)

    return emb
