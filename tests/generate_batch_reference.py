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

"""Write the golden batch reference, see tests/batch_reference.py.

Usage, from the repository root and on CPU::

    CUDA_VISIBLE_DEVICES="" python -m tests.generate_batch_reference

Only regenerate when the batch content is meant to change, and say why in the commit.
"""

import os

import numpy as np

from .batch_reference import REFERENCE_PATH, compute_reference


def main():
    reference = compute_reference()
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    np.savez_compressed(REFERENCE_PATH, **reference)

    size_kb = os.path.getsize(REFERENCE_PATH) / 1024
    print(f"wrote {len(reference)} arrays to {REFERENCE_PATH} ({size_kb:.0f} kB)")


if __name__ == "__main__":
    main()
