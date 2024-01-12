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


def get_auxiliary_channels(
    add_zenith=False,
    add_grid=False,
    grid_type=None,
    grid_num_frequencies=0,
    add_orography=False,
    add_landmask=False,
    **kwargs,
):
    """
    Auxiliary routine to return the list of appended channel names. Must match behavior of preprocessor and dataloader
    """
    channel_names = []

    if add_zenith:
        channel_names.append("xzen")

    if add_grid:
        if grid_type == "sinusoidal":
            for f in range(1, grid_num_frequencies + 1):
                channel_names += [f"xsgrlat{f}", f"xsgrlon{f}"]
        else:
            channel_names += ["xgrlat", "xgrlon"]

    if add_orography:
        channel_names.append("xoro")

    if add_landmask:
        channel_names += ["xlsml", "xlsms"]

    return channel_names
