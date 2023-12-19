# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cdsapi
import numpy as np
import os
import argparse


def main(args):

    # get base path
    base_path = os.path.join(args.output_dir, "raw")
    os.makedirs(base_path, exist_ok=True)

    # get years
    years = args.years

    # do all months in a year
    months = [str(jj).zfill(2) for jj in range(1,13)]

    # timestamps
    timestamps = [str(t).zfill(2) + ":00" for t in
                  list(range(0, 24, args.hourly_subsample))]

    # get pressure levels
    pressure_levels = list(range(50,1050,50))

    # create api instance
    c = cdsapi.Client()

    # download the data
    for year in years:
        
        year_str = str(year)

        for month in months:

            month_str = month

            for pl in pressure_levels:

                print(f"Downloading {month_str}.{year_str} and pressure level {pl}.")

                file_str = os.path.join(base_path, f"pl_{pl}_{year_str}-{month_str}.nc")

                if os.path.isfile(file_str):
                    if (not args.overwrite):
                        print(f"File {file_str} already exists and overwrite flag not set, skipping.")
                        continue
                    else:
                        print(f"File {file_str} already exists but overwrite flag set, removing.")
                        os.remove(file_str)
                
                c.retrieve('reanalysis-era5-complete', 
                           {
                               'class': 'ea',
                               'expver': '1',
                               'levtype': 'pl',
                               'stream': 'oper',
                               'type': 'an',
                               'grid': [args.resolution, args.resolution],
                               'format': 'netcdf',
                               'levelist': f'{pl}',
                               # u, v, w, z, t, q
                               "param": "131/132/135.128/129.128/130.128/133.128",
                               'date': f'{year_str}-{month_str}-01/to/{year_str}-{month_str}-31',
                               'time': timestamps,
                           },
                           file_str)

            # we have only one pressure level here
            file_str = os.path.join(base_path, f"sfc_{year_str}-{month_str}.nc")

            if os.path.isfile(file_str):
                if (not args.overwrite):
                    print(f"File {file_str} already exists and overwrite flag not set, skipping.")
                    continue
                else:
                    print(f"File {file_str} already exists but overwrite flag set, removing.")
                    os.remove(file_str)
            
            c.retrieve("reanalysis-era5-complete",
                       {
                           'class': 'ea',
                           'expver': '1',
                           'levtype': 'sfc',
                           'stream': 'oper',
                           'type': 'an',
                           'grid': [args.resolution, args.resolution],
                           'format': 'netcdf',
                           'date': f'{year_str}-{month_str}-01/to/{year_str}-{month_str}-31',
                           'time': timestamps,
                           # 10u, 10v, 100u, 100v, 2t, sp, msl, tcvw
                           "param": "165.128/166.128/246.228/247.228/167.128/134.128/151.128/137.128",
                       }, file_str)

    print("Done!")

    return
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help="Path containing the output files.", required=True)
    parser.add_argument('--years', type=int, nargs='+', help="List of years to process.", required=True)
    parser.add_argument('--pressure_level', type=int, nargs='+', default=list(range(50,1050,50)), help="List of pressure levels to process.")
    parser.add_argument('--hourly_subsample', type=int, default=1, help="Temporal subsampling.")
    parser.add_argument('--resolution', type=float, default=0.25, help="Spatial resolution.")
    parser.add_argument('--overwrite', action='store_true', help="Set this flag in order to overwrite existing files.")
    args = parser.parse_args()

    main(args)
