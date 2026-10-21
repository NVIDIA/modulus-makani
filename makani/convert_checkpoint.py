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

import argparse
import os
import glob
from collections import OrderedDict

import torch
from torch import nn

from makani.models.model_package import LocalPackage

# model registry
from makani.models import model_registry

# distributed computing stuff
from makani.utils.driver import Driver
from makani.utils.checkpoint_helpers import load_checkpoint
from makani.utils.YParams import ParamsBase


def parse_comm_grid(checkpoints):
    """
    Extracts a dict with all comm dimensions and file to comm ranks mapping
    """

    # mapping of file to comm ranks
    file_comm_ranks = {}

    # find all the comm names that exist
    comm_dims = {}
    for key, value in checkpoints.items():
        comm_dict = value["comm_grid"]

        # compute a dictionary which is comm name -> rank
        # this will be used as index in comm_mapping
        current_crank = {}
        for cname in comm_dict:
            current_crank[cname] = comm_dict[cname]["rank"]

            if cname not in comm_dims:
                comm_dims[cname] = comm_dict[cname]["size"]

        file_comm_ranks[key] = current_crank

        # do the mapping of current_crank -> file
        current_crank = tuple(current_crank.items())

    # assign each checkpoint
    return comm_dims, file_comm_ranks


def get_params(path):
    config = os.path.join(path, "config.json")
    return ParamsBase.from_json(config)


def consolidate_checkpoints(input_path, output_path, checkpoint_version=0):
    """
    Conversion routine for loading the model and saving it using the flexible format

    Parameters
    ============
    input_path: str
        Path from which to load the checkpoint
    output_path: str
        Output path to store the checkpoint
    """

    # get the params datastructure
    params = ParamsBase.from_json(os.path.join(input_path, "config.json"))

    # adjust checkpoint_path to be inside of ``path``. The checkpoint may not be in
    # the same location it was during training.
    if checkpoint_version == "best":
        checkpoint_template = os.path.basename(params.best_checkpoint_path)
    else:
        checkpoint_template = os.path.basename(params.checkpoint_path)

    print(
        os.path.join(input_path, "training_checkpoints", checkpoint_template).format(
            mp_rank="*", checkpoint_version=checkpoint_version
        )
    )

    # check for the first checkpoint
    checkpoint_paths = sorted(
        glob.glob(
            os.path.join(input_path, "training_checkpoints", checkpoint_template).format(
                mp_rank="*", checkpoint_version=checkpoint_version
            )
        )
    )

    print(checkpoint_paths)

    # load the static data necessary for instantiating the preprocessor (required due to the way the registry works)
    LocalPackage._load_static_data(LocalPackage(input_path), params)

    # get the model
    model = model_registry.get_model(params)

    # open all files in mmap mode
    checkpoints = OrderedDict()
    for checkpoint in checkpoint_paths:
        checkpoints[checkpoint] = load_checkpoint(checkpoint, map_location="cpu", mmap=True)

    # parse comm grids: extract comm dimensions and get a mapping of file to comm rank
    comm_dims, comm_ranks = parse_comm_grid(checkpoints)

    # extract model state
    model_states = OrderedDict()
    for key, value in checkpoints.items():
        state_dict = value["model_state"]
        nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
        model_states[key] = state_dict

    gathered_state_dict = OrderedDict()
    parameter_keys = model_states[checkpoint_paths[0]].keys()
    for pname in parameter_keys:

        p = model_states[checkpoint_paths[0]][pname]
        if not hasattr(p, "sharded_dims_mp"):
            gathered_state_dict[pname] = p.clone()
        else:
            # get the split shapes and the rank in each dimension
            split_shapes = []
            for idd, cdim in enumerate(p.sharded_dims_mp):

                if cdim is None:
                    # if the dimensions is not split we use the entire range
                    split_shapes.append([0, p.shape[idd]])
                else:
                    # if the dimension is sharded dwe need to iterate over
                    split_sizes = [0 for _ in range(comm_dims[cdim] + 1)]

                    # iterate over files and write split sizes
                    for cfile in model_states.keys():
                        crank = comm_ranks[cfile][cdim]
                        ssize = model_states[cfile][pname].shape[idd]
                        split_sizes[crank + 1] = ssize

                    # do the cumsum here to get the split shape
                    split_shape = []
                    cumsum = 0
                    for sshape in split_sizes:
                        cumsum += sshape
                        split_shape.append(cumsum)

                    split_shapes.append(split_shape)

            # initialize the empty tensor
            gathered_shape = [s[-1] for s in split_shapes]
            gathered_state_dict[pname] = torch.empty(*gathered_shape, dtype=p.dtype)

            # iterate over all the files and copy the range for each of the split shapes into the consolidated tensor
            for cfile in model_states.keys():
                cranks = [comm_ranks[cfile][cdim] if cdim is not None else 0 for cdim in p.sharded_dims_mp]
                idx_ranges = tuple(
                    slice(split_shapes[idd][crank], split_shapes[idd][crank + 1]) for idd, crank in enumerate(cranks)
                )
                gathered_state_dict[pname][idx_ranges] = model_states[cfile][pname]

    print("Loading model using the consolidated checkpoint")
    # finally, load the gathered state dict
    model.load_state_dict(gathered_state_dict, strict=True)

    # save the model
    print(f"Saving checkpoint in flexible format to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    Driver.save_checkpoint(
        os.path.join(output_path, checkpoint_template).format(mp_rank=0, checkpoint_version=checkpoint_version),
        model=model,
        checkpoint_mode="flexible",
    )


def average_checkpoints(input_path, output_path):
    """
    Conversion routine for averaging multiple checkpoints

    Parameters
    ============
    input_path: str
        Path from which to load the checkpoint
    output_path: str
        Output path to store the checkpoint
    """

    # check for the first checkpoint
    checkpoint_paths = input_path
    checkpoint_template = os.path.basename(checkpoint_paths[0])

    # checkpoint output path
    checkpoint_output_path = os.path.join(output_path, checkpoint_template)

    print(f"opening base checkpoint {checkpoint_paths[0]}")
    base_checkpoint = load_checkpoint(checkpoint_paths[0], map_location="cpu")
    model_state_dict = base_checkpoint["model_state"]

    # this is reworked to avoid loading modules related t
    for checkpoint_file in checkpoint_paths[1:]:
        print(f"processing {checkpoint_file}")
        current_checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
        current_model_state = current_checkpoint["model_state"]

        for k, v in model_state_dict.items():
            if k in current_model_state.keys():
                model_state_dict[k] += current_model_state[k]
            else:
                raise ValueError(f"key {k} not present in current state dict.")

    for k, v in model_state_dict.items():
        model_state_dict[k] /= len(checkpoint_paths)

    # save the model
    print(f"Saving averaged checkpoint to {checkpoint_output_path}")
    store_dict = {"model_state": model_state_dict}
    torch.save(store_dict, checkpoint_output_path)


def strip_module(input_path):
    """
    Conversion routine for stripping 'module' from the state dict of the model. This overwrites the original checkpoint.

    Parameters
    ============
    input_path: str
        Path from which to load the checkpoint
    """

    # adjust checkpoint_path to be inside of ``path``. The checkpoint may not be in
    # the same location it was during training.
    checkpoint_template = "*.tar"

    # check for the first checkpoint
    checkpoint_paths = sorted(glob.glob(os.path.join(input_path, "training_checkpoints", checkpoint_template)))

    for checkpoint_file in checkpoint_paths:
        print(f"processing {checkpoint_file}")
        checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
        nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["model_state"], "module.")
        torch.save(checkpoint, checkpoint_file)


help_str = """Helper script to fix checkpoints. Support conversion of legacy (distributed) checkpoints to a single consolidated checkpoint in flexible format. Alternatively also supports stripping 'module' in the model state dict. Finally, it also supports averaging all checkpoints found in the directory.

This script requires the necessary checkpoint files as well as the output parameters config file to restore the checkpoint.

Example::

    python3 convert_checkpoint.py --input /path_to_directory/config_target/ngpu256_sp4/ --output /output_path/checkpoint_name --mode consolidate
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage=help_str)
    parser.add_argument(
        "--input", nargs="+", help="Root directory where the checkpoint and parameters file are stored.", required=True
    )
    parser.add_argument("--output", help="Target location to save the collected checkpoint.", required=False)
    parser.add_argument(
        "--mode",
        default="consolidate",
        type=str,
        choices=["consolidate", "strip_module", "average"],
        help="Specify how the checkpoints should be modified.",
    )
    parser.add_argument(
        "--checkpoint_version", default="0", type=str, help="Select checkpoint version. Only relevant for conversion"
    )
    args = parser.parse_args()

    print(f"Running convert_checkpoint in {args.mode} mode")

    checkpoint_version = int(args.checkpoint_version) if args.checkpoint_version != "best" else args.checkpoint_version

    if args.output is None:
        args.output = args.input

    if args.mode == "consolidate":
        print(f"Launching converions in consolidate mode on {args.input}")
        consolidate_checkpoints(args.input[0], args.output, checkpoint_version=checkpoint_version)
    elif args.mode == "average":
        print(f"Launching averaging of checkpoints found in {args.input}")
        average_checkpoints(args.input, args.output)
    elif args.mode == "strip_module":
        print(f"Launching converions in strip_module mode on {args.input[0]}")
        strip_module(args.input)
    else:
        raise ValueError(f"Unknown conversion mode {args.mode}")
