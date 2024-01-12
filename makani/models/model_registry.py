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
import importlib.util
from importlib.metadata import EntryPoint, entry_points
import importlib.metadata

import logging

from typing import List, Union
from functools import partial

import torch.nn as nn

from makani.utils.YParams import ParamsBase
from makani.models import SingleStepWrapper, MultiStepWrapper


def _construct_registry() -> dict:
    registry = {}
    entrypoints = entry_points(group="makani.models")
    for entry_point in entrypoints:
        registry[entry_point.name] = entry_point
    return registry


def _register_from_module(model: nn.Module, name: Union[str, None] = None) -> None:
    """
    registers a module in the registry
    """

    # Check if model is a torch module
    if not issubclass(model, nn.Module):
        raise ValueError(f"Only subclasses of torch.nn.Module can be registered. " f"Provided model is of type {type(model)}")

    # If no name provided, use the model's name
    if name is None:
        name = model.__name__

    # Check if name already in use
    if name in _model_registry:
        raise ValueError(f"Name {name} already in use")

    # Add this class to the dict of model registry
    _model_registry[name] = model


def _register_from_file(model_string: str, name: Union[str, None] = None) -> None:
    """
    parses a string and attempts to get the module from the specified location
    """

    assert len(model_string.split(":")) == 2
    model_path, model_handle = model_string.split(":")

    if not os.path.exists(model_path):
        raise ValueError(f"Expected string of format 'path/to/model_file.py:ModuleName' but {model_path} does not exist.")

    module_spec = importlib.util.spec_from_file_location(model_handle, model_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = getattr(module, model_handle)

    _register_from_module(model, name)


def register_model(model: Union[str, nn.Module], name: Union[str, None] = None) -> None:
    """
    Registers a model in the model registry under the provided name. If no name
    is provided, the model's name (from its `__name__` attribute) is used. If the
    name is already in use, raises a ValueError.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be registered. Can be an instance of any class.
    name : str, optional
        The name to register the model under. If None, the model's name is used.

    Raises
    ------
    ValueError
        If the provided name is already in use in the registry.
    """

    if isinstance(model, str):
        _register_from_file(model, name)
    else:
        _register_from_module(model, name)

def list_models() -> List[str]:
    """
    Returns a list of the names of all models currently registered in the registry.

    Returns
    -------
    List[str]
        A list of the names of all registered models. The order of the names is not
        guaranteed to be consistent.
    """
    return list(_model_registry.keys())


def get_model(params: ParamsBase, **kwargs) -> "torch.nn.Module":
    """
    Convenience routine that constructs the model passing parameters and kwargs.
    Unloads all the parameters in the params datastructure as a dict.

    Parameters
    ----------
    params : ParamsBase
        parameter struct.

    Returns
    -------
    model : torch.nn.Module
        The registered model.

    Raises
    ------
    KeyError
        If no model is registered under the provided name.
    """

    if params is not None:
        # makani requires that these entries are set in params for now
        inp_shape = (params.img_crop_shape_x, params.img_crop_shape_y)
        out_shape = (params.out_shape_x, params.out_shape_y) if hasattr(params, "out_shape_x") and hasattr(params, "out_shape_y") else inp_shape
        inp_chans = params.N_in_channels
        out_chans = params.N_out_channels

    if params.nettype not in _model_registry:
        logging.warning(f"Net type {params.nettype} does not exist in the registry. Trying to register it.")
        register_model(params.nettype, params.nettype)
        
    model_handle = _model_registry.get(params.nettype)
    if model_handle is not None:
        if isinstance(model_handle, (EntryPoint, importlib.metadata.EntryPoint)):
            model_handle = model_handle.load()
        model_handle = partial(model_handle, inp_shape=inp_shape, out_shape=out_shape, inp_chans=inp_chans, out_chans=out_chans, **params.to_dict())
    else:
        raise KeyError(f"No model is registered under the name {name}")

    # wrap into Multi-Step if requested
    if params.n_future > 0:
        model = MultiStepWrapper(params, model_handle)
    else:
        model = SingleStepWrapper(params, model_handle)

    return model


# initialize the internal state upon import
_model_registry = _construct_registry()
