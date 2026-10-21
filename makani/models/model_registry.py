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
import importlib.util

from importlib.metadata import entry_points

import logging
import warnings

from typing import List, Union
from functools import partial

import torch
import torch.nn as nn

from makani.utils.YParams import ParamsBase, ensure_resampled_shapes
from makani.models import StepWrapper
from makani.models import StochasticInterpolantWrapper
from makani.utils.dataloaders.data_helpers import get_data_normalization


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
        raise ValueError(
            f"Only subclasses of torch.nn.Module can be registered. " f"Provided model is of type {type(model)}"
        )

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

    if len(model_string.split(":")) != 2:
        raise ValueError(f"Expected model string of format 'path/to/model_file.py:ModuleName' but got {model_string}")
    model_path, model_handle = model_string.split(":")

    if not os.path.exists(model_path):
        raise ValueError(
            f"Expected string of format 'path/to/model_file.py:ModuleName' but {model_path} does not exist."
        )

    # Load under a dotted module name derived from the file (not the class handle) and register
    # it in sys.modules. Otherwise the module's __name__ is the bare class name and is not
    # importable, which breaks torch.compile: dynamo resolves a traced function's module via
    # f_globals["__name__"] and calls importlib.import_module on it
    # (ModuleNotFoundError: No module named 'AtmoSphericNeuralOperatorNet'). Entry-point models
    # already get a proper module name; this brings the file-path fallback in line.
    module_name = "makani.models._dynamic." + os.path.splitext(os.path.basename(model_path))[0]
    module_spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
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


def warn_multistep_deprecated(multistep: bool):
    """Warn that ``multistep`` no longer selects a wrapper, if a caller still asks for one."""
    if multistep:
        warnings.warn(
            "multistep is deprecated and has no effect: the model wrapper serves single steps and rollouts, "
            "and the number of steps comes from the Rollout",
            FutureWarning,
            stacklevel=3,
        )


def get_model(
    params: ParamsBase, use_stochastic_interpolation: bool = False, multistep: bool = False
) -> "torch.nn.Module":
    """
    Convenience routine that constructs the model from the parameters.
    Unloads all the parameters in the params datastructure as a dict.

    Parameters
    ----------
    params : ParamsBase
        parameter struct.
    use_stochastic_interpolation : bool, optional
        Wrap the network in a :class:`StochasticInterpolantWrapper` instead of a
        :class:`~makani.models.stepper.StepWrapper`. By default ``False``.
    multistep : bool, optional
        Deprecated and without effect: the same wrapper serves single steps and
        rollouts, and the number of steps comes from the
        :class:`~makani.utils.dataloaders.rollout.Rollout`. ``True`` warns.

    Returns
    -------
    model : torch.nn.Module
        The registered network, wrapped in a
        :class:`~makani.models.stepper.StepWrapper` (or a
        :class:`StochasticInterpolantWrapper`).

    Raises
    ------
    KeyError
        If no model is registered under the provided name.
    """

    warn_multistep_deprecated(multistep)

    # conditional import for constraints
    if hasattr(params, "constraints"):
        from makani.models.parametrizations import ConstraintsWrapper

    if params is not None:
        # callers outside training (model packages predating resampling, earth2studio)
        # may not carry the resampled shapes; fall back to the unresampled ones. This
        # also covers Preprocessor2D, which is constructed further down from this same
        # params object.
        ensure_resampled_shapes(params)

        # makani requires that these entries are set in params for now
        inp_shape = (params.img_shape_x_resampled, params.img_shape_y_resampled)
        out_shape = (
            (params.out_shape_x, params.out_shape_y)
            if hasattr(params, "out_shape_x") and hasattr(params, "out_shape_y")
            else inp_shape
        )
        inp_chans = params.N_in_channels
        out_chans = params.N_out_channels

        if hasattr(params, "constraints"):
            cwrap = ConstraintsWrapper(
                constraints=params.constraints,
                channel_names=params.channel_names,
                bias=None,
                scale=None,
                model_handle=None,
            )
            out_chans = cwrap.N_in_channels

    # in the case that the model is not found in the model registry, we try to register it, given that it is a valid filepath:entrypoint
    if params.nettype not in _model_registry:
        logging.warning(f"Net type {params.nettype} does not exist in the registry. Trying to register it.")
        register_model(params.nettype, params.nettype)

    model_handle = _model_registry.get(params.nettype)
    if model_handle is not None:
        # Registry entries are either EntryPoint objects (from entry_points discovery)
        # or direct class refs (from register_model). Resolve EntryPoints to the callable.
        if hasattr(model_handle, "load") and callable(model_handle.load):
            model_handle = model_handle.load()

        model_kwargs = params.to_dict()

        # pass normalization statistics to the model
        normalization_mode = params.get("normalization", "none")
        if normalization_mode in ["zscore", "minmax"] or isinstance(normalization_mode, dict):
            if not hasattr(params, "out_channels"):
                raise ValueError(
                    f"normalization='{normalization_mode}' requires params.out_channels "
                    "to slice the normalization stats to the model's output channels"
                )
            bias, scale = get_data_normalization(params)
            if bias is not None:
                bias = bias.flatten()[params.out_channels]
            if scale is not None:
                scale = scale.flatten()[params.out_channels]
            if bias is not None and scale is not None:
                model_kwargs["normalization_means"] = bias
                model_kwargs["normalization_stds"] = scale

        hydrostatic_balance_means = params.get("hydrostatic_balance_means_path", None)
        if hydrostatic_balance_means is not None:
            from makani.utils.dataloaders.data_helpers import get_hydrostatic_balance_climatology

            hydrostatic_balance_means = get_hydrostatic_balance_climatology(params)
            model_kwargs["hydrostatic_balance_means"] = hydrostatic_balance_means

        # create model handle
        model_handle = partial(
            model_handle,
            inp_shape=inp_shape,
            out_shape=out_shape,
            inp_chans=inp_chans,
            out_chans=out_chans,
            **model_kwargs,
        )
    else:
        raise KeyError(f"No model is registered under the name {params.nettype}")

    # use the constraint wrapper
    if hasattr(params, "constraints"):
        # we need this in order to unormalize the data:
        # scale and bias
        bias, scale = get_data_normalization(params)
        bias = torch.from_numpy(bias)[:, params.out_channels, ...].to(torch.float32)
        scale = torch.from_numpy(scale)[:, params.out_channels, ...].to(torch.float32)

        # create a new wrapper handle
        model_handle = partial(
            ConstraintsWrapper,
            constraints=params.constraints,
            channel_names=params.channel_names,
            bias=bias,
            scale=scale,
            model_handle=model_handle,
        )

    if not use_stochastic_interpolation:
        model = StepWrapper(params, model_handle)
    else:
        model = StochasticInterpolantWrapper(
            params,
            model_handle,
            noise_epsilon=params.get("noise_epsilon", 1.0),
            use_foellmer=params.get("use_foellmer", False),
        )

    return model


# initialize the internal state upon import
_model_registry = _construct_registry()
