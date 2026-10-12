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

"""Time invariant input features, read from a single file with a channel axis.

The invariants file is written by ``data_process/convert_era5_invariants_to_makani_input.py``:
``fields`` is ``(channel, lat, lon)``, with ``channel``, ``lat``, ``lon`` and ``valid_data``
alongside. The configuration selects channels from it and says how each is encoded::

    invariants_path: /invariants/invariants.h5
    invariants:
      - {channel: z, encoding: normalize}
      - {channel: lsm, encoding: onehot, num_classes: 2, rounding: floor}

Encodings:

* ``normalize``: one channel, z-scored (area weighted if ``normalize_static_features`` is set).
* ``raw``: one channel, as stored.
* ``onehot``: ``num_classes`` channels. Values are rounded to integers with ``rounding``
  (``round``, the default, or ``floor``) and have to lie in ``[0, num_classes)``.

This module is kept free of torch, so that the channel bookkeeping can use it.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import h5py as h5

ENCODINGS = ("normalize", "raw", "onehot")
ROUNDINGS = ("round", "floor")

# options of the per-file invariants this module replaced; a config or checkpoint that
# still enables them would otherwise silently lose its invariant inputs
LEGACY_OPTIONS = ("add_orography", "add_landmask", "add_soiltype")


class InvariantFeature(NamedTuple):
    """One invariant channel of the model input, and how it is encoded."""

    channel: str
    encoding: str
    num_classes: Optional[int] = None
    rounding: str = "round"

    @property
    def num_channels(self) -> int:
        return self.num_classes if self.encoding == "onehot" else 1


def check_legacy_invariant_options(params):
    """Fail on the discontinued per-file invariant options.

    ``params`` is anything with a dict style ``get``, i.e. a dict or the run parameters.
    """
    enabled = [name for name in LEGACY_OPTIONS if params.get(name, False)]
    if enabled:
        raise ValueError(
            f"The options {enabled} are discontinued. Write an invariants file with "
            f"data_process/convert_era5_invariants_to_makani_input.py and select its channels with "
            f"'invariants_path' and 'invariants' instead, e.g. "
            f"[{{channel: z, encoding: normalize}}, {{channel: lsm, encoding: onehot, num_classes: 2, rounding: floor}}] "
            f"for the former add_orography and add_landmask."
        )


def parse_invariant_features(specs: Optional[Sequence[Dict]]) -> List[InvariantFeature]:
    """Validate the ``invariants`` configuration entry and turn it into :class:`InvariantFeature` s."""
    if not specs:
        return []

    features = []
    for spec in specs:
        spec = dict(spec)
        unknown = set(spec) - set(InvariantFeature._fields)
        if unknown:
            raise ValueError(f"Unknown keys {sorted(unknown)} in invariant {spec}.")
        if "channel" not in spec or "encoding" not in spec:
            raise ValueError(f"Invariant {spec} needs both 'channel' and 'encoding'.")
        if spec["encoding"] not in ENCODINGS:
            raise ValueError(f"Unknown encoding '{spec['encoding']}' of invariant {spec}, expected one of {ENCODINGS}.")

        if spec["encoding"] == "onehot":
            # the class count is configured rather than inferred from the data, so that the
            # channel count is known without reading the file
            num_classes = spec.get("num_classes", None)
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError(f"One-hot invariant {spec} needs an integer 'num_classes' of at least 2.")
            if spec.get("rounding", "round") not in ROUNDINGS:
                raise ValueError(
                    f"Unknown rounding '{spec['rounding']}' of invariant {spec}, expected one of {ROUNDINGS}."
                )
        elif "num_classes" in spec or "rounding" in spec:
            raise ValueError(f"'num_classes' and 'rounding' only apply to one-hot invariants, got {spec}.")

        features.append(InvariantFeature(**spec))

    channels = [feature.channel for feature in features]
    duplicates = sorted({channel for channel in channels if channels.count(channel) > 1})
    if duplicates:
        raise ValueError(f"Invariants {duplicates} are selected more than once.")

    return features


def read_invariants(path: str, channels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``channels`` from an invariants file.

    Returns
    -------
    fields : np.ndarray
        ``(len(channels), nlat, nlon)`` float32, in the order requested.
    lat, lon : np.ndarray
        The coordinates of the file.

    Raises
    ------
    ValueError
        If a channel is not in the file, is marked invalid there, or holds non-finite values.
    """
    with h5.File(path, "r") as f:
        available = [name.decode() if isinstance(name, bytes) else str(name) for name in f["channel"][...]]
        missing = [channel for channel in channels if channel not in available]
        if missing:
            raise ValueError(f"Invariants {missing} not found in {path}, which holds {available}.")

        indices = [available.index(channel) for channel in channels]
        invalid = [channel for channel, idx in zip(channels, indices) if not f["valid_data"][idx]]
        if invalid:
            raise ValueError(f"Invariants {invalid} are marked invalid in {path}.")

        fields = np.stack([f["fields"][idx] for idx in indices]).astype(np.float32)
        lat, lon = f["lat"][...], f["lon"][...]

    nonfinite = [channel for channel, field in zip(channels, fields) if not np.all(np.isfinite(field))]
    if nonfinite:
        raise ValueError(f"Invariants {nonfinite} in {path} hold non-finite values.")

    return fields, lat, lon
