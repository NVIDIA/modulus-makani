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

"""ERA5 sources for ``data_process/convert_era5_to_makani_input.py`` and
``data_process/convert_era5_invariants_to_makani_input.py``.

The converter owns everything the sources have in common: the metadata, the
date range, the yearly output files and their layout, the split of work over
MPI ranks and the progress bar. A source only knows how to read its archive
and is expected to subclass :class:`Source`. The invariant converter is the
same arrangement without the time axis and without MPI, and its sources
subclass :class:`InvariantSource`.

Sources are kept free of MPI, so each backend module can be imported, and its
pieces tested, without an MPI installation; the backend modules are imported
lazily by the converter, so a source's own dependencies are only needed when it
is actually used.
"""

from typing import Dict, List, Optional, Tuple
import datetime as dt

import h5py as h5

# one sample of a yearly file: its index in the file and its valid time
Sample = Tuple[int, dt.datetime]

# the smallest piece of work a source wants to see on a single rank
Unit = List[Sample]


class Source(object):
    """Base class for the archives the converter can read from.

    The converter calls the methods in this order for every yearly file:
    :meth:`split_units` on all samples of the year, :meth:`begin_year` with the
    units assigned to the local rank, then :meth:`fill` on consecutive batches of
    up to :attr:`units_per_fill` of those units. :meth:`close` is called once at
    the very end.

    Constructors take the metadata dictionary and the MPI rank, the latter only
    so that a source can restrict informational output to a single rank,
    followed by source specific keyword options.
    """

    # number of units handed to a single fill call
    units_per_fill: int = 1

    def __init__(self, metadata: Dict, comm_rank: int):
        self.channel_names = metadata["coords"]["channel"]
        self.lat = metadata["coords"]["lat"]
        self.lon = metadata["coords"]["lon"]
        self.dhours = metadata["dhours"]
        self.comm_rank = comm_rank

    def split_units(self, samples: List[Sample]) -> List[Unit]:
        """Group the samples of a year into units that must not be split across ranks.

        Units are handed out to ranks as contiguous runs, in the order returned.
        The default puts every sample in a unit of its own.
        """
        return [[sample] for sample in samples]

    def begin_year(self, units: List[Unit]):
        """Called with all units this rank will fill for the current year, e.g. to schedule prefetching."""
        pass

    def fill(self, out: h5.File, entry_key: str, units: List[Unit]):
        """Write the samples of ``units`` into ``out[entry_key]``.

        Samples that cannot be provided are to be marked in ``out["valid_data"]``,
        which is initialized to all valid.
        """
        raise NotImplementedError

    def skipped_channel_indices(self) -> List[int]:
        """Indices of the metadata channels this source cannot provide.

        The converter writes these as missing: NaN in the fields and cleared in
        ``valid_data``. A source that cannot provide a channel and is not told to
        skip it should fail in its constructor instead.
        """
        return []

    def summary(self) -> Optional[str]:
        """Optional line printed once the conversion is done."""
        return None

    def close(self):
        """Release whatever the source holds open."""
        pass


class InvariantSource(object):
    """Base class for the archives the invariant converter can read from.

    The time invariant counterpart of :class:`Source`: there are no samples to
    distribute, so the converter calls :meth:`fill` exactly once and
    :meth:`close` after it.

    Constructors take the metadata dictionary, whose ``coords["channel"]`` lists
    the invariant fields to extract, followed by source specific keyword options.
    """

    def __init__(self, metadata: Dict):
        self.channel_names = metadata["coords"]["channel"]
        self.lat = metadata["coords"]["lat"]
        self.lon = metadata["coords"]["lon"]

    def fill(self, out: h5.File, entry_key: str):
        """Write every field the source can provide into ``out[entry_key][cidx]``."""
        raise NotImplementedError

    def skipped_channel_indices(self) -> List[int]:
        """Indices of the channels this source cannot provide, see :meth:`Source.skipped_channel_indices`."""
        return []

    def summary(self) -> Optional[str]:
        """Optional line printed once the conversion is done."""
        return None

    def close(self):
        """Release whatever the source holds open."""
        pass
