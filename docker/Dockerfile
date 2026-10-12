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

ARG DLFW_VERSION=25.12
FROM nvcr.io/nvidia/pytorch:${DLFW_VERSION}-py3

# update repo info
RUN apt update -y && apt install -y libibmad5

# upgrade cmake
RUN apt remove cmake -y && \
    pip install cmake --upgrade

# install mpi4py
RUN pip install mpi4py

# install onnx
RUN pip install onnxruntime && \
    if [ "$(uname -m)" = "x86_64" ]; then pip install onnxruntime-gpu; fi

# hdf5 and h5py
ENV HDF5_VERSION=2.0.0
RUN cd /tmp && wget https://support.hdfgroup.org/releases/hdf5/v2_0/v2_0_0/downloads/hdf5-${HDF5_VERSION}.tar.gz && \
    gzip -cd hdf5-${HDF5_VERSION}.tar.gz | tar xvf - && \
    mkdir hdf5-${HDF5_VERSION}/build && cd hdf5-${HDF5_VERSION}/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/hdf5 \
    -DHDF5_ENABLE_OPTIMIZATION=ON \
    -DHDF5_ENABLE_ZLIB_SUPPORT=ON \
    -DHDF5_ENABLE_DIRECT_VFD=ON \
    -DHDF5_ENABLE_PARALLEL=ON \
    -DHDF5_TEST_API=ON \
    -DHDF5_TEST_VFD=ON \
    -DHDF5_TEST_PARALLEL=ON \
    .. && \
    make -j 8 && make install && \
    mkdir -p /opt/hdf5/lib/plugin
RUN CC="mpicc" HDF5_MPI=ON H5PY_DIRECT_VFD=1 HDF5_DIR=/opt/hdf5 pip install --no-build-isolation --no-binary=h5py h5py==3.16.0
ENV PATH=/opt/hdf5/bin:${PATH}

# install cdsapi for downloading the dataset
RUN pip install cdsapi>=0.7.2

# install zarr and data stuff. zarr is pinned to >=3 because makani is installed with
# --no-deps below, so the requirement in pyproject.toml is never enforced: an unpinned
# zarr is left at whatever the base image ships, and a v2 base silently breaks the zarr
# reader (v2 has no zarr.core.buffer, so importing dali_es_helper_2d fails outright).
RUN pip install more_itertools "zarr>=3" "xarray>=2025.1.0" gcsfs s3fs boto3

# moviepy for wandb video logging
RUN pip install "moviepy>=1.0.3"

# other python stuff
RUN pip install --upgrade "wandb>=0.13.7" ruamel.yaml "tqdm>=4.60.0"

# scoring tools
RUN pip install xskillscore properscoring

# Git dependencies are pinned to immutable commit SHAs rather than branches or tags,
# so that an upstream force-push or a repointed tag cannot change what lands in the
# image. SHAs below are the respective default branches as of 2026-08-04; bump them
# deliberately.

# weatherbench2
RUN pip install "git+https://github.com/google-research/weatherbench2.git@95c36d547b22abc2d191451a580b0b194fde67ef"

# some useful scripts from mlperf
RUN pip install --ignore-installed "git+https://github.com/NVIDIA/mlperf-common.git@c02c93c0414071400a10716d9cd5cb8f668b78d4"

# physicsnemo (SHA of tag v1.3.0)
RUN pip install git+https://github.com/NVIDIA/physicsnemo.git@14e0874847ffb56b35abf7708a1665e71605999c

# torch-harmonics (SHA of tag v0.9.3)
RUN pip install setuptools_scm
# torch-harmonics gates the CUDA build on this variable; without it BUILD_CUDA falls back
# to torch.cuda.is_available(), which is False inside docker build (no GPU on the builder),
# and TH silently compiles CPU-only kernels. The old name FORCE_CUDA_EXTENSION is dead.
ENV TORCH_HARMONICS_BUILD_CUDA_EXTENSION=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 9.0a 10.0a+PTX"
RUN cd /opt && git clone https://github.com/NVIDIA/torch-harmonics.git && \
    cd torch-harmonics && \
    git checkout 2028c7768e5ca3ad5a82e1d6bd8c304aa703754f && \
    pip install --no-build-isolation -e .

# copy source code
RUN mkdir -p /opt/makani
COPY config /opt/makani/config
COPY docker /opt/makani/docker
COPY data_process /opt/makani/data_process
COPY datasets /opt/makani/datasets
COPY makani /opt/makani/makani
COPY scripts /opt/makani/scripts
COPY tests /opt/makani/tests
COPY pyproject.toml /opt/makani/pyproject.toml
COPY README.md /opt/makani/README.md
RUN cd /opt/makani && pip install --no-deps -e .
