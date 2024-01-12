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

install:
	pip install --upgrade pip && pip install -e .

wheel:
	pip install --upgrade pip && pip wheel . --no-deps

test:
	coverage run --source ./makani -m pytest tests
	coverage report
	coverage xml

format:
	python3 -m black --line-length=180 ./makani

.PHONY: install wheel test
