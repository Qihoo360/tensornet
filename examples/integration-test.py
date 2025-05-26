#!/usr/bin/env python3

# Copyright 2020-2025 Qihoo Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

from common.config import Config as C
from main import main

if "OMPI_COMM_WORLD_RANK" not in os.environ or os.environ["OMPI_COMM_WORLD_RANK"] == "0":
    need_gen = False
    for d in C.TRAIN_DAYS:
        if not os.path.exists(f"{C.DATA_DIR}/{d}"):
            need_gen = True
            break

    if need_gen:
        import gen_example_data

        gen_example_data.main()

    # remove existing model
    shutil.rmtree(C.MODEL_DIR, ignore_errors=True)

main()
