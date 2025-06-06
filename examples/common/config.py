# -*- coding: utf-8 -*-


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


class Config(object):
    DATA_DIR = "./data/"
    FILE_MATCH_PATTERN = "tf-*"

    MODEL_DIR = "./model/"

    BATCH_SIZE = 32

    TRAIN_DAYS = ["2020-05-10", "2020-05-11"]

    SAVE_MODEL_INTERVAL_DAYS = 3

    DEEP_HIDDEN_UNITS = [512, 256, 256]

    LINEAR_SLOTS = ["1", "2", "3", "4"]
    DEEP_SLOTS = ["1", "2", "3", "4"]

    PREDICT_DT = None
    PREDICT_DUMP_PATH = "./predict"

    MODEL_TYPE = "WideDeep"  # supported models: WideDeep, DeepFM, DCN
