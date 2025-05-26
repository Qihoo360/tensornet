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

from tensornet.core import gen_balance_dataset_ops

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops


def list_files(datapath, days, match_pattern):
    with ops.name_scope("list_files"):
        file_pattern = []
        for day in days:
            file_pattern.append(os.path.join(datapath, day, match_pattern))

        file_pattern = ops.convert_to_tensor(file_pattern, dtype=dtypes.string, name="file_pattern")
        matching_files = gen_io_ops.matching_files(file_pattern)

        # Raise an exception if `file_pattern` does not match any files.
        condition = math_ops.greater(array_ops.shape(matching_files)[0], 0, name="match_not_empty")

        message = math_ops.add(
            "No files matched pattern: ", string_ops.reduce_join(file_pattern, separator=", "), name="message"
        )

        assert_not_empty = control_flow_ops.Assert(condition, [message], summarize=1, name="assert_not_empty")
        with ops.control_dependencies([assert_not_empty]):
            matching_files = array_ops.identity(matching_files)

        dataset = dataset_ops.Dataset.from_tensor_slices(matching_files)

        return dataset


class BalanceDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that balance input data between other cocurrent ops"""

    def __init__(self, input_dataset):
        self._input_dataset = input_dataset
        self._structure = input_dataset.element_spec
        variant_tensor = gen_balance_dataset_ops.balance_dataset(input_dataset._variant_tensor, **self._flat_structure)
        super(BalanceDataset, self).__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._structure
