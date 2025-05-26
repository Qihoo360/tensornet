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

# Author: zhangyansheng <zhangyansheng@360.cn>
# Copyright(C) 360.cn, all rights reserved.
# Date: 2020/02/10

import collections

from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import parsing_ops


class CategoryColumn(fc.CategoricalColumn, collections.namedtuple("CategoryColumn", ("key", "bucket_size"))):
    def _is_v2_column(self):
        return True

    @property
    def name(self):
        return self.key

    @property
    def num_buckets(self):
        """Returns number of buckets in this sparse feature."""
        return self.bucket_size

    def transform_feature(self, transformation_cache, state_manager):
        """mapping the values in the feature_column."""
        input_tensor = transformation_cache.get(self.key, state_manager)

        if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
            raise ValueError("CategoryColumn input must be a SparseTensor.")

        sparse_id_values = state_manager.get_feature_mapping_values(self.name)

        return sparse_tensor_lib.SparseTensor(input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

    def parse_example_spec(self):
        return {self.key: parsing_ops.VarLenFeature(dtypes.string)}

    @property
    def parents(self):
        return [self.key]

    def get_sparse_tensors(self, transformation_cache, state_manager):
        return fc.CategoricalColumn.IdWeightPair(transformation_cache.get(self, state_manager), None)

    def _get_config(self):
        config = dict(zip(self._fields, self))
        config["dtype"] = dtypes.string.name
        return config

    @classmethod
    def _from_config(cls, config, custom_objects=None, columns_by_name=None):
        fc._check_config_keys(config, cls._fields)
        kwargs = fc._standardize_and_copy_config(config)
        kwargs["dtype"] = dtypes.as_dtype(config["dtype"])
        return cls(**kwargs)


def category_column(key, bucket_size=1024):
    """Represents sparse feature where ids are set by hashing."""
    return CategoryColumn(key, bucket_size)
