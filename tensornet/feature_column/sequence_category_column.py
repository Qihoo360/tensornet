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

# Author: zhangyansheng <tangyanlin@360.cn>
# Copyright(C) 360.cn, all rights reserved.

import collections

from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


class SequenceCategoryColumn(
    fc.SequenceCategoricalColumn, collections.namedtuple("SequenceCategoricalColumn", ("categorical_column"))
):
    def _is_v2_column(self):
        return True

    @property
    def name(self):
        return self.categorical_column.key

    @property
    def num_buckets(self):
        """Returns number of buckets in this sparse feature."""
        return self.categorical_column.bucket_size

    def transform_feature(self, transformation_cache, state_manager):
        """mapping the values in the feature_column."""
        input_tensor = transformation_cache.get(self.categorical_column.key, state_manager)

        if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
            raise ValueError("CategoryColumn input must be a SparseTensor.")

        sparse_id_values = state_manager.get_feature_mapping_values(self.categorical_column.name)

        return sparse_tensor_lib.SparseTensor(input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

    def parse_example_spec(self):
        return {self.categorical_column.key: parsing_ops.VarLenFeature(dtypes.string)}

    @property
    def parents(self):
        return [self.categorical_column.key]

    def get_sparse_tensors(self, transformation_cache, state_manager):
        sparse_tensors = fc.CategoricalColumn.IdWeightPair(transformation_cache.get(self, state_manager), None)
        return self._get_sparse_tensors_helper(sparse_tensors)

    def get_config(self):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import serialize_feature_column  # pylint: disable=g-import-not-at-top

        config = dict(zip(self._fields, self))
        config["categorical_column"] = serialize_feature_column(self.categorical_column)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import deserialize_feature_column  # pylint: disable=g-import-not-at-top

        fc._check_config_keys(config, cls._fields)
        kwargs = fc._standardize_and_copy_config(config)
        kwargs["categorical_column"] = deserialize_feature_column(
            config["categorical_column"], custom_objects, columns_by_name
        )
        return cls(**kwargs)

    def _get_sparse_tensors_helper(self, sparse_tensors):
        id_tensor = sparse_tensors.id_tensor
        weight_tensor = sparse_tensors.weight_tensor
        # Expands third dimension, if necessary so that embeddings are not
        # combined during embedding lookup. If the tensor is already 3D, leave
        # as-is.
        shape = array_ops.shape(id_tensor)
        target_shape = [shape[0], shape[1], math_ops.reduce_prod(shape[2:])]
        id_tensor = sparse_ops.sparse_reshape(id_tensor, target_shape)
        if weight_tensor is not None:
            weight_tensor = sparse_ops.sparse_reshape(weight_tensor, target_shape)
        return fc.CategoricalColumn.IdWeightPair(id_tensor, weight_tensor)


def sequence_category_column(categorical_column):
    """Represents sparse feature where ids are set by hashing."""
    return SequenceCategoryColumn(categorical_column)
