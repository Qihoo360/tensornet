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
# Date: 2020/01/29

import json


from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.util import serialization
from .embedding_features import StateManagerImpl


class SequenceEmbeddingFeatures(Layer):
    """ """

    def __init__(self, feature_columns, sparse_opt, trainable=True, name=None, **kwargs):
        """create a embedding feature layer.
        when this layer is been called, all the embedding data of `feature_columns` will be
        pulled from ps server and return as a tensor list.

        Args:
            feature_columns: An iterable containing the FeatureColumns to use as
                inputs to your model. All items should be instances of classes derived
                from `embedding_column`.
            sparse_opt: the sparse optimizer of this embedding layer.
            trainable:  Boolean, whether the layer's variables will be updated via
                gradient descent during training.
            name: Name to give to the EmbeddingFeatures.
            is_concat: when this parameter is True, all the tensor of pulled will be concat
                with axis=-1 and returned.

        """
        super(SequenceEmbeddingFeatures, self).__init__(name=name, trainable=trainable, dynamic=False, **kwargs)

        assert len(feature_columns) != 0, "feature_columns must not empty"

        dim = feature_columns[0].dimension
        for feature_column in feature_columns:
            assert feature_column.dimension == dim, (
                "currently we only support feature_columns with same dimension in EmbeddingFeatures"
            )

        self._feature_columns = feature_columns
        self._state_manager = StateManagerImpl(self, name if name else "", sparse_opt, dim, self.trainable)  # pylint: disable=protected-access
        self.sparse_pulling_features = None

        for column in self._feature_columns:
            if not isinstance(column, fc.EmbeddingColumn):
                raise ValueError("Items of feature_columns must be a {}. Given: {}".format(fc.EmbeddingColumn, column))

    def build(self, input_shapes):
        for column in self._feature_columns:
            with ops.name_scope(column.name):
                column.create_state(self._state_manager)

        super(SequenceEmbeddingFeatures, self).build(None)

    def call(self, features, cols_to_output_tensors=None):
        if not isinstance(features, dict):
            raise ValueError("We expected a dictionary here. Instead we got: ", features)

        using_features = self.filter_not_used_features(features)
        transformation_cache = fc.FeatureTransformationCache(using_features)

        self.sparse_pulling_features = self.get_sparse_pulling_feature(using_features)

        pulled_mapping_values = self._state_manager.pull(self.sparse_pulling_features)

        output_tensors = []
        sequence_lengths = []
        for column in self._feature_columns:
            if column.categorical_column.name not in pulled_mapping_values:
                raise ValueError("column not found in pulled_mapping_values")

            mapping_value = pulled_mapping_values[column.categorical_column.name]
            with ops.control_dependencies([mapping_value]):
                tensor, sequence_length = column.get_sequence_dense_tensor(transformation_cache, self._state_manager)

            processed_tensors = self._process_dense_tensor(column, tensor)

            if cols_to_output_tensors is not None:
                cols_to_output_tensors[column] = processed_tensors

            output_tensors.append(processed_tensors)
            sequence_lengths.append(sequence_length)

        fc._verify_static_batch_size_equality(sequence_lengths, self._feature_columns)
        sequence_length = _assert_all_equal_and_return(sequence_lengths)
        return self._verify_and_concat_tensors(output_tensors), sequence_length

    def backwards(self, grads_and_vars):
        assert self.sparse_pulling_features

        return self._state_manager.push(grads_and_vars, self.sparse_pulling_features)

    def filter_not_used_features(self, features):
        new_features = {}
        for column in self._feature_columns:
            feature_key = column.categorical_column.name
            new_features[feature_key] = features[feature_key]
        return new_features

    def get_sparse_pulling_feature(self, features):
        new_features = {}
        for column_name, feature in features.items():
            if not isinstance(feature, sparse_tensor_lib.SparseTensor):
                continue

            new_features[column_name] = feature

        return new_features

    def save_sparse_table(self, filepath, mode):
        return self._state_manager.save_sparse_table(filepath, mode)

    def load_sparse_table(self, filepath, mode):
        return self._state_manager.load_sparse_table(filepath, mode)

    def show_decay(self, delta_days):
        return self._state_manager.show_decay(delta_days)

    def _target_shape(self, input_shape, total_elements):
        return (input_shape[0], input_shape[1], total_elements)

    def compute_output_shape(self, input_shape):
        total_elements = 0
        for column in self._feature_columns:
            total_elements += column.variable_shape.num_elements()
        return self._target_shape(input_shape, total_elements)

    def _process_dense_tensor(self, column, tensor):
        """ """
        num_elements = column.variable_shape.num_elements()
        target_shape = self._target_shape(array_ops.shape(tensor), num_elements)
        return array_ops.reshape(tensor, shape=target_shape)

    def _verify_and_concat_tensors(self, output_tensors):
        """Verifies and concatenates the dense output of several columns."""
        fc._verify_static_batch_size_equality(output_tensors, self._feature_columns)
        return array_ops.concat(output_tensors, -1)

    def get_config(self):
        # Import here to avoid circular imports.
        from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top

        column_configs = serialization.serialize_feature_columns(self._feature_columns)
        config = {"feature_columns": column_configs}

        base_config = super(  # pylint: disable=bad-super-call
            SequenceEmbeddingFeatures, self
        ).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Import here to avoid circular imports.
        from tensorflow.python.feature_column import serialization  # pylint: disable=g-import-not-at-top

        config_cp = config.copy()
        config_cp["feature_columns"] = serialization.deserialize_feature_columns(
            config["feature_columns"], custom_objects=custom_objects
        )

        return cls(**config_cp)

    @property
    def _is_feature_layer(self):
        return True

    @property
    def _tracking_metadata(self):
        """String stored in metadata field in the SavedModel proto.

        Returns:
          A serialized JSON storing information necessary for recreating this layer.
        """
        metadata = json.loads(super(SequenceEmbeddingFeatures, self)._tracking_metadata)
        metadata["_is_feature_layer"] = True
        return json.dumps(metadata, default=serialization.get_json_type)


def _assert_all_equal_and_return(tensors, name=None):
    """Asserts that all tensors are equal and returns the first one."""
    with ops.name_scope(name, "assert_all_equal", values=tensors):
        if len(tensors) == 1:
            return tensors[0]
        assert_equal_ops = []
        for t in tensors[1:]:
            assert_equal_ops.append(check_ops.assert_equal(tensors[0], t))
        with ops.control_dependencies(assert_equal_ops):
            return array_ops.identity(tensors[0])
