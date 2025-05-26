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

import collections
import json

import tensorflow as tf
import tensornet as tn
from tensornet.core import gen_sparse_table_ops

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import serialization


class StateManagerImpl(fc.StateManager):
    """ """

    def __init__(self, layer, name, sparse_opt, dimension, trainable, target_columns=None, use_cvm=False):
        self._trainable = trainable
        self._layer = layer
        self.use_cvm = use_cvm
        if target_columns:
            self.use_cvm = True

        self.sparse_table_handle = tn.core.create_sparse_table(
            sparse_opt, name if name else "", dimension, self.use_cvm
        )
        self.pulled_mapping_values = {}

        if self._layer is not None and not hasattr(self._layer, "_resources"):
            self._layer._resources = []  # pylint: disable=protected-access

        self._target_columns = target_columns

        # be different with tensorflow StateManager implementation, we only support
        # store one variable for one unique feature column, which is name 'embedding_weights'
        self._cols_to_var_map = collections.defaultdict(lambda: None)
        self._var_to_cols_map = collections.defaultdict(lambda: None)

    def create_variable(
        self,
        feature_column,
        name,
        shape,
        dtype=None,
        trainable=True,
        use_resource=True,
        initializer=None,
        max_norm=None,
    ):
        column_name = feature_column.categorical_column.name
        if column_name in self._cols_to_var_map:
            raise ValueError("variable already created ", column_name)

        assert isinstance(feature_column, fc.EmbeddingColumn)

        var_name = column_name + "/" + name
        new_shape = shape

        if self._target_columns:
            new_shape = (shape[0], shape[-1] + 2)

        var = self._layer.add_weight(
            name=var_name,
            shape=new_shape,
            dtype=dtype,
            initializer=initializer,
            trainable=self._trainable and trainable,
            use_resource=use_resource,
        )

        self._cols_to_var_map[column_name] = var
        self._var_to_cols_map[var.ref()] = column_name

        return var

    def get_variable(self, feature_column, name):
        column_name = feature_column.categorical_column.name

        return self._cols_to_var_map[column_name]

    def pull(self, features):
        vars = []
        feature_values = []

        for column_name, sparse_feature in features.items():
            if self._target_columns and column_name in self._target_columns:
                continue
            if column_name not in self._cols_to_var_map and column_name != "label":
                raise ValueError("slot embedding variable not created, ", column_name)

            if not isinstance(sparse_feature, sparse_tensor_lib.SparseTensor):
                raise ValueError("sparse_feature input must be a SparseTensor.")

            vars.append(self._cols_to_var_map[column_name])
            feature_values.append(sparse_feature)

        pulled_mapping_values = gen_sparse_table_ops.sparse_table_pull(
            [var.handle for var in vars], [f.values for f in feature_values], table_handle=self.sparse_table_handle
        )
        assert len(pulled_mapping_values) == len(vars)

        for var, mapping_value in zip(vars, pulled_mapping_values):
            assert var.ref() in self._var_to_cols_map

            column_name = self._var_to_cols_map[var.ref()]

            self.pulled_mapping_values[column_name] = mapping_value

        return self.pulled_mapping_values

    def push(self, grads_and_vars, features):
        grads = []
        feature_values = []
        feature_labels = []

        for grad, var in grads_and_vars:
            if var.ref() not in self._var_to_cols_map:
                continue

            assert isinstance(grad, ops.IndexedSlices)

            column_name = self._var_to_cols_map[var.ref()]

            if column_name not in features:
                raise ValueError("column_name not found in features, ", column_name)

            sparse_feature = features[column_name]

            assert isinstance(sparse_feature, sparse_tensor_lib.SparseTensor)

            grads.append(grad.values)
            feature_values.append(sparse_feature.values)
            feature_label = self._layer.feature_clicks[column_name]
            feature_labels.append(tf.squeeze(feature_label.value()))

        # grads and feature_values must not empty
        assert grads and feature_values

        return gen_sparse_table_ops.sparse_table_push(
            feature_values, grads, feature_labels, table_handle=self.sparse_table_handle
        )

    def get_feature_mapping_values(self, column_name):
        return self.pulled_mapping_values[column_name]

    def save_sparse_table(self, filepath, mode):
        return tn.core.save_sparse_table(self.sparse_table_handle, filepath, mode)

    def load_sparse_table(self, filepath, mode):
        return tn.core.load_sparse_table(self.sparse_table_handle, filepath, mode)

    def show_decay(self, delta_days=0):
        return tn.core.show_decay(self.sparse_table_handle, delta_days)


class EmbeddingFeatures(Layer):
    """ """

    def __init__(
        self, feature_columns, sparse_opt, trainable=True, name=None, is_concat=False, target_columns=None, **kwargs
    ):
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
            target_columns: labels used for cvm plugin. labels will be counted as a feature,
                calculate total_count, label_count (usually used for ctr, counting show/click number)
                embedding output will include embedding size float + total_count + label_count / total_count

        """
        super(EmbeddingFeatures, self).__init__(name=name, trainable=trainable, dynamic=False, **kwargs)

        assert len(feature_columns) != 0, "feature_columns must not empty"

        dim = feature_columns[0].dimension
        for feature_column in feature_columns:
            assert feature_column.dimension == dim, (
                "currently we only support feature_columns with same dimension in EmbeddingFeatures"
            )

        self._sparse_opt = sparse_opt
        self._feature_columns = feature_columns
        self.sparse_pulling_features = None
        self.is_concat = is_concat
        self._target_columns = target_columns
        if target_columns and len(target_columns) > 1:
            raise ValueError("For now cvm plugin only support one column, Given: {}".format(target_columns))
        self._state_manager = StateManagerImpl(self, name, sparse_opt, dim, self.trainable, target_columns)  # pylint: disable=protected-access
        self.sparse_target_features = None
        self.feature_clicks = {}

        for column in self._feature_columns:
            if not isinstance(column, fc.EmbeddingColumn):
                raise ValueError("Items of feature_columns must be a {}. Given: {}".format(fc.EmbeddingColumn, column))

    def build(self, input_shapes):
        for column in self._feature_columns:
            initial_value = tf.zeros([0, 1], dtype=tf.int64)
            var = tf.Variable(
                initial_value, shape=[None, 1], name="label_count" + column.categorical_column.name, trainable=False
            )
            self.feature_clicks[column.categorical_column.name] = var
            with ops.name_scope(column.name):
                column.create_state(self._state_manager)

        super(EmbeddingFeatures, self).build(None)

    def call(self, features, cols_to_output_tensors=None, training=None):
        if not isinstance(features, dict):
            raise ValueError("We expected a dictionary here. Instead we got: ", features)
        tn.core.set_sparse_init_mode(self._sparse_opt, tf_utils.constant_value(training))
        using_features = self.filter_not_used_features(features)
        transformation_cache = fc.FeatureTransformationCache(using_features)

        self.sparse_pulling_features, self.sparse_target_features = self.get_sparse_pulling_feature(using_features)

        if self._target_columns:
            labels = features[self._target_columns[0]]

        for tensor_index, sparse_tensor_key in enumerate(self.sparse_pulling_features):
            sparse_tensor = features[sparse_tensor_key]
            indices_for_gather = tf.expand_dims(sparse_tensor.indices[:, 0], axis=-1)
            if self._target_columns:
                feature_values = tf.gather_nd(labels, indices_for_gather)
                self.feature_clicks[str(sparse_tensor_key)].assign(feature_values)
            else:
                num_elements = tf.shape(indices_for_gather)[0]
                zeros_tensor = tf.zeros((num_elements, 1), dtype=tf.int64)
                self.feature_clicks[str(sparse_tensor_key)].assign(zeros_tensor)

        pulled_mapping_values = self._state_manager.pull(self.sparse_pulling_features)

        output_tensors = []
        for column in self._feature_columns:
            if column.categorical_column.name not in pulled_mapping_values:
                raise ValueError("column not found in pulled_mapping_values")

            mapping_value = pulled_mapping_values[column.categorical_column.name]
            with ops.control_dependencies([mapping_value]):
                tensor = column.get_dense_tensor(transformation_cache, self._state_manager)

            processed_tensors = self._process_dense_tensor(column, tensor)

            if self._target_columns:
                tensor_shape = tf.shape(processed_tensors)

                num_features = tensor_shape[1]
                mask = tf.concat(
                    [tf.ones([1, num_features - 2], dtype=tensor.dtype), tf.zeros([1, 2], dtype=tensor.dtype)], axis=1
                )
                new_tensor = processed_tensors * mask + tf.stop_gradient(processed_tensors * (1 - mask))
            else:
                new_tensor = processed_tensors

            if cols_to_output_tensors is not None:
                cols_to_output_tensors[column] = new_tensor

            output_tensors.append(new_tensor)

        if self.is_concat:
            return self._verify_and_concat_tensors(output_tensors)
        else:
            return output_tensors

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
        target_features = {}
        for column_name, feature in features.items():
            if not isinstance(feature, sparse_tensor_lib.SparseTensor):
                continue

            if self._target_columns and column_name in self._target_columns:
                target_features[column_name] = feature
                continue

            new_features[column_name] = feature

        return new_features, target_features

    def save_sparse_table(self, filepath, mode):
        return self._state_manager.save_sparse_table(filepath, mode)

    def load_sparse_table(self, filepath, mode):
        return self._state_manager.load_sparse_table(filepath, mode)

    def show_decay(self, delta_days=0):
        return self._state_manager.show_decay(delta_days)

    def _target_shape(self, input_shape, total_elements):
        return (input_shape[0], total_elements)

    def compute_output_shape(self, input_shape):
        total_elements = 0
        for column in self._feature_columns:
            total_elements += column.variable_shape.num_elements()
            if self._target_columns:
                total_elements += 2
        return self._target_shape(input_shape, total_elements)

    def _process_dense_tensor(self, column, tensor):
        """ """
        num_elements = column.variable_shape.num_elements()
        if self._target_columns:
            num_elements += 2
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
            EmbeddingFeatures, self
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
        metadata = json.loads(super(EmbeddingFeatures, self)._tracking_metadata)
        metadata["_is_feature_layer"] = True
        return json.dumps(metadata, default=serialization.get_json_type)
