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

import tensorflow as tf

from tensorflow.keras.layers import Layer


class PositionMappingLayer(Layer):
    """
    One layer to return a one-hot tensor base on provided value collection. For each row, will return len(position_array) - 1 zeros,
    and a one located at its values' index of position_array
    """

    def __init__(self, position_array, **kwargs):
        super(PositionMappingLayer, self).__init__(**kwargs)

        assert len(position_array) > 0, "position array must be not empty"

        self._position_array = position_array
        self._position_number = len(position_array)
        self._position_slot_tensor = tf.constant(position_array, dtype=tf.int64)
        self._position_output_tensor = None

    def build(self, input_shapes):
        initial_value = tf.zeros([0, self._position_number], dtype=tf.float32)
        self._position_output_tensor = tf.Variable(
            initial_value, shape=[None, self._position_number], name="position_matrix", trainable=False
        )
        super(PositionMappingLayer, self).build(None)

    def call(self, inputs, training=None):
        if training:
            position_value = inputs.values
            position_value = tf.expand_dims(position_value, axis=1)
            is_present = tf.reduce_any(tf.equal(position_value, self._position_slot_tensor), axis=1)
            all_present = tf.reduce_all(is_present)
            assert_op = tf.debugging.assert_equal(
                all_present, True, message="inputs contains no existing value from provided position array"
            )
            with tf.control_dependencies([assert_op]):
                indices = tf.map_fn(lambda x: tf.where(self._position_slot_tensor == x), position_value, dtype=tf.int64)
                indices = tf.squeeze(indices, axis=-1)

            one_hot_tensor = tf.one_hot(indices, depth=self._position_number, dtype=tf.int64)
            one_hot_tensor = tf.reshape(one_hot_tensor, [-1, self._position_number])

            self._position_output_tensor.assign(tf.cast(one_hot_tensor, tf.float32))
        else:
            # if not training phase, position is unknown. Make it default slot one for all rows
            num_rows = tf.shape(inputs)[0]
            ones_column = tf.ones((num_rows, 1), dtype=tf.float32)
            zeros_columns = tf.zeros((num_rows, self._position_number - 1), dtype=tf.float32)
            self._position_output_tensor.assign(tf.concat([ones_column, zeros_columns], axis=1))

        return self._position_output_tensor
