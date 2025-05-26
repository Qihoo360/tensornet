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
import tensornet as tn

from tensornet.core import gen_bn_table_ops
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer


class TNBatchNormalizationBase(Layer):
    """
    Reference: https://github.com/keras-team/keras/blob/v3.5.0/keras/src/layers/normalization/batch_normalization.py

    Args:
        center, scale, epsilon are the same as original batch normalization layer.
        momentum: same definition of original batch normalization, but it's for bn statistics, not original moving_mean, moving_var
        synchronized: Whether bn statistics(sum, squared sum, count) should be passed to other tensornet rank during training.
            If set to False, on train end, rank 0 will pull all statistics from other rank and calculate moving_mean and moving var, only once.
            If set to True, with 'sync_freq' argument, every 'sync_freq' batches, incremental bn statistics will be broadcast to all other ranks.
        sync_freq: frequency that bn statistics will be sent to other ranks(based on batches). Only should be used when 'synchronized' is True
        max_count: Threshold that to avoid bn statistics overflow. Note that: it's record number, not batch number. This is an empirical parameter that needs to be adjusted based on the size of the training data.
    """

    _USE_PCTR_DNN_BN = False

    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=1e-5,
        momentum=0.99,
        name=None,
        synchronized=False,
        sync_freq=1,
        max_count=100000,
        **kwargs,
    ):
        super(TNBatchNormalizationBase, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.momentum = momentum
        self.moments_axes = []
        self.apply_axis = []
        self.gamma, self.beta = None, None
        self.beta_initializer = initializers.get("zeros")
        self.gamma_initializer = initializers.get("ones")
        self.moving_mean_initializer = initializers.get("zeros")
        self.moving_variance_initializer = initializers.get("ones")
        self.local_count_initializer = initializers.get("zeros")
        self.local_sum_initializer = initializers.get("zeros")
        self.local_squared_num_initializer = initializers.get("zeros")
        self.beta_regularizer = regularizers.get(None)
        self.gamma_regularizer = regularizers.get(None)
        self.beta_constraint = constraints.get(None)
        self.gamma_constraint = constraints.get(None)
        self.synchronized = synchronized
        self.sync_freq = sync_freq
        self.batch_counter = tf.Variable(0, name="batch_counter")
        self.max_count = max_count

    def build(self, input_shape):
        input_rank = len(input_shape)
        self.moments_axes = list(range(input_rank - 1))
        self.apply_axis = input_shape[-1:]
        self.params_reshape = [1 for _ in range(1, input_rank - 1)] + [input_shape[-1]]

        if self.scale:
            self.gamma = self.add_weight(
                shape=self.apply_axis,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )

        if self.center:
            self.beta = self.add_weight(
                shape=self.apply_axis,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )

        self.moving_mean = self.add_weight(
            shape=self.apply_axis, name="moving_mean", initializer=self.moving_mean_initializer, trainable=False
        )

        self.moving_variance = self.add_weight(
            shape=self.apply_axis, name="moving_variance", initializer=self.moving_variance_initializer, trainable=False
        )

        self.local_count = self.add_weight(
            shape=self.apply_axis, name="local_count", initializer=self.local_count_initializer, trainable=False
        )

        self.local_sum = self.add_weight(
            shape=self.apply_axis, name="local_sum", initializer=self.local_sum_initializer, trainable=False
        )

        self.local_squared_sum = self.add_weight(
            shape=self.apply_axis,
            name="local_squared_sum",
            initializer=self.local_squared_num_initializer,
            trainable=False,
        )

        self.bn_table_handle = tn.core.create_bn_table(
            self.name, self.apply_axis[0], self.synchronized, self.momentum, self.max_count, self._USE_PCTR_DNN_BN
        )

    def call(self, inputs, training=None):
        @tf.function
        def _increment_and_check_count():
            self.batch_counter.assign_add(1)
            if tf.equal(self.batch_counter, self.sync_freq):
                self.bn_statistics_push(True)
                self.batch_counter.assign(0)
            else:
                self.bn_statistics_push(False)

        if training:
            local_count_sample = tf.ones_like(inputs, name="count")
            self.local_sum.assign(tf.reduce_sum(inputs, axis=self.moments_axes))
            self.local_squared_sum.assign(tf.reduce_sum(tf.square(inputs), axis=self.moments_axes))
            self.local_count.assign(tf.reduce_sum(local_count_sample, axis=self.moments_axes))
            if self.synchronized:
                _increment_and_check_count()
            else:
                self.bn_statistics_push(False)
            self.update_moments()

        mean = self.moving_mean
        var = self.moving_variance

        outputs = tf.nn.batch_normalization(
            x=inputs, mean=mean, variance=var, offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon
        )

        return outputs

    def update_moments(self):
        gen_bn_table_ops.update_moments(
            [self.moving_mean.handle, self.moving_variance.handle], table_handle=self.bn_table_handle
        )

    def bn_statistics_push(self, synchronized):
        gen_bn_table_ops.bn_statistics_push(
            [self.local_sum.handle, self.local_squared_sum.handle, self.local_count.handle],
            table_handle=self.bn_table_handle,
            synchronized=synchronized,
        )

    def bn_statistics_pull(self):
        # if sync_freq is greater than 1, force sync statistics once at the end of training
        if not self.synchronized or self.sync_freq > 1:
            self.batch_counter.assign(0)
            gen_bn_table_ops.bn_statistics_pull(
                [self.moving_mean.handle, self.moving_variance.handle], table_handle=self.bn_table_handle
            )

    def save_bn_table(self, filepath):
        return tn.core.save_bn_table(self.bn_table_handle, filepath)

    def load_bn_table(self, filepath):
        return tn.core.load_bn_table(self.bn_table_handle, filepath)


class TNBatchNormalization(TNBatchNormalizationBase):
    """
    Calculate incremental count, sum, squared sum. use (squared_sum / count - (sum / count).square) as var
    """


class PCTRDNNBatchNormalization(TNBatchNormalizationBase):
    """
    Calculate incremental count, sum. Calculate incremental (data - mean).sqrt() as var
    """

    _USE_PCTR_DNN_BN = True

    def call(self, inputs, training=None):
        @tf.function
        def _increment_and_check_count():
            self.batch_counter.assign_add(1)
            if tf.equal(self.batch_counter, self.sync_freq):
                self.bn_statistics_push(True)
                self.batch_counter.assign(0)
            else:
                self.bn_statistics_push(False)

        self.update_moments()
        mean = self.moving_mean
        var = self.moving_variance

        if training:
            local_count_sample = tf.ones_like(inputs, name="count")
            self.local_sum.assign(tf.reduce_sum(inputs, axis=self.moments_axes))
            self.local_squared_sum.assign(tf.reduce_sum(tf.square(inputs - self.moving_mean), axis=self.moments_axes))
            self.local_count.assign(tf.reduce_sum(local_count_sample, axis=self.moments_axes))
            if self.synchronized:
                _increment_and_check_count()
            else:
                self.bn_statistics_push(False)

        outputs = tf.nn.batch_normalization(
            x=inputs, mean=mean, variance=var, offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon
        )

        return outputs
