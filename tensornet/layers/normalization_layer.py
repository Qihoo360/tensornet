# -*- coding: utf-8 -*-
import tensorflow as tf
import tensornet as tn

from tensornet.core import gen_bn_table_ops
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import variable_scope, array_ops


class BatchNormalization(Layer):
    def __init__(self, center=True, scale=True, epsilon=1e-5, decay=0.99, name=None, synchronized=False, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.decay = decay
        self.moments_axes = []
        self.apply_axis = []
        self.gamma, self.beta = None, None
        self.beta_initializer = initializers.get('zeros')
        self.gamma_initializer = initializers.get('ones')
        self.moving_mean_initializer = initializers.get('zeros')
        self.moving_variance_initializer = initializers.get('ones')
        self.local_count_initializer = initializers.get('zeros')
        self.local_sum_initializer = initializers.get('zeros')
        self.local_squared_num_initializer = initializers.get('zeros')
        self.beta_regularizer = regularizers.get(None)
        self.gamma_regularizer = regularizers.get(None)
        self.beta_constraint = constraints.get(None)
        self.gamma_constraint = constraints.get(None)
        self.synchronized = synchronized


    def build(self, input_shape):
        input_rank = len(input_shape)
        self.moments_axes = list(range(input_rank - 1))
        self.apply_axis = input_shape[-1:]
        self.params_reshape = [1 for _ in range(
            1, input_rank - 1)] + [input_shape[-1]]

        if self.scale:
            self.gamma = self.add_weight(shape=self.apply_axis, name='gamma', initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer, constraint=self.gamma_constraint)

        if self.center:
            self.beta = self.add_weight(shape=self.apply_axis, name='beta', initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer, constraint=self.beta_constraint)

        self.moving_mean = self.add_weight(
            shape=self.apply_axis,
            name="moving_mean",
            initializer=self.moving_mean_initializer,
            trainable=False)

        self.moving_variance = self.add_weight(
            shape=self.apply_axis,
            name="moving_variance",
            initializer=self.moving_variance_initializer,
            trainable=False)
        
        self.local_count = self.add_weight(
            shape=self.apply_axis,
            name="local_count",
            initializer=self.local_count_initializer,
            trainable=False
            )

        self.local_sum = self.add_weight(
            shape=self.apply_axis,
            name="local_sum",
            initializer=self.local_sum_initializer,
            trainable=False)
 
        self.local_squared_sum = self.add_weight(
            shape=self.apply_axis,
            name="local_squared_sum",
            initializer=self.local_squared_num_initializer,
            trainable=False)

        self.bn_table_handle = tn.core.create_bn_table(self.name, self.apply_axis[0])



    def call(self, inputs, training=None):
        
        if training:
            local_count_sample = tf.ones_like(inputs, name="count")
            self.local_sum.assign(tf.reduce_sum(inputs, axis=self.moments_axes))
            self.local_squared_sum.assign(tf.reduce_sum(tf.square(inputs), axis=self.moments_axes))
            self.local_count.assign(tf.reduce_sum(local_count_sample, axis=self.moments_axes))
            self.bn_statistics_push()
            self.update_moments()
        
        mean = self.moving_mean
        var = self.moving_variance

        outputs = tf.nn.batch_normalization(x=inputs, mean=mean, variance=var, offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon)

        return outputs

    def update_moments(self):
        gen_bn_table_ops.update_moments([self.moving_mean.handle, self.moving_variance.handle], table_handle=self.bn_table_handle)

    def bn_statistics_push(self):
        gen_bn_table_ops.bn_statistics_push([self.local_sum.handle, self.local_squared_sum.handle, self.local_count.handle], table_handle=self.bn_table_handle, synchronized=self.synchronized)

    def bn_statistics_pull(self):
        if not self.synchronized:
            gen_bn_table_ops.bn_statistics_pull([self.moving_mean.handle, self.moving_variance.handle], table_handle=self.bn_table_handle)
