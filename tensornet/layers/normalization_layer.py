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
    def __init__(self, center=True, scale=True, epsilon=1e-5, decay=0.999, name=None, **kwargs):
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
        self.moving_count_initializer = initializers.get('zeros')
        self.beta_regularizer = regularizers.get(None)
        self.gamma_regularizer = regularizers.get(None)
        self.beta_constraint = constraints.get(None)
        self.gamma_constraint = constraints.get(None)


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
        
        self.moving_count = self.add_weight(
            shape=[],
            name="moving_count",
            initializer=self.moving_count_initializer,
            trainable=False
            )

        self.bn_table_handle = tn.core.create_bn_table(self.name, self.apply_axis[0])



    def call(self, inputs, training=None):
        
        if training:
            mean, var = tf.nn.moments(inputs, axes=self.moments_axes)
            self.moving_mean.assign(self.moving_mean * self.decay + mean * (1.0 - self.decay))
            self.moving_variance.assign(self.moving_variance * self.decay + var * (1.0 - self.decay))
            self.moving_count.assign(self.moving_count + 1)
        else:
            mean = self.moving_mean
            var = self.moving_variance

        outputs = tf.nn.batch_normalization(x=inputs, mean=mean, variance=var, offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon)

        return outputs

    def set_bn_vars(self):
        gen_bn_table_ops.set_bn_vars([self.moving_mean, self.moving_variance, self.moving_count], table_handle=self.bn_table_handle)
        tn.core.barrier()

    def bn_vars_pull(self):
        gen_bn_table_ops.bn_vars_pull([self.moving_mean, self.moving_variance, self.moving_count], table_handle=self.bn_table_handle)
        tn.core.barrier()

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        # Only add TensorFlow-specific parameters if they are set, so as to preserve
        # model compatibility with external Keras.
        if self.renorm:
            config['renorm'] = True
            config['renorm_clipping'] = self.renorm_clipping
            config['renorm_momentum'] = self.renorm_momentum
        if self.virtual_batch_size is not None:
            config['virtual_batch_size'] = self.virtual_batch_size
        # Note: adjustment is not serializable.
        if self.adjustment is not None:
            logging.warning('The `adjustment` function of this `BatchNormalization` '
                      'layer cannot be serialized and has been omitted from '
                      'the layer config. It will not be included when '
                      're-creating the layer from the saved config.')
        base_config = super(BatchNormalizationBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
