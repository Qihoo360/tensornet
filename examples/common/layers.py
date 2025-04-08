# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.keras.layers import Layer


class DeepCrossLayer(Layer):
    """
    Wang, Ruoxi, et al. "Deep & cross network for ad click predictions." Proceedings of the ADKDD'17. 2017. 1-7.
    input_shape = [batch_size, fields*emb]
    """

    def __init__(self, num_layer=3, **kwargs):
        self.num_layer = num_layer
        super(DeepCrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = [batch_size, fields*emb]
        self.input_dim = input_shape[1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(
                self.add_weight(
                    shape=[self.input_dim, 1], initializer="glorot_uniform", name="w_" + str(i), trainable=True
                )
            )
            self.bias.append(
                self.add_weight(
                    shape=[
                        self.input_dim,
                    ],
                    initializer="zeros",
                    name="b_" + str(i),
                    trainable=True,
                )
            )

    def call(self, inputs):
        for i in range(self.num_layer):
            if i == 0:
                cross = inputs * tf.matmul(inputs, self.W[i]) + self.bias[i] + inputs
            else:
                cross = inputs * tf.matmul(cross, self.W[i]) + self.bias[i] + cross
        return cross

    def get_config(self):
        config = super(DeepCrossLayer, self).get_config()
        config.update(
            {
                "num_layer": self.num_layer,
            }
        )

        return config


class FMLayer(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.
    input shape = (batch_size,field_size,embedding_size)
    output shape = (batch_size, 1)
    """

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions"
                % (len(input_shape))
            )

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if backend.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (backend.ndim(inputs)))

        square_of_sum = tf.square(tf.math.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.math.reduce_sum(inputs * inputs, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        fm = 0.5 * tf.math.reduce_sum(cross_term, axis=-1, keepdims=False)

        return fm

    def compute_output_shape(self, input_shape):
        return (None, 1)
