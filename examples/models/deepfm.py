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
from common.layers import FMLayer
from common.feature_column import tn_category_columns_builder, embedding_columns_builder, create_emb_model


def create_sub_model(linear_embs, deep_embs, deep_hidden_units):
    linear_emb_input_shapes = [emb.shape for emb in linear_embs]
    deep_emb_input_shapes = [emb.shape for emb in deep_embs]

    linear_inputs = [
        tf.keras.layers.Input(name="linear_emb_{}".format(i), dtype="float32", shape=shape[1:])
        for i, shape in enumerate(linear_emb_input_shapes)
    ]
    deep_inputs = [
        tf.keras.layers.Input(name="deep_emb_{}".format(i), dtype="float32", shape=shape[1:])
        for i, shape in enumerate(deep_emb_input_shapes)
    ]

    linear, fm, deep = None, None, None

    if linear_inputs:
        linear = tf.keras.layers.Concatenate(name="linear_concact", axis=-1)(linear_inputs)

    if deep_inputs:
        deep = tf.keras.layers.Concatenate(name="deep_concact", axis=-1)(deep_inputs)

        for i, unit in enumerate(deep_hidden_units):
            deep = tf.keras.layers.Dense(unit, activation="relu", name="dnn_{}".format(i))(deep)

        fm_inputs = [tf.expand_dims(inputs, axis=1) for inputs in deep_inputs]
        concated_embeds_value = tf.keras.layers.Concatenate(name="fm_concact", axis=1)(fm_inputs)
        fm = FMLayer()(concated_embeds_value)

    if linear_inputs and not deep_inputs:
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(linear)
    elif deep_inputs and not linear_inputs:
        both = tf.keras.layers.concatenate([fm, deep], name="deep_fm")
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(both)
    else:
        both = tf.keras.layers.concatenate([linear, fm, deep], name="deep_fm")
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(both)

    return tn.model.Model(inputs=[linear_inputs, deep_inputs], outputs=output, name="sub_model")


def DeepFM(linear_features, dnn_features, dnn_hidden_units=(128, 128)):
    features = set(linear_features + dnn_features)
    columns_group = {}
    tn_category_columns = tn_category_columns_builder(features)
    columns_group["linear"] = embedding_columns_builder(linear_features, tn_category_columns, 1)
    columns_group["deep"] = embedding_columns_builder(dnn_features, tn_category_columns, 8)
    inputs = {}
    for slot in features:
        inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)
    emb_model = create_emb_model(features, columns_group)
    linear_embs, deep_embs = emb_model(inputs)
    sub_model = create_sub_model(linear_embs, deep_embs, dnn_hidden_units)
    output = sub_model([linear_embs, deep_embs])
    model = tn.model.Model(inputs=inputs, outputs=output, name="full_model")
    return model, sub_model
