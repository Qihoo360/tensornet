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


def tn_category_columns_builder(features):
    """Builds tensornet category feature columns."""
    columns = {}
    for slot in set(features):
        columns[slot] = tn.feature_column.category_column(key=slot)
    return columns


def embedding_columns_builder(features, tn_category_columns, embedding_size=8):
    embedding_columns = []
    for slot in features:
        feature_column = tf.feature_column.embedding_column(tn_category_columns[slot], dimension=embedding_size)
        embedding_columns.append(feature_column)

    return embedding_columns


def create_emb_model(features, columns_group, suffix="_input"):
    model_output = []
    inputs = {}
    for slot in features:
        inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)

    inputs["label"] = tf.keras.layers.Input(name="label", shape=(None,), dtype="int64", sparse=False)

    sparse_opt = tn.core.AdaGrad(learning_rate=0.01, initial_g2sum=0.1, initial_scale=0.1)

    for column_group_name in columns_group.keys():
        embs = tn.layers.EmbeddingFeatures(
            columns_group[column_group_name], sparse_opt, name=column_group_name + suffix, target_columns=["label"]
        )(inputs)
        # name=column_group_name + suffix)(inputs)
        model_output.append(embs)

    emb_model = tn.model.Model(inputs=inputs, outputs=model_output, name="emb_model")

    return emb_model
