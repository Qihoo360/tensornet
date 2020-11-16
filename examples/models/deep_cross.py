# -*- coding: utf-8 -*-
import tensorflow as tf
import tensornet as tn
from common.config import Config as C
from common.layers import DeepCrossLayer
from common.feature_column import tn_category_columns_builder, embedding_columns_builder, create_emb_model  


def create_sub_model(embeddings, deep_hidden_units):
    emb_input_shapes = [emb.shape for emb in embeddings]
    inputs = [tf.keras.layers.Input(name="emb_{}".format(i), dtype="float32", shape=shape[1:])
              for i, shape in enumerate(emb_input_shapes)]

    cross, deep = None, None
    concacted = tf.keras.layers.Concatenate(name='deep_concact', axis=-1)(inputs)
    deep = concacted

    for i, unit in enumerate(deep_hidden_units):
        deep = tf.keras.layers.Dense(unit, activation='relu', name='dnn_{}'.format(i))(deep)

    cross_feature = DeepCrossLayer(num_layer=3)(concacted)

    both = tf.keras.layers.concatenate([deep, cross_feature], name='stack')
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)

    return tn.model.Model(inputs=inputs, outputs=output, name="sub_model")

def DCN(features, dnn_hidden_units=(128, 128)):
    columns_group = {}
    tn_category_columns = tn_category_columns_builder(features)
    columns_group["dcn"] = embedding_columns_builder(features, tn_category_columns, 8)
    inputs = {}
    for slot in features:
        inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)
    emb_model = create_emb_model(features, columns_group)
    [embs] = emb_model(inputs)
    sub_model = create_sub_model(embs, dnn_hidden_units)
    output = sub_model(embs)
    model = tn.model.Model(inputs=inputs, outputs=output, name="full_model")
    return model, sub_model
