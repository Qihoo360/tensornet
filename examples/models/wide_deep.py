# -*- coding: utf-8 -*-
import tensorflow as tf
import tensornet as tn
from common.config import Config as C
from common.layers import FMLayer
from common.feature_column import tn_category_columns_builder, embedding_columns_builder, create_emb_model  

def create_sub_model(linear_embs, deep_embs, deep_hidden_units):
    linear_emb_input_shapes = [emb.shape for emb in linear_embs]
    deep_emb_input_shapes = [emb.shape for emb in deep_embs]

    linear_inputs = [tf.keras.layers.Input(name="linear_emb_{}".format(i), dtype="float32", shape=shape[1:])
                    for i, shape in enumerate(linear_emb_input_shapes)]
    deep_inputs = [tf.keras.layers.Input(name="deep_emb_{}".format(i), dtype="float32", shape=shape[1:])
                    for i, shape in enumerate(deep_emb_input_shapes)]

    linear, deep = None, None

    if linear_inputs:
        linear = tf.keras.layers.Concatenate(name='linear_concact', axis=-1)(linear_inputs)

    if deep_inputs:
        deep = tf.keras.layers.Concatenate(name='deep_concact', axis=-1)(deep_inputs)

        for i, unit in enumerate(C.DEEP_HIDDEN_UNITS):
            deep = tf.keras.layers.Dense(unit, activation='relu', name='dnn_{}'.format(i))(deep)

#    if linear_inputs and not deep_inputs:
#        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(linear)
#    elif deep_inputs and not linear_inputs:
#        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(deep)
#    else:
    both = tf.keras.layers.concatenate([deep, linear], name='both')
    both = tn.layers.TNBatchNormalization(synchronized=True, sync_freq=4, max_count=1000000)(both)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)

    return tn.model.Model(inputs=[linear_inputs, deep_inputs], outputs=output, name="sub_model")


def WideDeep(linear_features, dnn_features, dnn_hidden_units=(128, 128)):
    features = set(linear_features + dnn_features) 
    columns_group = {}
    tn_category_columns = tn_category_columns_builder(features)
    columns_group["linear"] = embedding_columns_builder(linear_features, tn_category_columns, 1)
    columns_group["deep"] = embedding_columns_builder(dnn_features, tn_category_columns, 8)
    inputs = {}
    for slot in features:
        inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)
    inputs['label'] = tf.keras.layers.Input(name="label", shape=(None,), dtype="int64", sparse=False)
    emb_model = create_emb_model(features, columns_group)
    linear_embs, deep_embs = emb_model(inputs)
    sub_model = create_sub_model(linear_embs, deep_embs, dnn_hidden_units)
    output = sub_model([linear_embs, deep_embs])
    model = tn.model.Model(inputs=inputs, outputs=output, name="full_model")
    return model, sub_model
