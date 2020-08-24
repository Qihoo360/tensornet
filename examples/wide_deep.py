# -*- coding: utf-8 -*-
import os
from datetime import datetime

import tensorflow as tf
import tensornet as tn
import numpy as np


class Config(object):
    DATA_DIR = './data/'
    FILE_MATCH_PATTERN = "tf-*"

    MODEL_DIR = "./model/"

    BATCH_SIZE = 32

    TRAIN_DAYS = ['2020-05-10', '2020-05-11']

    SAVE_MODEL_INTERVAL_DAYS = 3

    DEEP_HIDDEN_UNITS = [512, 256, 256]

    WIDE_SLOTS = [ "1","2","3","4"]
    DEEP_SLOTS = [ "1","2","3","4"]

    PREDICT_DT = None
    PREDICT_DUMP_PATH = "./predict"


C = Config

def columns_builder():
    """Builds a set of wide and deep feature columns."""

    columns = {}
    for slot in set(C.WIDE_SLOTS + C.DEEP_SLOTS):
        columns[slot] = tn.feature_column.category_column(key=slot)

    wide_columns = []
    for slot in C.WIDE_SLOTS:
        feature_column = tf.feature_column.embedding_column(columns[slot], dimension=1)

        wide_columns.append(feature_column)

    deep_columns = []
    for slot in C.DEEP_SLOTS:
        feature_column = tf.feature_column.embedding_column(columns[slot], dimension=8)
        deep_columns.append(feature_column)

    return wide_columns, deep_columns


def parse_line_batch(example_proto):
    fea_desc = {
        "uniq_id": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    for slot in set(C.WIDE_SLOTS + C.DEEP_SLOTS):
        fea_desc[slot]  = tf.io.VarLenFeature(tf.int64)

    feature_dict = tf.io.parse_example(example_proto, fea_desc)

    # [batch_size, label]
    label = feature_dict.pop('label')

    return feature_dict, label


def read_dataset(data_path, days, match_pattern, num_parallel_calls = 12):
    ds_data_files = tn.data.list_files(data_path, days=days, match_pattern=match_pattern)
    dataset = ds_data_files.shard(num_shards=tn.core.shard_num(), index=tn.core.self_shard_id())
    dataset = dataset.interleave(lambda f: tf.data.TFRecordDataset(f, buffer_size=1024 * 100),
                                       cycle_length=4, block_length=8,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(C.BATCH_SIZE)
    dataset = dataset.map(map_func=parse_line_batch, num_parallel_calls=num_parallel_calls)
    dataset = tn.data.BalanceDataset(dataset)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_emb_model(wide_columns, deep_columns):
    wide_embs, deep_embs = [], []

    inputs = {}
    for slot in set(C.WIDE_SLOTS + C.DEEP_SLOTS):
        inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)

    sparse_opt = tn.core.AdaGrad(learning_rate=0.01, initial_g2sum=0.1, initial_scale=0.1)

    if wide_columns:
        wide_embs = tn.layers.EmbeddingFeatures(wide_columns, sparse_opt, name='wide_inputs')(inputs)

    if deep_columns:
        deep_embs = tn.layers.EmbeddingFeatures(deep_columns, sparse_opt, name='deep_inputs')(inputs)

    # must put wide embs at front of outputs list
    emb_model = tf.keras.Model(inputs=inputs, outputs=[wide_embs, deep_embs], name="emb_model")

    return emb_model

def create_sub_model(wide_emb_input_shapes, deep_emb_input_shapes):
    wide, deep = None, None

    wide_inputs = [tf.keras.layers.Input(name="wide_emb_{}".format(i), dtype="float32", shape=shape[1:])
                    for i, shape in enumerate(wide_emb_input_shapes)]

    deep_inputs = [tf.keras.layers.Input(name="deep_emb_{}".format(i), dtype="float32", shape=shape[1:])
                    for i, shape in enumerate(deep_emb_input_shapes)]

    if wide_inputs:
        wide = tf.keras.layers.Concatenate(name='wide_concact', axis=-1)(wide_inputs)

    if deep_inputs:
        deep = tf.keras.layers.Concatenate(name='deep_concact', axis=-1)(deep_inputs)

        for i, unit in enumerate(C.DEEP_HIDDEN_UNITS):
            deep = tf.keras.layers.Dense(unit, activation='relu', name='dnn_{}'.format(i))(deep)

    if wide_inputs and not deep_inputs:
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(wide)
    elif deep_inputs and not wide_inputs:
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(deep)
    else:
        both = tf.keras.layers.concatenate([deep, wide], name='both')
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)

    model = tn.model.Model(inputs=[wide_inputs, deep_inputs], outputs=output, name="sub_model")

    return model

def create_model(wide_columns, deep_columns):
    inputs = {}
    for slot in set(C.WIDE_SLOTS + C.DEEP_SLOTS):
        inputs[slot] = tf.keras.layers.Input(name=slot, shape=(None,), dtype="int64", sparse=True)

    emb_model = create_emb_model(wide_columns, deep_columns)

    assert len(emb_model.output) == 2, "expected emb_model output length is 2 but {}".format(emb_model.output)
    wide_emb_input_shapes = [emb.shape for emb in emb_model.output[0]]
    deep_emb_input_shapes = [emb.shape for emb in emb_model.output[1]]

    wide_embs, deep_embs = emb_model(inputs)
    sub_model = create_sub_model(wide_emb_input_shapes, deep_emb_input_shapes)
    output = sub_model([wide_embs, deep_embs])
    model = tn.model.Model(inputs=inputs, outputs=output, name="full_model")

    dense_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    model.compile(optimizer=tn.optimizer.Optimizer(dense_opt),
                  loss='binary_crossentropy',
                  metrics=['acc', "mse", "mae", 'mape', tf.keras.metrics.AUC(),
                           tn.metric.CTR(), tn.metric.PCTR(), tn.metric.COPC()])

    return model, sub_model


def trained_delta_days(cur_dt):
    last_train_dt = tn.model.read_last_train_dt(C.MODEL_DIR)

    if not last_train_dt:
        return 1

    last_train_dt = datetime.fromisoformat(last_train_dt)
    cur_dt = datetime.fromisoformat(cur_dt)

    return (cur_dt - last_train_dt).days

def dump_predict(result):
    result = np.concatenate(result, axis=1)
    content = ""

    for y, y_pred in result:
        content += "{}\t{}\n".format(y, y_pred)

    filename = "{}/part-{:05d}".format(C.PREDICT_DUMP_PATH, tn.core.self_shard_id())
    tf.io.write_file(filename, content)

    return


def main():
    strategy = tn.distribute.PsStrategy()

    with strategy.scope():
        wide_column, deep_column = columns_builder()
        model, sub_model = create_model(wide_column, deep_column)

        logdir = os.path.join(C.MODEL_DIR, "log", datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                            histogram_freq=1, profile_batch="50,60", embeddings_freq=1)

        days = []
        for dt in C.TRAIN_DAYS:
            delta_day = trained_delta_days(dt)

            # skip have been trained data
            if delta_day <= 0:
                continue

            days.append(dt)

            if delta_day % C.SAVE_MODEL_INTERVAL_DAYS != 0 and dt != C.TRAIN_DAYS[-1]:
                continue

            train_dataset = read_dataset(C.DATA_DIR, days, C.FILE_MATCH_PATTERN)

            cp_cb = tn.callbacks.PsWeightCheckpoint(C.MODEL_DIR, need_save_model=True, dt=dt)
            model.fit(train_dataset, epochs=1, verbose=1, callbacks=[cp_cb])

            days = []

            infer_batch_size = 100
            for tensor in sub_model.inputs:
                tensor.set_shape([infer_batch_size] + list(tensor.shape)[1:])

            sub_model.save('model/tmp')

        if C.PREDICT_DT:
            dataset = read_dataset(C.DATA_DIR, [C.PREDICT_DT], C.FILE_MATCH_PATTERN)
            cp_cb = tn.callbacks.PsWeightCheckpoint(C.MODEL_DIR, need_save_model=False, dt=C.PREDICT_DT)

            result = model.predict(dataset, verbose=1, callbacks=[cp_cb])

            dump_predict(result)

    return

if __name__ == "__main__":
    main()

