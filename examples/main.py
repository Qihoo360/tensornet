#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import tensorflow as tf
import tensornet as tn

from common.util import read_dataset, trained_delta_days, dump_predict
from common.config import Config as C

from models.deepfm import DeepFM
from models.wide_deep import WideDeep
from models.deep_cross import DCN


def parse_line_batch(example_proto):
    fea_desc = {"uniq_id": tf.io.FixedLenFeature([], tf.string), "label": tf.io.FixedLenFeature([], tf.int64)}

    for slot in set(C.LINEAR_SLOTS + C.DEEP_SLOTS):
        fea_desc[slot] = tf.io.VarLenFeature(tf.int64)

    feature_dict = tf.io.parse_example(example_proto, fea_desc)
    label = feature_dict["label"]
    return feature_dict, label


def create_model():
    if C.MODEL_TYPE == "DeepFM":
        return DeepFM(C.LINEAR_SLOTS, C.DEEP_SLOTS, C.DEEP_HIDDEN_UNITS)
    elif C.MODEL_TYPE == "WideDeep":
        return WideDeep(C.LINEAR_SLOTS, C.DEEP_SLOTS, C.DEEP_HIDDEN_UNITS)
    elif C.MODEL_TYPE == "DCN":
        return DCN(C.DEEP_SLOTS, C.DEEP_HIDDEN_UNITS)
    else:
        import sys

        sys.exit("unsupported model type: " + C.MODEL_TYPE)


def main():
    strategy = tn.distribute.PsStrategy()

    with strategy.scope():
        model, sub_model = create_model()
        dense_opt = tn.core.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        model.compile(
            optimizer=tn.optimizer.Optimizer(dense_opt),
            loss="binary_crossentropy",
            metrics=[
                "acc",
                "mse",
                "mae",
                "mape",
                tf.keras.metrics.AUC(),
                tn.metric.CTR(),
                tn.metric.PCTR(),
                tn.metric.COPC(),
            ],
        )

        logdir = os.path.join(C.MODEL_DIR, "log", datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=logdir, histogram_freq=1, profile_batch="50,60", embeddings_freq=1
        )

        days = []
        for dt in C.TRAIN_DAYS:
            delta_day = trained_delta_days(dt, C.MODEL_DIR)

            # skip have been trained data
            if delta_day <= 0:
                continue

            days.append(dt)

            if delta_day % C.SAVE_MODEL_INTERVAL_DAYS != 0 and dt != C.TRAIN_DAYS[-1]:
                continue

            train_dataset = read_dataset(C.DATA_DIR, days, C.FILE_MATCH_PATTERN, C.BATCH_SIZE, parse_line_batch)

            cp_cb = tn.callbacks.PsWeightCheckpoint(C.MODEL_DIR, need_save_model=True, dt=dt)
            model.fit(train_dataset, epochs=1, verbose=1, callbacks=[cp_cb, tb_cb])

            days = []

            infer_batch_size = 100
            for tensor in sub_model.inputs:
                tensor.set_shape([infer_batch_size] + list(tensor.shape)[1:])

            sub_model.save("model/saved_model")

        if C.PREDICT_DT:
            dataset = read_dataset(C.DATA_DIR, [C.PREDICT_DT], C.FILE_MATCH_PATTERN, parse_line_batch)
            cp_cb = tn.callbacks.PsWeightCheckpoint(C.MODEL_DIR, need_save_model=False, dt=C.PREDICT_DT)

            result = model.predict(dataset, verbose=1, callbacks=[cp_cb])

            dump_predict(result, C.PREDICT_DUMP_PATH)

    return


if __name__ == "__main__":
    main()
