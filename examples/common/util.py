# -*- coding: utf-8 -*-
from datetime import datetime

import tensorflow as tf
import tensornet as tn
import numpy as np


def read_dataset(data_path, days, match_pattern, batch_size, parse_func, num_parallel_calls=12):
    ds_data_files = tn.data.list_files(data_path, days=days, match_pattern=match_pattern)
    dataset = ds_data_files.shard(num_shards=tn.core.shard_num(), index=tn.core.self_shard_id())
    dataset = dataset.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=1024 * 100),
        cycle_length=4,
        block_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        map_func=lambda example_proto: parse_func(example_proto), num_parallel_calls=num_parallel_calls
    )
    dataset = tn.data.BalanceDataset(dataset)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def trained_delta_days(cur_dt, model_dir):
    last_train_dt = tn.model.read_last_train_dt(model_dir)

    if not last_train_dt:
        return 1

    last_train_dt = datetime.fromisoformat(last_train_dt)
    cur_dt = datetime.fromisoformat(cur_dt)

    return (cur_dt - last_train_dt).days


def dump_predict(result, path):
    result = np.concatenate(result, axis=1)
    content = ""

    for y, y_pred in result:
        content += "{}\t{}\n".format(y, y_pred)

    filename = "{}/part-{:05d}".format(path, tn.core.self_shard_id())
    tf.io.write_file(filename, content)

    return
