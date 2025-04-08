# -*- coding: utf-8 -*-


class Config(object):
    DATA_DIR = "./data/"
    FILE_MATCH_PATTERN = "tf-*"

    MODEL_DIR = "./model/"

    BATCH_SIZE = 32

    TRAIN_DAYS = ["2020-05-10", "2020-05-11"]

    SAVE_MODEL_INTERVAL_DAYS = 3

    DEEP_HIDDEN_UNITS = [512, 256, 256]

    LINEAR_SLOTS = ["1", "2", "3", "4"]
    DEEP_SLOTS = ["1", "2", "3", "4"]

    PREDICT_DT = None
    PREDICT_DUMP_PATH = "./predict"

    MODEL_TYPE = "WideDeep"  # supported models: WideDeep, DeepFM, DCN
