#!/usr/bin/env python3

import os
import shutil

from common.config import Config as C
from main import main

if "OMPI_COMM_WORLD_RANK" not in os.environ or os.environ["OMPI_COMM_WORLD_RANK"] == "0":
    need_gen = False
    for d in C.TRAIN_DAYS:
        if not os.path.exists(f"{C.DATA_DIR}/{d}"):
            need_gen = True
            break

    if need_gen:
        import gen_example_data

        gen_example_data.main()

    # remove existing model
    shutil.rmtree(C.MODEL_DIR, ignore_errors=True)

main()
