# Copyright (c) 2020, Qihoo, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
import datetime

import tensornet as tn
from tensorflow.python.keras.callbacks import Callback


class PsWeightCheckpoint(Callback):
    """Save ps weight after every fit.
    """
    def __init__(self, checkpoint_dir, need_save_model=True, save_mode="txt", dt=None):
        """
        :param checkpoint_dir: path of save model
        :param need_save_model: whether save model
        :param save_mode:
            'txt' : model will save with text format.
            'bin' : model will save with binary format
        """
        self.checkpoint_dir = checkpoint_dir
        self.need_save_model = need_save_model
        self.save_mode = save_mode
        self.dt = dt

        super(PsWeightCheckpoint, self).__init__()

    def load_model(self):
        tn.core.barrier()

        self.model.load_weights(self.checkpoint_dir, mode=self.save_mode)

        tn.core.barrier()

    def reset_balance_dataset(self):
        tn.core.barrier()

        tn.core.reset_balance_dataset()

        tn.core.barrier()

    def on_train_begin(self, logs=None):
        assert isinstance(self.model.optimizer, tn.optimizer.Optimizer)

        self.load_model()

        self.reset_balance_dataset()

    def on_train_end(self, logs=None):
        tn.core.barrier()

        self.model.show_decay()

        if not self.need_save_model:
            return

        self.model.save_weights(self.checkpoint_dir, dt=self.dt, mode=self.save_mode)

    def on_predict_begin(self, logs=None):
        self.load_model()

        self.reset_balance_dataset()

    def on_predict_end(self, logs=None):
        tn.core.barrier()

    def on_test_begin(self, logs=None):
        self.load_model()

        self.reset_balance_dataset()

    def on_test_end(self, logs=None):
        tn.core.barrier()

