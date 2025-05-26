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

import tensornet as tn
from tensorflow.python.keras.callbacks import Callback


class PsWeightCheckpoint(Callback):
    """Save ps weight after every fit."""

    def __init__(
        self,
        checkpoint_dir,
        checkpoint_save=None,
        need_save_model=False,
        dt=None,
        delta_days=0,
        save_mode="txt",
        model_path_incl_dt=False,
        **kwargs,
    ):
        """
        :param checkpoint_dir: path of save model
        :param need_save_model: whether save model
        :param checkpoint_save: path of the saving model path [None in predict or evaluate mode]
        :param model_path_incl_dt: path checkpoint_dir include dt[train & predict are different]
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_save = checkpoint_save if checkpoint_save else checkpoint_dir
        self.need_save_model = need_save_model
        self.need_load_model = kwargs.get("need_load_model", True)
        self.save_mode = save_mode
        self.load_mode = kwargs.get("load_mode", self.save_mode)
        self.model_path_incl_dt = model_path_incl_dt
        self.dt = dt
        self.delta_days = delta_days

        super(PsWeightCheckpoint, self).__init__()

    def load_model(self):
        tn.core.barrier()
        if self.need_load_model:
            self.model.load_weights(self.checkpoint_dir, include_dt=self.model_path_incl_dt, mode=self.load_mode)
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
        self.model.show_decay(self.delta_days)
        if not self.need_save_model:
            return
        self.model.save_weights(self.checkpoint_save, dt=self.dt, mode=self.save_mode)

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

    def on_epoch_end(self, epoch, logs=None):
        tn.core.barrier()
        super().on_epoch_end(epoch, logs)
