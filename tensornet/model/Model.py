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

import os
import json

import tensornet as tn
import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.framework import ops
from tensornet import _opentelemetry as otel


def load_done_info(cp_dir):
    done_file = os.path.join(cp_dir, "_checkpoint")
    if not tf.io.gfile.exists(done_file):
        return

    last_done_info = tf.io.read_file(done_file)

    try:
        return json.loads(last_done_info.numpy())
    except Exception:
        return


def save_done_info(cp_dir, dt):
    done_file = os.path.join(cp_dir, "_checkpoint")
    done_info = {
        "dt": dt,
    }

    if tf.io.gfile.exists(done_file):
        tf.io.gfile.remove(done_file)

    tf.io.write_file(done_file, json.dumps(done_info))


def read_last_train_dt(filepath):
    done_info = load_done_info(filepath)

    if not done_info or "dt" not in done_info:
        return

    return done_info["dt"]


class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        # used by PsWeightCheckpoint. this variable to indicate model have been loaded
        # when model.fit() is called multiple times through multi days training
        self.is_loaded_from_checkpoint = False

        self._backward_count = self.add_weight("backward_count", shape=[], dtype=tf.int64, trainable=False)

    def fit(self, *args, **kwargs):
        with otel.start_as_current_span("tensornet-fit"):
            return super().fit(*args, **kwargs)

    def train_on_batch(self, *args, **kwargs):
        with otel.start_as_current_span("tensornet-train-on-batch"):
            return super().train_on_batch(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with otel.start_as_current_span("tensornet-predict"):
            return super().predict(*args, **kwargs)

    def predict_on_batch(self, *args, **kwargs):
        with otel.start_as_current_span("tensornet-predict-on-batch"):
            return super().predict_on_batch(*args, **kwargs)

    def train_step(self, data):
        """override parent train_step, see description in parent
        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.backwards(list(zip(gradients, self.trainable_variables)))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        """override parent inference step, support return y label together"""
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)

        return y, y_pred

    def backwards(self, grads_and_vars):
        backward_ops = []

        for layer in self.layers:
            if not hasattr(layer, "backwards"):
                continue

            op = layer.backwards(grads_and_vars)
            if op:
                backward_ops.append(op)

        with ops.control_dependencies(backward_ops):
            # trick, use backward_count tensor add graph dependency
            self._backward_count.assign_add(1)

        return

    def save_weights(self, filepath, overwrite=True, save_format=None, dt="", root=True, mode="txt", **kwargs):
        cp_dir = os.path.join(filepath, dt)
        # sparse weight
        for layer in self.layers:
            assert type(layer) is not tf.keras.Model, "not support direct use keras.Model, use tn.model.Model instead"

            if isinstance(layer, type(self)):
                layer.save_weights(filepath, overwrite, save_format, dt, False, mode)
            elif isinstance(layer, tn.layers.EmbeddingFeatures):
                layer.save_sparse_table(cp_dir, mode)
            elif isinstance(layer, tn.layers.SequenceEmbeddingFeatures):
                layer.save_sparse_table(cp_dir, mode)
            elif isinstance(layer, tn.layers.TNBatchNormalizationBase):
                if tn.core.self_shard_id() == 0:
                    layer.bn_statistics_pull()
                    layer.save_bn_table(cp_dir)

        if self.optimizer:
            self.optimizer.save_dense_table(cp_dir)

        # only the first node save the model, other node use the first node saved model
        # when load_weights
        if tn.core.self_shard_id() == 0 and root:
            # actually, we use tensorflow checkpoint only when system restart, this could be
            # done by tensornet checkpoint instead, but we need a pull operation to fetch
            # weight from remote node. TODO use tensornet checkpoint to refine code
            tf_cp_file = os.path.join(cp_dir, "tf_checkpoint")
            super(Model, self).save_weights(tf_cp_file, overwrite, save_format="tf")

            save_done_info(filepath, dt)

        self.is_loaded_from_checkpoint = True

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, include_dt=False, root=True, mode="txt"):
        if not include_dt:
            last_train_dt = read_last_train_dt(filepath)
            # not saved model info found
            if not last_train_dt:
                return
            cp_dir = os.path.join(filepath, last_train_dt)
        else:
            model_ckpt = os.path.join(filepath, "checkpoint")
            if not tf.io.gfile.exists(model_ckpt):
                return
            cp_dir = filepath

        if not self.is_loaded_from_checkpoint:
            # sparse weight
            for layer in self.layers:
                assert type(layer) is not tf.keras.Model, (
                    "not support direct use keras.Model, use tn.model.Model instead"
                )

                if isinstance(layer, type(self)):
                    layer.load_weights(filepath, by_name, skip_mismatch, include_dt, False, mode)
                elif isinstance(layer, tn.layers.EmbeddingFeatures):
                    layer.load_sparse_table(cp_dir, mode)
                elif isinstance(layer, tn.layers.SequenceEmbeddingFeatures):
                    layer.load_sparse_table(cp_dir, mode)
                elif isinstance(layer, tn.layers.TNBatchNormalizationBase):
                    layer.load_bn_table(cp_dir)

            # dense weight
            if self.optimizer:
                self.optimizer.load_dense_table(cp_dir)

            self.is_loaded_from_checkpoint = True

        if root:
            tf_cp_file = os.path.join(cp_dir, "tf_checkpoint")
            super(Model, self).load_weights(tf_cp_file, by_name, skip_mismatch).expect_partial()

    def load_sparse_weights(self, filepath, by_name=False, skip_mismatch=False, include_dt=False, root=True):
        if not include_dt:
            last_train_dt = read_last_train_dt(filepath)
            # not saved model info found
            if not last_train_dt:
                return
            cp_dir = os.path.join(filepath, last_train_dt)
        else:
            model_ckpt = os.path.join(filepath, "checkpoint")
            if not tf.io.gfile.exists(model_ckpt):
                return
            cp_dir = filepath
        print(self.is_loaded_from_checkpoint)

        if not self.is_loaded_from_checkpoint:
            # sparse weight
            for layer in self.layers:
                assert type(layer) is not tf.keras.Model, (
                    "not support direct use keras.Model, use tn.model.Model instead"
                )

                if isinstance(layer, type(self)):
                    layer.load_sparse_weights(filepath, by_name, skip_mismatch, include_dt, False)
                elif isinstance(layer, tn.layers.EmbeddingFeatures):
                    layer.load_sparse_table(cp_dir)

            self.is_loaded_from_checkpoint = True

        if root:
            tf_cp_file = os.path.join(cp_dir, "tf_checkpoint")
            super(Model, self).load_weights(tf_cp_file, by_name, skip_mismatch)

    def show_decay(self, delta_days=0):
        for layer in self.layers:
            assert type(layer) is not tf.keras.Model, "not support direct use keras.Model, use tn.model.Model instead"

            if isinstance(layer, type(self)):
                layer.show_decay(delta_days)
            elif isinstance(layer, tn.layers.EmbeddingFeatures):
                layer.show_decay(delta_days)
            elif isinstance(layer, tn.layers.SequenceEmbeddingFeatures):
                layer.show_decay(delta_days)


class PCGradModel(Model):
    def train_step(self, data):
        """override parent train_step, see description in parent
        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)
            loss, losses = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

            print("total loss: %s, sub loss:%s" % (loss, losses))

            grads_and_vars = self.optimizer.compute_gradients(losses, self.trainable_variables, tape)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        print("grads_and_vars:%s" % grads_and_vars)
        self.backwards(list(zip(gradients, self.trainable_variables)))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
