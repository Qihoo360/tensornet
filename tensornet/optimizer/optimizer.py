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

import tensorflow as tf
import tensornet as tn

from tensornet.core import gen_dense_table_ops

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
import numpy as np


class Optimizer(optimizer_v2.OptimizerV2):
    """ """

    def __init__(self, dense_opt, name="TensornetOptimizer", **kwargs):
        scheduler = tn.core.get_opt_learning_rate(dense_opt)
        if isinstance(scheduler, learning_rate_schedule.LearningRateSchedule):
            self.learning_rate_scheduler = tn.core.get_opt_learning_rate(dense_opt)
        else:
            if isinstance(scheduler, float):
                lr_constant = tf.constant(float(scheduler), dtype=tf.float32)
                self.learning_rate_scheduler = lambda x: lr_constant
        self.dense_table_handle = tn.core.create_dense_table(dense_opt)
        self.is_var_inited = False

        super(Optimizer, self).__init__(name, **kwargs)

    def save_dense_table(self, filepath):
        return tn.core.save_dense_table(self.dense_table_handle, filepath)

    def load_dense_table(self, filepath):
        return tn.core.load_dense_table(self.dense_table_handle, filepath)

    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
        dense_vars = {}

        for grad, var in grads_and_vars:
            if isinstance(grad, ops.IndexedSlices):
                continue

            dense_vars[var.name] = (var, grad)

        vars = [dense_vars[i][0].handle for i in sorted(dense_vars.keys())]
        grads = [dense_vars[i][1] for i in sorted(dense_vars.keys())]

        if not self.is_var_inited:
            with ops.init_scope():
                gen_dense_table_ops.dense_table_init(vars, table_handle=self.dense_table_handle)

            self.is_var_inited = True

            tn.core.barrier()

        gen_dense_table_ops.dense_table_push_pull(
            vars, grads, self.learning_rate_scheduler(self.iterations), table_handle=self.dense_table_handle
        )

        super(Optimizer, self)._distributed_apply(distribution, grads_and_vars, name, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        return control_flow_ops.no_op()

    def _resource_apply_dense(self, grad, var, apply_state=None):
        return control_flow_ops.no_op()

    def get_config(self):
        config = super(Optimizer, self).get_config()
        return config


class PCGrad(Optimizer):
    """ """

    def __init__(self, dense_opt, name="TensornetPCGrad", **kwargs):
        super(PCGrad, self).__init__(dense_opt, name, **kwargs)

    def compute_gradients(self, loss, var_list, tape, grad_loss=None):
        assert type(loss) is list
        num_tasks = len(loss)
        tf.random.shuffle(loss)
        # loss = tf.stack(loss)
        # tf.random.shuffle(loss)
        # print('loss:%s' % loss)

        def sub_loss_compute(x, tape, var_list, grad_loss):
            print("x:%s" % x)
            temp_loss_value = []
            gradients = tape.gradient(x, var_list, grad_loss)
            for grad in gradients:
                if grad is not None:
                    temp_loss_value.append(
                        tf.reshape(
                            grad,
                            [
                                -1,
                            ],
                        )
                    )

            return tf.concat(temp_loss_value, axis=0)

        # compute gradient projections
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task * grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(grads_task[k] * grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.0) * grads_task[k]
            return grad_task

        # compute per-task gradients
        proj_grads_flatten = []
        grads_task = []
        for ls in loss:
            grad_task = sub_loss_compute(ls, tape, var_list, grad_loss)
            grads_task.append(grad_task)

        for grad_task in grads_task:
            proj_grads_flatten.append(proj_grad(grad_task))

        # unpack flattened projected gradients back to their original shapes
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx : start_idx + flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                proj_grad = self._clip_gradients(proj_grad)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars
