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


import tensornet as tn

from tensornet.core import gen_dense_table_ops

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops


class Optimizer(optimizer_v2.OptimizerV2):
    """
    """
    def __init__(self,
                 dense_opt,
                 name='TensornetOptimizer',
                 **kwargs):
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

        gen_dense_table_ops.dense_table_push_pull(vars, grads, table_handle=self.dense_table_handle)

        super(Optimizer, self)._distributed_apply(distribution, grads_and_vars, name, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        return control_flow_ops.no_op()

    def _resource_apply_dense(self, grad, var, apply_state=None):
        return control_flow_ops.no_op()

    def get_config(self):
        config = super(Optimizer, self).get_config()
        return config
