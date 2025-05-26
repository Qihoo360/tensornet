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

# Author: zhangyansheng <zhangyansheng@360.cn>
# Copyright(C) 360.cn, all rights reserved.
# Date: 2020/04/15

from tensorflow.keras import metrics

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops.losses import util as tf_losses_utils


class CTR(metrics.Mean):
    """calculate average of label"""

    def __init__(self, name="CTR", dtype=None):
        """Creates a `CTR` instance.

        Args:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(CTR, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        [y_true, y_pred], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight
        )
        y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

        return super(CTR, self).update_state(y_true, sample_weight=sample_weight)


class PCTR(metrics.Mean):
    """calculate average of label"""

    def __init__(self, name="PCTR", dtype=None):
        """Creates a `CTR` instance.

        Args:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(PCTR, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        [y_true, y_pred], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight
        )
        y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

        return super(PCTR, self).update_state(y_pred, sample_weight=sample_weight)


class COPC(metrics.Metric):
    def __init__(self, name="COPC", dtype=None):
        super(COPC, self).__init__(name=name, dtype=dtype)

        with ops.init_scope():
            self.ctr_total = self.add_weight("ctr_total", initializer=init_ops.zeros_initializer)
            self.pctr_total = self.add_weight("pctr_total", initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        [y_true, y_pred], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight
        )
        y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

        ctr_sum = math_ops.reduce_sum(y_true)
        with ops.control_dependencies([ctr_sum]):
            ctr_sum_op = self.ctr_total.assign_add(ctr_sum)

        pctr_sum = math_ops.reduce_sum(y_pred)
        with ops.control_dependencies([pctr_sum]):
            pctr_sum_op = self.pctr_total.assign_add(pctr_sum)

        return control_flow_ops.group(ctr_sum_op, pctr_sum_op)

    def result(self):
        return math_ops.div_no_nan(self.ctr_total, self.pctr_total)
