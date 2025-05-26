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

from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.util import nest


from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_utils


def match_dtype_and_rank(y_t, y_p, sw):
    """Match dtype and rank of predictions."""
    if y_t.shape.rank == 1 and y_p.shape.rank == 2:
        y_t = array_ops.expand_dims_v2(y_t, axis=-1)
    if sw is not None:
        if sw.shape.rank == 1 and y_p.shape.rank == 2:
            sw = array_ops.expand_dims_v2(sw, axis=-1)

    # Dtype.
    y_t = math_ops.cast(y_t, y_p.dtype)
    if sw is not None:
        sw = math_ops.cast(sw, y_p.dtype)
    return y_t, y_p, sw


def apply_mask(y_p, sw):
    """Applies any mask on predictions to sample weights."""
    # Handle Keras mask on outputs.
    mask = getattr(y_p, "_keras_mask", None)
    if mask is not None:
        mask = math_ops.cast(mask, y_p.dtype)
        if sw is not None:
            mask, _, sw = tf_losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=sw)
            sw *= mask
        else:
            sw = mask
    return sw


class PCGradLossesContainer(compile_utils.LossesContainer):
    def __init__(self, losses, loss_weights=None, output_names=None):
        super(PCGradLossesContainer, self).__init__(losses, loss_weights=loss_weights, output_names=output_names)

    def __call__(self, y_true, y_pred, sample_weight=None, regularization_losses=None):
        y_true = self._conform_to_outputs(y_pred, y_true)
        sample_weight = self._conform_to_outputs(y_pred, sample_weight)

        if not self._built:
            self._build(y_pred)

        y_pred = nest.flatten(y_pred)
        y_true = nest.flatten(y_true)
        sample_weight = nest.flatten(sample_weight)

        loss_values = []  # Used for gradient calculation.
        not_none_loss_values = []
        loss_metric_values = []  # Used for loss metric calculation.
        batch_dim = None
        zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights, self._per_output_metrics)
        for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
            if y_t is None or loss_obj is None:  # Ok to have no loss for an output.
                continue
            print("loss_weight:%s" % loss_weight)
            y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            sw = apply_mask(y_p, sw)

            loss_value = loss_obj(y_t, y_p, sample_weight=sw)
            loss_metric_value = loss_value
            # Correct for the `Mean` loss metrics counting each replica as a batch.
            if loss_obj.reduction == losses_utils.ReductionV2.SUM:
                loss_metric_value *= ds_context.get_strategy().num_replicas_in_sync

            if batch_dim is None:
                batch_dim = array_ops.shape(y_t)[0]
            if metric_obj is not None:
                metric_obj.update_state(loss_metric_value, sample_weight=batch_dim)

            if loss_weight is not None:
                loss_value *= loss_weight
                loss_metric_value *= loss_weight

            if (
                loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
                or loss_obj.reduction == losses_utils.ReductionV2.AUTO
            ):
                loss_value = losses_utils.scale_loss_for_distribution(loss_value)

            loss_values.append(loss_value)
            loss_metric_values.append(loss_metric_value)
            if loss_weight != 0.0:
                not_none_loss_values.append(loss_value)

        if regularization_losses:
            regularization_losses = losses_utils.cast_losses_to_common_dtype(regularization_losses)
            reg_loss = math_ops.add_n(regularization_losses)
            loss_metric_values.append(reg_loss)
            loss_values.append(losses_utils.scale_loss_for_distribution(reg_loss))

        if loss_values:
            loss_metric_values = losses_utils.cast_losses_to_common_dtype(loss_metric_values)
            total_loss_metric_value = math_ops.add_n(loss_metric_values)
            self._loss_metric.update_state(total_loss_metric_value, sample_weight=batch_dim)

            loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
            not_none_loss_values = losses_utils.cast_losses_to_common_dtype(not_none_loss_values)
            total_loss = math_ops.add_n(loss_values)

            return total_loss, not_none_loss_values
        else:
            # Ok for a model to have no compiled loss.
            return array_ops.zeros(shape=())
