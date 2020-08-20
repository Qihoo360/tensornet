"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: balance_dataset_ops.cc
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export


@_dispatch.add_dispatch_list
@tf_export('balance_dataset')
def balance_dataset(input_dataset, output_types, output_shapes, name=None):
  r"""balance input data between datasets

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "BalanceDataset", name,
        tld.op_callbacks, input_dataset, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._FallbackException:
      try:
        return balance_dataset_eager_fallback(
            input_dataset, output_types=output_types,
            output_shapes=output_shapes, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              balance_dataset, input_dataset=input_dataset,
                               output_types=output_types,
                               output_shapes=output_shapes, name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'balance_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'balance_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BalanceDataset", input_dataset=input_dataset,
                          output_types=output_types,
                          output_shapes=output_shapes, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          balance_dataset, input_dataset=input_dataset,
                           output_types=output_types,
                           output_shapes=output_shapes, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BalanceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BalanceDataset = tf_export("raw_ops.BalanceDataset")(_ops.to_raw_op(balance_dataset))


def balance_dataset_eager_fallback(input_dataset, output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'balance_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'balance_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"BalanceDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BalanceDataset", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

