"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: sparse_table_ops.cc
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
@tf_export('sparse_table_pull')
def sparse_table_pull(resources, values, table_handle, name=None):
  r"""pull variable from parameter server

  Args:
    resources: A list of at least 1 `Tensor` objects with type `resource`.
    values: A list with the same length as `resources` of `Tensor` objects with type `int64`.
    table_handle: An `int`.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `resources` of `Tensor` objects with type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "SparseTablePull", name,
        tld.op_callbacks, resources, values, "table_handle", table_handle)
      return _result
    except _core._FallbackException:
      try:
        return sparse_table_pull_eager_fallback(
            resources, values, table_handle=table_handle, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              sparse_table_pull, resources=resources, values=values,
                                 table_handle=table_handle, name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if not isinstance(resources, (list, tuple)):
    raise TypeError(
        "Expected list for 'resources' argument to "
        "'sparse_table_pull' Op, not %r." % resources)
  _attr_N = len(resources)
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'sparse_table_pull' Op, not %r." % values)
  if len(values) != _attr_N:
    raise ValueError(
        "List argument 'values' to 'sparse_table_pull' Op with length %d "
        "must match length %d of argument 'resources'." %
        (len(values), _attr_N))
  table_handle = _execute.make_int(table_handle, "table_handle")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseTablePull", resources=resources, values=values,
                           table_handle=table_handle, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          sparse_table_pull, resources=resources, values=values,
                             table_handle=table_handle, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("table_handle", _op._get_attr_int("table_handle"), "N",
              _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseTablePull", _inputs_flat, _attrs, _result)
  return _result

SparseTablePull = tf_export("raw_ops.SparseTablePull")(_ops.to_raw_op(sparse_table_pull))


def sparse_table_pull_eager_fallback(resources, values, table_handle, name, ctx):
  if not isinstance(resources, (list, tuple)):
    raise TypeError(
        "Expected list for 'resources' argument to "
        "'sparse_table_pull' Op, not %r." % resources)
  _attr_N = len(resources)
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'sparse_table_pull' Op, not %r." % values)
  if len(values) != _attr_N:
    raise ValueError(
        "List argument 'values' to 'sparse_table_pull' Op with length %d "
        "must match length %d of argument 'resources'." %
        (len(values), _attr_N))
  table_handle = _execute.make_int(table_handle, "table_handle")
  resources = _ops.convert_n_to_tensor(resources, _dtypes.resource)
  values = _ops.convert_n_to_tensor(values, _dtypes.int64)
  _inputs_flat = list(resources) + list(values)
  _attrs = ("table_handle", table_handle, "N", _attr_N)
  _result = _execute.execute(b"SparseTablePull", _attr_N, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseTablePull", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_dispatch_list
@tf_export('sparse_table_push')
def sparse_table_push(values, grads, table_handle, name=None):
  r"""push variable from parameter server

  Args:
    values: A list of at least 1 `Tensor` objects with type `int64`.
    grads: A list with the same length as `values` of `Tensor` objects with type `float32`.
    table_handle: An `int`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "SparseTablePush", name,
        tld.op_callbacks, values, grads, "table_handle", table_handle)
      return _result
    except _core._FallbackException:
      try:
        return sparse_table_push_eager_fallback(
            values, grads, table_handle=table_handle, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              sparse_table_push, values=values, grads=grads,
                                 table_handle=table_handle, name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'sparse_table_push' Op, not %r." % values)
  _attr_N = len(values)
  if not isinstance(grads, (list, tuple)):
    raise TypeError(
        "Expected list for 'grads' argument to "
        "'sparse_table_push' Op, not %r." % grads)
  if len(grads) != _attr_N:
    raise ValueError(
        "List argument 'grads' to 'sparse_table_push' Op with length %d "
        "must match length %d of argument 'values'." %
        (len(grads), _attr_N))
  table_handle = _execute.make_int(table_handle, "table_handle")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseTablePush", values=values, grads=grads,
                           table_handle=table_handle, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          sparse_table_push, values=values, grads=grads,
                             table_handle=table_handle, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
SparseTablePush = tf_export("raw_ops.SparseTablePush")(_ops.to_raw_op(sparse_table_push))


def sparse_table_push_eager_fallback(values, grads, table_handle, name, ctx):
  if not isinstance(values, (list, tuple)):
    raise TypeError(
        "Expected list for 'values' argument to "
        "'sparse_table_push' Op, not %r." % values)
  _attr_N = len(values)
  if not isinstance(grads, (list, tuple)):
    raise TypeError(
        "Expected list for 'grads' argument to "
        "'sparse_table_push' Op, not %r." % grads)
  if len(grads) != _attr_N:
    raise ValueError(
        "List argument 'grads' to 'sparse_table_push' Op with length %d "
        "must match length %d of argument 'values'." %
        (len(grads), _attr_N))
  table_handle = _execute.make_int(table_handle, "table_handle")
  values = _ops.convert_n_to_tensor(values, _dtypes.int64)
  grads = _ops.convert_n_to_tensor(grads, _dtypes.float32)
  _inputs_flat = list(values) + list(grads)
  _attrs = ("table_handle", table_handle, "N", _attr_N)
  _result = _execute.execute(b"SparseTablePush", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

