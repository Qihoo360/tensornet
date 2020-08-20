"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: dense_table_ops.cc
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
@tf_export('dense_table_init')
def dense_table_init(vars, table_handle, name=None):
  r"""dense table init

  Args:
    vars: A list of at least 1 `Tensor` objects with type `resource`.
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
        _ctx._context_handle, tld.device_name, "DenseTableInit", name,
        tld.op_callbacks, vars, "table_handle", table_handle)
      return _result
    except _core._FallbackException:
      try:
        return dense_table_init_eager_fallback(
            vars, table_handle=table_handle, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              dense_table_init, vars=vars, table_handle=table_handle,
                                name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if not isinstance(vars, (list, tuple)):
    raise TypeError(
        "Expected list for 'vars' argument to "
        "'dense_table_init' Op, not %r." % vars)
  _attr_N = len(vars)
  table_handle = _execute.make_int(table_handle, "table_handle")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseTableInit", vars=vars, table_handle=table_handle, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          dense_table_init, vars=vars, table_handle=table_handle, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
DenseTableInit = tf_export("raw_ops.DenseTableInit")(_ops.to_raw_op(dense_table_init))


def dense_table_init_eager_fallback(vars, table_handle, name, ctx):
  if not isinstance(vars, (list, tuple)):
    raise TypeError(
        "Expected list for 'vars' argument to "
        "'dense_table_init' Op, not %r." % vars)
  _attr_N = len(vars)
  table_handle = _execute.make_int(table_handle, "table_handle")
  vars = _ops.convert_n_to_tensor(vars, _dtypes.resource)
  _inputs_flat = list(vars)
  _attrs = ("table_handle", table_handle, "N", _attr_N)
  _result = _execute.execute(b"DenseTableInit", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_dispatch_list
@tf_export('dense_table_push_pull')
def dense_table_push_pull(vars, grads, table_handle, name=None):
  r"""push pull variable from parameter server

  Args:
    vars: A list of at least 1 `Tensor` objects with type `resource`.
    grads: A list with the same length as `vars` of `Tensor` objects with type `float32`.
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
        _ctx._context_handle, tld.device_name, "DenseTablePushPull", name,
        tld.op_callbacks, vars, grads, "table_handle", table_handle)
      return _result
    except _core._FallbackException:
      try:
        return dense_table_push_pull_eager_fallback(
            vars, grads, table_handle=table_handle, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              dense_table_push_pull, vars=vars, grads=grads,
                                     table_handle=table_handle, name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if not isinstance(vars, (list, tuple)):
    raise TypeError(
        "Expected list for 'vars' argument to "
        "'dense_table_push_pull' Op, not %r." % vars)
  _attr_N = len(vars)
  if not isinstance(grads, (list, tuple)):
    raise TypeError(
        "Expected list for 'grads' argument to "
        "'dense_table_push_pull' Op, not %r." % grads)
  if len(grads) != _attr_N:
    raise ValueError(
        "List argument 'grads' to 'dense_table_push_pull' Op with length %d "
        "must match length %d of argument 'vars'." %
        (len(grads), _attr_N))
  table_handle = _execute.make_int(table_handle, "table_handle")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DenseTablePushPull", vars=vars, grads=grads,
                              table_handle=table_handle, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          dense_table_push_pull, vars=vars, grads=grads,
                                 table_handle=table_handle, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
DenseTablePushPull = tf_export("raw_ops.DenseTablePushPull")(_ops.to_raw_op(dense_table_push_pull))


def dense_table_push_pull_eager_fallback(vars, grads, table_handle, name, ctx):
  if not isinstance(vars, (list, tuple)):
    raise TypeError(
        "Expected list for 'vars' argument to "
        "'dense_table_push_pull' Op, not %r." % vars)
  _attr_N = len(vars)
  if not isinstance(grads, (list, tuple)):
    raise TypeError(
        "Expected list for 'grads' argument to "
        "'dense_table_push_pull' Op, not %r." % grads)
  if len(grads) != _attr_N:
    raise ValueError(
        "List argument 'grads' to 'dense_table_push_pull' Op with length %d "
        "must match length %d of argument 'vars'." %
        (len(grads), _attr_N))
  table_handle = _execute.make_int(table_handle, "table_handle")
  vars = _ops.convert_n_to_tensor(vars, _dtypes.resource)
  grads = _ops.convert_n_to_tensor(grads, _dtypes.float32)
  _inputs_flat = list(vars) + list(grads)
  _attrs = ("table_handle", table_handle, "N", _attr_N)
  _result = _execute.execute(b"DenseTablePushPull", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

