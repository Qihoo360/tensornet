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
@tf_export('set_bn_vars')
def set_bn_vars(vars, table_handle, name=None):
  r"""set bn table mean var count

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
        _ctx._context_handle, tld.device_name, "BnVarsSet", name,
        tld.op_callbacks, vars, "table_handle", table_handle)
      return _result
    except _core._FallbackException:
      try:
        return set_bn_vars_eager_fallback(
            vars, table_handle=table_handle, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              set_bn_vars, vars=vars, table_handle=table_handle,
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
        "'set_bn_vars' Op, not %r." % vars)
  _attr_N = len(vars)
  table_handle = _execute.make_int(table_handle, "table_handle")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BnVarsSet", vars=vars, table_handle=table_handle, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          set_bn_vars, vars=vars, table_handle=table_handle, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
SetBnVars = tf_export("raw_ops.BnVarsSet")(_ops.to_raw_op(set_bn_vars))


def set_bn_vars_eager_fallback(vars, table_handle, name, ctx):
  if not isinstance(vars, (list, tuple)):
    raise TypeError(
        "Expected list for 'vars' argument to "
        "'bn_vars_pull' Op, not %r." % vars)
  _attr_N = len(vars)
  table_handle = _execute.make_int(table_handle, "table_handle")
  vars = _ops.convert_n_to_tensor(vars, _dtypes.resource)
  _inputs_flat = list(vars)
  _attrs = ("table_handle", table_handle, "N", _attr_N)
  _result = _execute.execute(b"BnVarsSet", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_dispatch_list
@tf_export('bn_vars_pull')
def bn_vars_pull(vars, table_handle, name=None):
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
        _ctx._context_handle, tld.device_name, "BnVarsPull", name,
        tld.op_callbacks, vars, grads, "table_handle", table_handle)
      return _result
    except _core._FallbackException:
      try:
        return bn_vars_pull_eager_fallback(
            vars, grads, table_handle=table_handle, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
      except (TypeError, ValueError):
        result = _dispatch.dispatch(
              bn_vars_pull, vars=vars, table_handle=table_handle, name=name)
        if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
          return result
        raise
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if not isinstance(vars, (list, tuple)):
    raise TypeError(
        "Expected list for 'vars' argument to "
        "'bn_vars_pull' Op, not %r." % vars)
  _attr_N = len(vars)
  table_handle = _execute.make_int(table_handle, "table_handle")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BnVarsPull", vars=vars,
                              table_handle=table_handle, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          bn_vars_pull, vars=vars,
                                 table_handle=table_handle, name=name)
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
BnVarsPull = tf_export("raw_ops.BnVarsPull")(_ops.to_raw_op(bn_vars_pull))


def bn_vars_pull_eager_fallback(vars, table_handle, name, ctx):
  if not isinstance(vars, (list, tuple)):
    raise TypeError(
        "Expected list for 'vars' argument to "
        "'bn_vars_pull' Op, not %r." % vars)
  _attr_N = len(vars)
  table_handle = _execute.make_int(table_handle, "table_handle")
  vars = _ops.convert_n_to_tensor(vars, _dtypes.resource)
  _inputs_flat = list(vars)
  _attrs = ("table_handle", table_handle, "N", _attr_N)
  _result = _execute.execute(b"BnVarsPull", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

