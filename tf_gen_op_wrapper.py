#!/usr/bin/env python3

import sys
from tensorflow.python import _pywrap_python_op_gen
from tensorflow.python.client import pywrap_tf_session as py_tf

lib_file = sys.argv[1]
lib_handle = py_tf.TF_LoadLibrary(lib_file)
lib_ops = py_tf.TF_GetOpList(lib_handle)

wrapper_codes = _pywrap_python_op_gen.GetPythonWrappers(lib_ops).decode()

source_files = sys.argv[2:]

if source_files:
    marker = "This file is MACHINE GENERATED! Do not edit."
    source_line = f"Original C++ source file: {', '.join(source_files)}"
    wrapper_codes = wrapper_codes.replace(marker, f"{marker}\n{source_line}", 1)

print(wrapper_codes)
