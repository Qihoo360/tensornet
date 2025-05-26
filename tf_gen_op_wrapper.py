#!/usr/bin/env python3

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
