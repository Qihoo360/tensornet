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

import os
import tensorflow as tf
from tensorflow import sysconfig
from tensorflow.python import _pywrap_tensorflow_internal as internal

main_inc_dir = sysconfig.get_include()
main_link_dir = sysconfig.get_lib()


def escape_cmake_str(s):
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("$", "\\$")
    return f'"{s}"'


compile_defs = []
compile_flags = []
include_dirs = [main_inc_dir]
link_flags = []
link_dirs = [main_link_dir]
link_libs = []

for flag in sysconfig.get_compile_flags():
    if flag.startswith("-I"):
        d = flag[2:]
        if d not in include_dirs:
            include_dirs.append(d)
    elif flag.startswith("-D"):
        if flag not in compile_defs:
            compile_defs.append(flag)
    else:
        compile_flags.append(flag)

for flag in sysconfig.get_link_flags():
    if flag.startswith("-L"):
        d = flag[2:]
        if d not in link_dirs:
            link_dirs.append(d)
    elif (
        flag.startswith("-l")
        or flag.endswith(".so")
        or flag.endswith(".dylib")
        or flag.endswith(".dll")
        or flag.endswith(".a")
        or flag.endswith(".lib")
        or flag.endswith(".o")
        or flag.endswith(".obj")
    ):
        if flag not in link_libs:
            link_libs.append(flag)
    else:
        link_flags.append(flag)


print(f"set(Tensorflow_VERSION {escape_cmake_str(tf.__version__)})")
print(f"set(Tensorflow_ROOT_DIR {escape_cmake_str(os.path.dirname(tf.__file__))})")

print("")
print("unset(Tensorflow_DEFINITIONS)")
for x in compile_defs:
    print(f"list(APPEND Tensorflow_DEFINITIONS {escape_cmake_str(x)})")

print("")
print("unset(Tensorflow_COMPILE_OPTIONS)")
for x in compile_flags:
    print(f"list(APPEND Tensorflow_COMPILE_OPTIONS {escape_cmake_str(x)})")

print("")
print(f"set(Tensorflow_INCLUDE_DIR {escape_cmake_str(main_inc_dir)})")
print("unset(Tensorflow_INCLUDE_DIRS)")
for x in include_dirs:
    print(f"list(APPEND Tensorflow_INCLUDE_DIRS {escape_cmake_str(x)})")

print("")
print("unset(Tensorflow_LIBRARY_DIRS)")
for x in link_dirs:
    print(f"list(APPEND Tensorflow_LIBRARY_DIRS {escape_cmake_str(x)})")

print("")
print(f"set(Tensorflow_LIBRARY {escape_cmake_str(link_libs[0])})")  # at least one
print("unset(Tensorflow_LIBRARIES)")
for x in link_libs:
    print(f"list(APPEND Tensorflow_LIBRARIES {escape_cmake_str(x)})")

print("")
print("unset(Tensorflow_LIBRARY_OPTIONS)")
for x in link_flags:
    print(f"list(APPEND Tensorflow_LIBRARY_OPTIONS {escape_cmake_str(x)})")

print("")
print(f"set(Tensorflow_internal_LIBRARY {escape_cmake_str(internal.__file__)})")
