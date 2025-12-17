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
import sys
import sysconfig
from distutils.sysconfig import get_python_lib
import google.protobuf

def escape_cmake_str(s):
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("$", "\\$")
    return f'"{s}"'

# Get protobuf version
proto_version = google.protobuf.__version__

# Get protobuf include and lib paths
python_lib = get_python_lib()
# Use the standard include path that contains google/protobuf
proto_include = os.path.join(sys.prefix, 'include')

proto_lib_dir = python_lib

# Find protoc executable
# Try multiple common locations
protoc_candidates = [
    os.path.join(os.path.dirname(sysconfig.get_config_var('BINDIR')), 'protoc') if sysconfig.get_config_var('BINDIR') else None,
    os.path.join(sysconfig.get_config_var('BINDIR') or '', 'protoc'),
    'protoc'  # fallback to PATH
]

protoc_exe = None
for candidate in protoc_candidates:
    if candidate and os.path.exists(candidate):
        protoc_exe = candidate
        break
    elif candidate:
        # Check if it's in PATH
        try:
            import shutil
            if shutil.which(candidate):
                protoc_exe = candidate
                break
        except:
            pass

if not protoc_exe:
    # Last resort: try to find it in common locations
    common_paths = [
        os.path.join(sys.prefix, 'bin', 'protoc'),
        os.path.join(sys.base_prefix, 'bin', 'protoc'),
        '/usr/bin/protoc',
        '/usr/local/bin/protoc'
    ]
    for path in common_paths:
        if os.path.exists(path):
            protoc_exe = path
            break

if not protoc_exe:
    raise RuntimeError("Could not find protoc executable")

# Find protobuf library files
proto_libs = []
proto_lib_names = ['libprotobuf.so', 'libprotobuf.a', 'protobuf.lib']

# Search for protobuf library in the lib directory
lib_dirs = [
    proto_lib_dir,
    os.path.join(python_lib, '..', 'lib'),
    os.path.join(sys.prefix, 'lib'),
    os.path.join(sys.base_prefix, 'lib')
]

for lib_dir in lib_dirs:
    if os.path.exists(lib_dir):
        for lib_name in proto_lib_names:
            lib_path = os.path.join(lib_dir, lib_name)
            if os.path.exists(lib_path):
                proto_libs.append(lib_path)

# If no library found, use the standard library name
if not proto_libs:
    proto_libs = ['protobuf']

# Find protoc library specifically
protoc_lib = None
for lib_dir in lib_dirs:
    if os.path.exists(lib_dir):
        for lib_name in ['libprotoc.so', 'libprotoc.a']:
            lib_path = os.path.join(lib_dir, lib_name)
            if os.path.exists(lib_path):
                protoc_lib = lib_path
                break
    if protoc_lib:
        break

# Output CMake variables
print(f"set(Protobuf_VERSION {escape_cmake_str(proto_version)})")
print(f"set(Protobuf_INCLUDE_DIRS {escape_cmake_str(proto_include)})")
print(f"set(Protobuf_LIBRARIES {escape_cmake_str(proto_libs[0])})")  # Use the first found library
print(f"set(Protobuf_PROTOC_EXECUTABLE {escape_cmake_str(protoc_exe)})")
if protoc_lib:
    print(f"set(Protobuf_PROTOC_LIBRARY {escape_cmake_str(protoc_lib)})")
print(f"set(Protobuf_FOUND TRUE)")
