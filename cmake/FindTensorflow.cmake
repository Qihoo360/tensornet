# Distributed under the Apache.  See accompanying file LICENSE for details.

#[=======================================================================[.rst:
FindTensorflow
-------

Finds the Tensorflow platform libraries.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``Tensorflow::framework``
  The main tensorflow dynamic library

``Tensorflow::protobuf``
``Tensorflow::boringssl``
``Tensorflow::pybind11``

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Tensorflow_FOUND``
  True if the system has the tensorflow installed.
``Tensorflow_VERSION``
  The version of the tensorflow which was found.
``Tensorflow_ROOT_DIR``
  The installed dir of the tensorflow which was found.
``Tensorflow_INCLUDE_DIRS``
  Include directories needed to use Foo.
``Tensorflow_LIBRARIES``
  Libraries needed to link to Foo.
``Tensorflow_DEFINITIONS``
``Tensorflow_COMPILE_OPTIONS``
``Tensorflow_LIBRARY_DIRS``
``Tensorflow_LIBRARY_OPTIONS``

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Tensorflow_INCLUDE_DIR``
  The directory from ``tf.sysconfig.get_include()``.
``Tensorflow_LIBRARY``
  The path to the tensorflow_framework dynamic library.

#]=======================================================================]

if(CMAKE_VERSION VERSION_LESS "3.19")
  message(FATAL_ERROR "FindTensorflow module requires CMake >= 3.19")
endif()

# added to configuring dependents
include(IncludeWithWatch)
watch_me_for_configuring_if_needed()

if(NOT Python3_EXECUTABLE)
  if(Tensorflow_FIND_QUIETLY)
    find_package(Python3 QUIETLY REQUIRED COMPONENTS Interpreter)
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
  endif()
endif()

# # Set LD_LIBRARY_PATH to ensure TensorFlow can find the correct libstdc++
# if(DEFINED ENV{CONDA_PREFIX})
#   if(DEFINED ENV{LD_LIBRARY_PATH})
#     set(ENV{LD_LIBRARY_PATH} "$ENV{CONDA_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}")
#   else()
#     set(ENV{LD_LIBRARY_PATH} "$ENV{CONDA_PREFIX}/lib")
#   endif()
# endif()

execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/find-tensorflow.py"
                OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-parameters.cmake" COMMAND_ERROR_IS_FATAL ANY)

watch_for_configuring_if_needed("${CMAKE_CURRENT_LIST_DIR}/find-tensorflow.py")
include_watch("${CMAKE_CURRENT_BINARY_DIR}/tensorflow-parameters.cmake")

add_library(Tensorflow::framework INTERFACE IMPORTED GLOBAL)
target_include_directories(Tensorflow::framework INTERFACE ${Tensorflow_INCLUDE_DIRS})
target_compile_definitions(Tensorflow::framework INTERFACE ${Tensorflow_DEFINITIONS})
target_compile_options(Tensorflow::framework INTERFACE ${Tensorflow_COMPILE_OPTIONS})
target_link_directories(Tensorflow::framework INTERFACE ${Tensorflow_LIBRARY_DIRS})
target_link_options(Tensorflow::framework INTERFACE ${Tensorflow_LIBRARY_OPTIONS})
target_link_libraries(Tensorflow::framework INTERFACE ${Tensorflow_LIBRARY})

add_library(Tensorflow::protobuf INTERFACE IMPORTED GLOBAL)
set_target_properties(Tensorflow::protobuf PROPERTIES SYSTEM OFF) # to override the protobuf headers from system
target_include_directories(Tensorflow::protobuf INTERFACE ${Tensorflow_INCLUDE_DIRS})
target_compile_definitions(Tensorflow::protobuf INTERFACE ${Tensorflow_DEFINITIONS})
target_compile_options(Tensorflow::protobuf INTERFACE ${Tensorflow_COMPILE_OPTIONS})
target_link_directories(Tensorflow::protobuf INTERFACE ${Tensorflow_LIBRARY_DIRS})
target_link_options(Tensorflow::protobuf INTERFACE ${Tensorflow_LIBRARY_OPTIONS})
target_link_libraries(Tensorflow::protobuf INTERFACE ${Tensorflow_LIBRARY})

find_file(
  Tensorflow_boringssl_INCLUDE
  NAMES is_boringssl.h
  HINTS ${Tensorflow_INCLUDE_DIR}/external/boringssl/src/include/openssl/ NO_CACHE REQUIRED
  NO_DEFAULT_PATH)
cmake_path(GET Tensorflow_boringssl_INCLUDE PARENT_PATH Tensorflow_boringssl_INCLUDE)
cmake_path(GET Tensorflow_boringssl_INCLUDE PARENT_PATH Tensorflow_boringssl_INCLUDE)

add_library(Tensorflow::boringssl INTERFACE IMPORTED GLOBAL)
set_target_properties(Tensorflow::boringssl PROPERTIES SYSTEM OFF) # to override the openssl headers from system
target_include_directories(Tensorflow::boringssl INTERFACE ${Tensorflow_boringssl_INCLUDE})
target_compile_definitions(Tensorflow::boringssl INTERFACE ${Tensorflow_DEFINITIONS})
target_compile_options(Tensorflow::boringssl INTERFACE ${Tensorflow_COMPILE_OPTIONS})
target_link_directories(Tensorflow::boringssl INTERFACE ${Tensorflow_LIBRARY_DIRS})
target_link_options(Tensorflow::boringssl INTERFACE ${Tensorflow_LIBRARY_OPTIONS})
target_link_libraries(Tensorflow::boringssl INTERFACE ${Tensorflow_LIBRARY})

find_file(
  Tensorflow_pybind11_INCLUDE
  NAMES pybind11.h
  HINTS ${Tensorflow_INCLUDE_DIR}/external/pybind11/_virtual_includes/pybind11/pybind11 NO_CACHE REQUIRED
  NO_DEFAULT_PATH)
cmake_path(GET Tensorflow_pybind11_INCLUDE PARENT_PATH Tensorflow_pybind11_INCLUDE)
cmake_path(GET Tensorflow_pybind11_INCLUDE PARENT_PATH Tensorflow_pybind11_INCLUDE)

add_library(Tensorflow::pybind11 INTERFACE IMPORTED GLOBAL)
set_target_properties(Tensorflow::pybind11 PROPERTIES SYSTEM OFF) # to override the pybind11 headers from system
target_include_directories(Tensorflow::pybind11 INTERFACE ${Tensorflow_pybind11_INCLUDE})
target_compile_definitions(Tensorflow::pybind11 INTERFACE ${Tensorflow_DEFINITIONS})
target_compile_options(Tensorflow::pybind11 INTERFACE ${Tensorflow_COMPILE_OPTIONS})

add_library(Tensorflow::internal INTERFACE IMPORTED GLOBAL)
target_compile_definitions(Tensorflow::internal INTERFACE ${Tensorflow_DEFINITIONS})
target_compile_options(Tensorflow::internal INTERFACE ${Tensorflow_COMPILE_OPTIONS})
target_link_libraries(Tensorflow::internal INTERFACE ${Tensorflow_internal_LIBRARY})

include_watch(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Tensorflow
  FOUND_VAR Tensorflow_FOUND
  REQUIRED_VARS
    Tensorflow_ROOT_DIR
    Tensorflow_DEFINITIONS
    Tensorflow_INCLUDE_DIRS
    Tensorflow_INCLUDE_DIR
    Tensorflow_LIBRARY_DIRS
    Tensorflow_LIBRARIES
    Tensorflow_LIBRARY
    Tensorflow_boringssl_INCLUDE
    Tensorflow_pybind11_INCLUDE
  VERSION_VAR Tensorflow_VERSION)

mark_as_advanced(Tensorflow_INCLUDE_DIR Tensorflow_LIBRARY)
