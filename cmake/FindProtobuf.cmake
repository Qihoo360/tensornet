# Distributed under the Apache.  See accompanying file LICENSE for details.

#[=======================================================================[.rst:
FindProtobuf
-------

Finds the Protocol Buffers libraries.

This module is customized to work with pixi environments.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``protobuf::libprotobuf``
  The Protocol Buffers library

``Protobuf::protoc``
  The Protocol Buffers compiler

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Protobuf_FOUND``
  True if the system has the protobuf installed.
``Protobuf_VERSION``
  The version of the protobuf which was found.
``Protobuf_INCLUDE_DIRS``
  Include directories needed to use protobuf.
``Protobuf_LIBRARIES``
  Libraries needed to link to protobuf.
``Protobuf_PROTOC_EXECUTABLE``
  Path to the protoc compiler.

#]=======================================================================]

if(CMAKE_VERSION VERSION_LESS "3.19")
  message(FATAL_ERROR "FindProtobuf module requires CMake >= 3.19")
endif()

# added to configuring dependents
include(IncludeWithWatch)
watch_me_for_configuring_if_needed()

# If PIXI_EXE is defined, use pixi environment to find protobuf
if(DEFINED ENV{PIXI_EXE})
  if(NOT Protobuf_FIND_QUIETLY)
    message(STATUS "Using pixi protoc environment to find Protobuf...")
  endif()

  # Get protobuf information from pixi protoc environment
  execute_process(COMMAND "$ENV{PIXI_EXE}" run -e protoc python3 "${CMAKE_CURRENT_LIST_DIR}/find-protobuf.py"
                  OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/protobuf-parameters.cmake" COMMAND_ERROR_IS_FATAL ANY)

  watch_for_configuring_if_needed("${CMAKE_CURRENT_LIST_DIR}/find-protobuf.py")
  include_watch("${CMAKE_CURRENT_BINARY_DIR}/protobuf-parameters.cmake")

else()
  # Fallback to standard find_package if pixi is not available
  if(NOT Protobuf_FIND_QUIETLY)
    message(STATUS "Pixi not available, using standard find_package for Protobuf...")
  endif()

  # Use the standard FindProtobuf module
  find_package(Protobuf REQUIRED)
  return()
endif()

# Create imported targets
add_library(protobuf::libprotobuf INTERFACE IMPORTED GLOBAL)
target_include_directories(protobuf::libprotobuf INTERFACE ${Protobuf_INCLUDE_DIRS})
target_link_libraries(protobuf::libprotobuf INTERFACE ${Protobuf_LIBRARIES})

add_executable(protobuf::protoc IMPORTED GLOBAL)
set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION ${Protobuf_PROTOC_EXECUTABLE})

# Add alias for backward compatibility
add_executable(Protobuf::protoc ALIAS protobuf::protoc)

# Set PROTOC_LIB for brpc compatibility
if(DEFINED Protobuf_PROTOC_LIBRARY)
  set(PROTOC_LIB
      ${Protobuf_PROTOC_LIBRARY}
      CACHE FILEPATH "Protoc library")
  message(STATUS "Found Protobuf_PROTOC_LIBRARY: ${Protobuf_PROTOC_LIBRARY}")
else()
  message(WARNING "Protobuf_PROTOC_LIBRARY is not defined")
endif()

include_watch(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Protobuf
  FOUND_VAR Protobuf_FOUND
  REQUIRED_VARS Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protobuf_PROTOC_EXECUTABLE
  VERSION_VAR Protobuf_VERSION)

mark_as_advanced(Protobuf_INCLUDE_DIRS Protobuf_LIBRARIES Protobuf_PROTOC_EXECUTABLE)
