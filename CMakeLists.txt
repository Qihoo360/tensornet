cmake_minimum_required(VERSION 3.24)

project(tensornet)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  message(FATAL_ERROR "Tensornet only works on Linux x86_64, but found ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Makefiles$")
  message(FATAL_ERROR "Tensornet only tested with Makefile generators.")
endif()

include(cmake/AddModulePath.cmake)
# include_watch(DebugUtils)
include_watch(FetchContent)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "Debug"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS # chmod 755
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)

# build brpc from source
set(_brpc_patches
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/patches/02-fix-brpc-compile.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/patches/03-fix-brpc-cxx-flags.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/patches/04-boringssl.patch")

find_package(Patch REQUIRED)

set(_brpc_patch_command)
foreach(pf ${_brpc_patches})
  list(
    APPEND
    _brpc_patch_command
    &&
    "${Patch_EXECUTABLE}"
    -p
    1
    -i
    "${pf}")
  watch_for_configuring_if_needed("${pf}")
endforeach()
list(REMOVE_AT _brpc_patch_command 0)

set(_brpc_source_url https://github.com/apache/brpc/archive/1.5.0.tar.gz)
if(DEFINED ENV{URL_MIRROR_github_com} AND NOT "$ENV{URL_MIRROR_github_com}" STREQUAL "")
  string(REPLACE "https://github.com/" "$ENV{URL_MIRROR_github_com}" _brpc_source_url "${_brpc_source_url}")
endif()

# hash for brpc source tgz, compute with cmd: openssl dgst -sha3-512 path
set(_brpc_source_hash
    "\
6f6891ea2a35313f073e68c9857a9e0ea3d70f5b95335e4229c3789dec0395ed\
eeb7a652736e7289e4269fa43ac698cf22b9589690255addd2242cdf74b6157f")

FetchContent_Declare(
  brpc
  URL ${_brpc_source_url}
  URL_HASH SHA3_512=${_brpc_source_hash}
  PATCH_COMMAND ${_brpc_patch_command} OVERRIDE_FIND_PACKAGE)

if(DEFINED brpc_SOURCE_DIR AND EXISTS "${brpc_SOURCE_DIR}")
  # re-generate the brpc source
  foreach(pf ${_brpc_patches})
    if("${pf}" IS_NEWER_THAN "${brpc_SOURCE_DIR}")
      execute_process(COMMAND "${CMAKE_COMMAND}" -E rm -rf --
                              "${brpc_SOURCE_DIR}/../brpc-subbuild/brpc-populate-prefix/src/brpc-populate-stamp")
      break()
    endif()
  endforeach()
endif()

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_package(Tensorflow REQUIRED)
find_package(Protobuf REQUIRED) # for compiling .proto files
find_package(
  Boost
  COMPONENTS iostreams
  REQUIRED)
find_package(MPI REQUIRED)

# options for brpc
set(GFLAGS_STATIC TRUE)
set(BUILD_SHARED_LIBS 1)
set(BUILD_BRPC_TOOLS OFF)
find_package(brpc REQUIRED)

# Tensorflow::framework is a system library, use user libraries to avoid ssl and protobuf conflicts
target_link_libraries(BUTIL_LIB Tensorflow::boringssl)
target_link_libraries(PROTO_LIB Tensorflow::protobuf)
target_link_libraries(SOURCES_LIB Tensorflow::boringssl)

# skip protoc-gen-mcpack
set_target_properties(protoc-gen-mcpack PROPERTIES EXCLUDE_FROM_ALL TRUE)

target_include_directories(brpc-static INTERFACE ${brpc_BINARY_DIR}/output/include)
target_include_directories(brpc-static SYSTEM INTERFACE ${GFLAGS_INCLUDE_PATH})

# build brpc-shared to verify all symboles are resolved
target_link_libraries(brpc-shared Tensorflow::protobuf Tensorflow::boringssl)
target_link_options(brpc-shared PRIVATE LINKER:--no-undefined)

# setup tensornet targets
set(TN_COMMON_TARGET_PROPERTIES
    #
    # use stdc++ 14
    CXX_STANDARD
    14
    CXX_EXTENSIONS
    FALSE
    #
    # enable LTO
    INTERPROCEDURAL_OPTIMIZATION
    TRUE
    #
    # remove the absolute toolchain lib dir from RPATH
    INSTALL_REMOVE_ENVIRONMENT_RPATH
    TRUE)
set(TN_COMMON_TARGET_PROPERTIES_LINT_OK)

set(PROTO_SOURCES core/ps_interface/ps_server.proto)

set(CC_SOURCES
    core/kernels/bn_table_ops.cc
    core/kernels/data/balance_dataset_ops.cc
    core/kernels/dense_table_ops.cc
    core/kernels/sparse_table_ops.cc
    core/ops/balance_dataset_ops.cc
    core/ops/bn_table_ops.cc
    core/ops/dense_table_ops.cc
    core/ops/sparse_table_ops.cc
    core/ps/optimizer/ada_grad_kernel.cc
    core/ps/optimizer/adam_kernel.cc
    core/ps/optimizer/data_struct.cc
    core/ps/optimizer/ftrl_kernel.cc
    core/ps/optimizer/optimizer.cc
    core/ps/ps_cluster.cc
    core/ps/ps_remote_server.cc
    core/ps/ps_service_impl.cc
    core/ps/table/bn_table.cc
    core/ps/table/dense_table.cc
    core/ps/table/sparse_table.cc
    core/ps/ps_local_server.cc
    core/utility/file_io.cc
    core/utility/mpi_manager.cc
    core/utility/net_util.cc)

set(CC_HEADERS
    core/kernels/resource_var_wrapper.h
    core/kernels/data/balance_dataset_ops.h
    core/ps_interface/ps_raw_interface.h
    core/ps/ps_remote_server.h
    core/ps/optimizer/ada_grad_kernel.h
    core/ps/optimizer/optimizer_kernel.h
    core/ps/optimizer/data_struct.h
    core/ps/optimizer/optimizer.h
    core/ps/optimizer/adam_kernel.h
    core/ps/optimizer/ftrl_kernel.h
    core/ps/ps_cluster.h
    core/ps/ps_service_impl.h
    core/ps/ps_server_interface.h
    core/ps/table/sparse_table.h
    core/ps/table/bn_table.h
    core/ps/table/dense_table.h
    core/ps/ps_local_server.h
    core/public/version.h
    core/utility/semaphore.h
    core/utility/file_io.h
    core/utility/net_util.h
    core/utility/allocator.h
    core/utility/mpi_manager.h
    core/utility/random.h
    core/utility/fix_redef.h)

# this target holds tensornet dependents that should be hiden to _pywrap_tn, but public to test
add_library(tensornet-interface INTERFACE)
target_link_libraries(
  tensornet-interface INTERFACE Tensorflow::framework Tensorflow::internal Boost::iostreams MPI::MPI_CXX
                                $<TARGET_PROPERTY:brpc-shared,INTERFACE_LINK_LIBRARIES>)

# the main TN dynamic library
add_library(tensornet SHARED ${CC_SOURCES} ${CC_HEADERS} ${PROTO_SOURCES})
protobuf_generate(TARGET tensornet)
set_target_properties(
  tensornet
  PROPERTIES ${TN_COMMON_TARGET_PROPERTIES} ${TN_COMMON_TARGET_PROPERTIES_LINT_OK}
             # set RPATH to search libmpi.so, libstdc++, and etc. in conda environments
             INSTALL_RPATH "$ORIGIN/../../../..")
target_compile_options(
  tensornet
  PUBLIC -march=x86-64-v2
         -mavx
         -mtune=broadwell
         # workround tensorflow issue #19747, tf operators using RefCount should be compiled with -DNDEBUG and -O[123]
         # should be fixed from tf 2.8 by
         # https://github.com/tensorflow/tensorflow/commit/895943537b80e8cea0e1ba71c6dd4c2d651b84d0
         -DNDEBUG
         $<$<CONFIG:Debug,RelWithDebInfo>:-g>
         $<$<CONFIG:Debug>:-O1>
         $<$<CONFIG:MinSizeRel>:-O2>
         $<$<CONFIG:Release,RelWithDebInfo>:-O3>
         $<$<CONFIG:Release,MinSizeRel>:-Os>)
target_include_directories(tensornet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
# hide all linked dynamic libraries for py wrappers
target_link_libraries(
  tensornet
  PRIVATE tensornet-interface
  PUBLIC brpc-static)
target_link_options(tensornet PUBLIC -fuse-ld=gold $<$<CONFIG:Release,MinSizeRel,RelWithDebInfo>:-s> PRIVATE
                    LINKER:-as-needed)

# this target is used for test, publish tensornet and all the dynamic dependencies
add_library(tensornet-full-link INTERFACE)
target_link_libraries(tensornet-full-link INTERFACE tensornet tensornet-interface)

# python wrapper
python3_add_library(_pywrap_tn MODULE WITH_SOABI core/main/py_wrapper.cc)

set_target_properties(
  _pywrap_tn
  PROPERTIES ${TN_COMMON_TARGET_PROPERTIES} ${TN_COMMON_TARGET_PROPERTIES_VAKE_VALUE}
             # set RPATH to search libtensornet.so, libstdc++, and etc. in conda environments as a python package
             INSTALL_RPATH "$ORIGIN:$ORIGIN/../../../..")

target_include_directories(_pywrap_tn PRIVATE $<TARGET_PROPERTY:tensornet,INCLUDE_DIRECTORIES>)
target_link_libraries(_pywrap_tn PRIVATE Tensorflow::pybind11 tensornet)
target_link_options(_pywrap_tn PRIVATE LINKER:-as-needed)
if(DEFINED ENV{CONDA_PREFIX})
  # The specs for conda gcc has a rule that the link command always appends $PREFIX/lib at the end of rpath, see the
  # install script of the package. After build, the result rpath on the target so is "<user-rpaths>::$PREFIX/lib", then
  # cmake install cannot remove the ':$PREFIX/lib' suffix even if set INSTALL_REMOVE_ENVIRONMENT_RPATH. So here we
  # *prepend* it to rpath first, the result rpath turns to "$RPATH/lib:<user-rpaths>:". Then cmake install can correctly
  # replace it by ${INSTALL_RPATH} if set INSTALL_REMOVE_ENVIRONMENT_RPATH.
  #
  # Ref: https://github.com/conda-forge/ctng-compilers-feedstock/blob/main/recipe/install-gcc.sh#L169
  target_link_options(_pywrap_tn PRIVATE LINKER:-rpath,$ENV{CONDA_PREFIX}/lib)
endif()

add_custom_target(targets_to_install COMMENT "Collect all installable targets.")
add_dependencies(targets_to_install tensornet _pywrap_tn)

# python wrapper for other versions
if(DEFINED ENV{PIXI_EXE})
  # build py wrapper for python ${ver}
  function(build_pywrap ver)
    if(NOT ver)
      return()
    endif()

    string(REPLACE "." "" pixi_env "py${ver}")

    execute_process(
      COMMAND
        "$ENV{PIXI_EXE}" run -e ${pixi_env} "${CMAKE_COMMAND}" -DPY_VER=${ver}
        "-DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${pixi_env}-parameters.cmake" -P
        "${CMAKE_CURRENT_LIST_DIR}/cmake/PrintPythonModule.cmake" COMMAND_ERROR_IS_FATAL ANY)

    include_watch("${CMAKE_CURRENT_BINARY_DIR}/${pixi_env}-parameters.cmake")

    python3_add_library(_pywrap_tn_${ver} MODULE WITH_SOABI core/main/py_wrapper.cc)
    target_link_libraries(_pywrap_tn_${ver} PRIVATE Tensorflow::pybind11 tensornet)

    get_target_property(link_libs _pywrap_tn_${ver} LINK_LIBRARIES)
    list(REMOVE_ITEM link_libs Python3::Module)

    set_target_properties(
      _pywrap_tn_${ver}
      PROPERTIES ${TN_COMMON_TARGET_PROPERTIES} ${TN_COMMON_TARGET_PROPERTIES_VAKE_VALUE}
                 OUTPUT_NAME _pywrap_tn
                 LINK_LIBRARIES "${link_libs}"
                 SUFFIX ".${Python${ver}_SOABI}"
                 INSTALL_RPATH "$ORIGIN:$ORIGIN/../../../..")

    target_include_directories(_pywrap_tn_${ver} PRIVATE ${Python${ver}_INCLUDE_DIRS}
                                                         $<TARGET_PROPERTY:tensornet,INCLUDE_DIRECTORIES>)
    target_link_options(_pywrap_tn_${ver} PRIVATE LINKER:-as-needed)

    if(DEFINED ENV{CONDA_PREFIX})
      target_link_options(_pywrap_tn_${ver} PRIVATE LINKER:-rpath,$ENV{CONDA_PREFIX}/lib)
    endif()

    install(TARGETS _pywrap_tn_${ver} LIBRARY DESTINATION tensornet/core
                                              PERMISSIONS ${CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS})
    add_dependencies(targets_to_install _pywrap_tn_${ver})
  endfunction()

  foreach(ver IN ITEMS 3.5 3.6 3.7)
    build_pywrap(${ver})
  endforeach()
endif()

install(TARGETS tensornet LIBRARY DESTINATION tensornet/core PERMISSIONS ${CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS})
install(TARGETS _pywrap_tn LIBRARY DESTINATION tensornet/core
                                   PERMISSIONS ${CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS})
install(
  DIRECTORY tensornet/
  DESTINATION tensornet
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ # chmod 644
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "gen_*_ops.py" EXCLUDE
  PATTERN "gen_*_ops.py" EXCLUDE
  PATTERN "__pycache__" EXCLUDE)

# generate op python wrappers
set(TN_OP_WRAPPERS)
foreach(op IN ITEMS balance_dataset_ops bn_table_ops dense_table_ops sparse_table_ops)
  if(op STREQUAL "balance_dataset_ops")
    add_library(tfkernel-${op} SHARED core/kernels/data/${op}_dummy.cc core/ops/${op}.cc)
    target_link_libraries(tfkernel-${op} PRIVATE Tensorflow::framework Tensorflow::internal)
  else()
    add_library(tfkernel-${op} SHARED core/kernels/${op}_dummy.cc core/ops/${op}.cc)
    target_link_libraries(tfkernel-${op} PRIVATE Tensorflow::framework)
  endif()
  set_target_properties(tfkernel-${op} PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS FALSE)
  target_include_directories(tfkernel-${op} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_options(tfkernel-${op} PRIVATE LINKER:--no-undefined)

  set(_wrapper "${PROJECT_BINARY_DIR}/tensornet/core/gen_${op}.py")
  add_custom_command(
    OUTPUT ${_wrapper}
    COMMENT "Generating python wrapper for ${op} op ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/tensornet/core/"
    COMMAND ${CMAKE_COMMAND} -E env ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tf_gen_op_wrapper.py"
            $<TARGET_FILE:tfkernel-${op}> $<TARGET_PROPERTY:tfkernel-${op},SOURCES> > ${_wrapper}
    DEPENDS tf_gen_op_wrapper.py
    VERBATIM COMMAND_EXPAND_LISTS)
  list(APPEND TN_OP_WRAPPERS ${_wrapper})
endforeach()

add_custom_target(
  gen_all_tf_ops ALL
  COMMENT "Generate all python wrappers."
  DEPENDS ${TN_OP_WRAPPERS})

install(
  FILES ${TN_OP_WRAPPERS}
  DESTINATION tensornet/core
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ # chmod 644
)

add_dependencies(targets_to_install gen_all_tf_ops)

include_watch(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
  add_custom_command(
    TARGET targets_to_install
    POST_BUILD
    COMMENT "Copy core library for local integration test."
    COMMAND ${CMAKE_COMMAND} -E env CMAKE_INSTALL_MODE=SYMLINK ${CMAKE_COMMAND} --install "${PROJECT_BINARY_DIR}"
            --prefix "${PROJECT_BINARY_DIR}"
    VERBATIM COMMAND_EXPAND_LISTS)
endif()
