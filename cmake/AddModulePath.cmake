include_guard(GLOBAL) # require 3.10

if(NOT COMMAND add_module_dir)
  # Add dir into CMAKE_MODULE_PATH
  function(add_module_dir dir)
    if(NOT dir)
      return()
    endif()

    cmake_path(ABSOLUTE_PATH dir NORMALIZE)
    list(FIND CMAKE_MODULE_PATH "${dir}" idx)
    if(${idx} EQUAL -1)
      message(STATUS "Add the module path: ${dir}")
      list(APPEND CMAKE_MODULE_PATH "${dir}")
      set(CMAKE_MODULE_PATH
          "${CMAKE_MODULE_PATH}"
          PARENT_SCOPE)
    endif()
  endfunction()
endif()

add_module_dir("${CMAKE_CURRENT_LIST_DIR}")
include(IncludeWithWatch)
watch_me_for_configuring_if_needed()
