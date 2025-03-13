include_guard(GLOBAL) # require 3.10

# watch_for_configuring_if_needed: if needed, the file should be watched as a configure-time dependency
function(watch_for_configuring_if_needed file)
  if(NOT file OR file STREQUAL "NOTFOUND")
    return()
  endif()

  cmake_path(ABSOLUTE_PATH file NORMALIZE)

  list(FIND CMAKE_CONFIGURE_DEPENDS "${file}" idx)
  if(idx GREATER_EQUAL 0)
    return()
  endif()

  string(FIND "${file}" "${CMAKE_SOURCE_DIR}/" __M_P_IN_SRC)
  string(FIND "${file}" "${CMAKE_BINARY_DIR}/" __M_P_IN_BIN)
  string(FIND "${file}" "${CMAKE_CURRENT_BINARY_DIR}/" __M_P_IN_CBIN)
  if(IS_DIRECTORY "$ENV{PIXI_PROJECT_ROOT}")
    string(FIND "${file}" "$ENV{PIXI_PROJECT_ROOT}/.pixi/" m_p_in_pixi)
  else()
    set(m_p_in_pixi -1)
  endif()

  if(__M_P_IN_SRC GREATER_EQUAL 0
     AND __M_P_IN_BIN LESS 0
     AND __M_P_IN_CBIN LESS 0
     AND m_p_in_pixi LESS 0)
    # watch file when under the top src dir but not under the top bin dir and the current bin dir and pixi env dir
    list(APPEND CMAKE_CONFIGURE_DEPENDS "${file}")
    message(STATUS "Add the configure dependency: ${file}")
    set(CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CONFIGURE_DEPENDS}"
        PARENT_SCOPE)
    set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${CMAKE_CONFIGURE_DEPENDS}")
  endif()
endfunction()

# WATCH_ME_FOR_CONFIGURING_IF_NEEDED: try add the current list file as configure depends
macro(WATCH_ME_FOR_CONFIGURING_IF_NEEDED)
  watch_for_configuring_if_needed("${CMAKE_CURRENT_LIST_FILE}")
endmacro()

# result_var_of_include_args: parse result var name for include() arguments
function(result_var_of_include_args)
  list(FIND ARGV RESULT_VARIABLE idx)
  if(idx LESS 0)
    set(${ARGV0}
        "FALSE"
        PARENT_SCOPE)
  else()
    math(EXPR idx "${idx} + 1")
    list(GET ARGV ${idx} ${ARGV0})
    set(${ARGV0}
        "${ARGV0}"
        PARENT_SCOPE)
  endif()
endfunction()

# INCLUDE_WATCH(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>] [NO_POLICY_SCOPE]) Watch the included file if the under
# the source dir.
macro(INCLUDE_WATCH)
  result_var_of_include_args(__include_watch_RESULT_VAR_NAME ${ARGV})
  if(__include_watch_RESULT_VAR_NAME)
    include(${ARGV})
    watch_for_configuring_if_needed("${__include_watch_RESULT_VAR_NAME}")
  else()
    include(${ARGV} RESULT_VARIABLE __include_watch_RESULT)
    watch_for_configuring_if_needed("${__include_watch_RESULT}")
    unset(__include_watch_RESULT)
  endif()
  unset(__include_watch_RESULT_VAR_NAME)
endmacro()

watch_me_for_configuring_if_needed()

if(IS_DIRECTORY "$ENV{PIXI_PROJECT_ROOT}")
  watch_for_configuring_if_needed("$ENV{PIXI_PROJECT_ROOT}/pixi.lock")
endif()
