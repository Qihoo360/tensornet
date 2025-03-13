# required variable:
#
# * PY_VER: version tag for the required python version, 37 for python 3.7, 310 for python 3.10
#
# optional variable:
#
# * OUTPUT_FILE: if defined, write to the file as a cmake script; else write debug info into stdout

if(NOT DEFINED PY_VER)
  message(
    FATAL_ERROR "PY_VER must be set for this script. For python 3.7, set PY_VER=3.7, for python 3.10, set PY_VER=3.10")
endif()

find_package(
  Python3 ${PY_VER} EXACT
  COMPONENTS Development.Module
  REQUIRED)

string(
  CONFIGURE
    "set(Python\${PY_VER}_VERSION ${Python3_VERSION})
set(Python\${PY_VER}_INCLUDE_DIRS \"${Python3_INCLUDE_DIRS}\")
set(Python\${PY_VER}_SOABI ${Python3_SOABI})
"
    OUTPUT_STRING
  ESCAPE_QUOTES)

if(DEFINED OUTPUT_FILE AND NOT OUTPUT_FILE STREQUAL "")
  file(WRITE "${OUTPUT_FILE}" "${OUTPUT_STRING}")
else()
  message("${OUTPUT_STRING}")
endif()
