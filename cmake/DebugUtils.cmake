include_guard(GLOBAL) # require 3.10

include(CMakePrintSystemInformation)

if(CMAKE_VERSION VERSION_LESS "3.19")
  message(FATAL_ERROR "FindTensorflow module requires CMake >= 3.19")
endif()

# Get all properties that cmake supports
execute_process(COMMAND "${CMAKE_COMMAND}" --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST
                                                                                COMMAND_ERROR_IS_FATAL ANY)

# Convert command output into a CMake list
string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)

# remove properties cannot got from targets
list(REMOVE_ITEM CMAKE_PROPERTY_LIST LOCATION VS_DEPLOYMENT_LOCATION MACOSX_PACKAGE_LOCATION "LOCATION_<CONFIG>")

include_watch(CMakePrintHelpers)

# Print all properties of the target ${tgt}
function(print_target_properties tgt)
  if(NOT TARGET ${tgt})
    message(WARNING "There is no target named '${tgt}'")
    return()
  endif()

  set(prop_list)
  foreach(prop ${CMAKE_PROPERTY_LIST})
    string(REPLACE "<CONFIG>" "$<CONFIG>" prop ${prop})
    get_target_property(propval ${tgt} ${prop})
    if(propval)
      list(APPEND prop_list "${prop}")
    endif()
  endforeach()
  cmake_print_properties(TARGETS ${tgt} PROPERTIES ${prop_list})
endfunction()

# Print all defined and imported targets for dir
function(print_directory_targets)
  if(ARGC EQUAL 0)
    set(dir "${CMAKE_CURRENT_SOURCE_DIR}")
  else()
    set(dir "${ARGV0}")
  endif()
  if(NOT IS_DIRECTORY "${dir}")
    message(WARNING "'${dir}' is not a directory")
    return()
  endif()
  cmake_print_properties(DIRECTORIES ${dir} PROPERTIES BUILDSYSTEM_TARGETS IMPORTED_TARGETS)
endfunction()
