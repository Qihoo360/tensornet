find_package(GTest REQUIRED)
include_watch(GoogleTest)

find_package(
  Python3
  COMPONENTS Development.Embed
  REQUIRED)

add_executable(tn_test ps/optimizer/optimizer_kernel_test.cc ps/table/sparse_table_test.cc utility/allocator_test.cc)
set_target_properties(tn_test PROPERTIES ${TN_COMMON_TARGET_PROPERTIES} ${TN_COMMON_TARGET_PROPERTIES_LINT_OK})
target_link_libraries(tn_test PRIVATE tensornet-full-link GTest::Main)

# Force linking embed python lib is required by _pywrap_tensorflow_internal.so python module. Before cmake 4.0, portable
# linke option syntax `LINKER:-option` cannot be used in target_link_libraries(...), so manually collect the options in
# order with target_link_options(...)
target_link_options(tn_test PRIVATE ${Python3_LINK_OPTIONS} LINKER:-no-as-needed ${Python3_LIBRARIES} LINKER:-as-needed)

# add ut
gtest_discover_tests(tn_test)

# add local integration test
add_test(
  NAME integration-test
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=. "${PROJECT_SOURCE_DIR}/examples/integration-test.py"
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}" COMMAND_EXPAND_LISTS)

if(CMAKE_GENERATOR MATCHES "Makefiles")
  add_custom_target(
    test_autorun ALL
    COMMAND eval if test 1x = "$(SKIP_RUN_TESTS)x" "\;" then ${CMAKE_COMMAND} -E cmake_echo_color "--switch=$(COLOR)"
            --cyan "\"Skip running tests...\"" "\;" fi
    COMMAND eval if test 1x != "$(SKIP_RUN_TESTS)x" "\;" then ${CMAKE_COMMAND} --build . --target test --
            CLICOLOR=$(COLOR) "\;" fi
    COMMENT "Auto run tests ..."
    VERBATIM COMMAND_EXPAND_LISTS)
else()
  add_custom_target(
    test_autorun ALL
    COMMAND ${CMAKE_COMMAND} --build . --target test -- CLICOLOR=$(COLOR)
    COMMENT "Auto run tests ..."
    VERBATIM COMMAND_EXPAND_LISTS)
endif()

add_dependencies(test_autorun tn_test targets_to_install)
