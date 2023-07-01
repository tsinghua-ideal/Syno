# Function for adding a test
function(add_kas_test TEST_NAME TEST_ENTRY_PATH)
    cmake_parse_arguments(ADD_KAS_TEST "" "ADDITIONAL_LIBS" "" ${ARGN})
    add_executable(${TEST_NAME} ${TEST_ENTRY_PATH})
    get_property(KAS_ALL_LIBS GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY)
    target_link_libraries(${TEST_NAME}
            gtest gtest_main ${KAS_ALL_LIBS} ${ADD_KAS_TEST_ADDITIONAL_LIBS} pthread)
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

# Tests
add_subdirectory(Core)
add_subdirectory(Search)
add_subdirectory(Semantics)
add_subdirectory(Transforms)
add_kas_test(codegen-tests CodeGen.cpp)
add_kas_test(utils-tests Utils.cpp)
