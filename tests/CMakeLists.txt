# Function for adding a test
function(add_kas_test TEST_NAME TEST_ENTRY_PATH)
    add_executable(${TEST_NAME} ${TEST_ENTRY_PATH})
    get_property(KAS_ALL_LIBS GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY)
    target_link_libraries(${TEST_NAME}
            gtest gtest_main pthread ${KAS_ALL_LIBS})
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

# Tests
add_subdirectory(Core)
add_subdirectory(Transforms)
add_kas_test(codegen-tests CodeGen.cpp)
add_kas_test(search-tests Search.cpp)
add_kas_test(utils-tests Utils.cpp)
