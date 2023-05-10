pybind11_add_module(kas_cpp_bindings Bindings.cpp)
target_link_libraries(kas_cpp_bindings PRIVATE ${KAS_ALL_LIBS})
target_compile_definitions(kas_cpp_bindings
                           PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})
install(TARGETS kas_cpp_bindings LIBRARY DESTINATION .)

pybind11_add_module(kas_runtime Runtime.cpp)
target_link_libraries(kas_runtime PRIVATE ${KAS_ALL_LIBS} ${TORCH_LIBRARIES})
target_compile_definitions(kas_runtime
                           PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})
install(TARGETS kas_runtime LIBRARY DESTINATION .)
