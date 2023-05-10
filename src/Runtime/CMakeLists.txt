add_executable(RuntimeGen RuntimeGen.cpp)
target_link_libraries(RuntimeGen PRIVATE Halide::Halide)

set(kas_halide_runtime_path ${CMAKE_CURRENT_BINARY_DIR}/kas-halide-runtime.a)

add_custom_target(
    kas_halide_runtime
    DEPENDS RuntimeGen
    COMMAND RuntimeGen ${kas_halide_runtime_path}
    BYPRODUCTS ${kas_halide_runtime_path}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(kas-halide-runtime-lib STATIC IMPORTED GLOBAL)
add_dependencies(kas-halide-runtime-lib kas_halide_runtime)
set_target_properties(
    kas-halide-runtime-lib
    PROPERTIES IMPORTED_LOCATION ${kas_halide_runtime_path}
)

add_library(runtime-lib STATIC Loader.cpp)
target_link_libraries(runtime-lib PRIVATE kas-halide-runtime-lib ${TORCH_LIBRARIES} CUDA::cudart ${CMAKE_DL_LIBS})
