add_library(codegen-lib STATIC GraphvizGen.cpp HalideGen.cpp Kernel.cpp Loader.cpp)
target_link_libraries(codegen-lib PRIVATE Halide::Halide CUDA::cudart ${TORCH_LIBRARIES})
