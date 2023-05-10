add_library(codegen-lib STATIC GraphvizGen.cpp HalideGen.cpp Kernel.cpp)
target_link_libraries(codegen-lib PRIVATE Halide::Halide)
