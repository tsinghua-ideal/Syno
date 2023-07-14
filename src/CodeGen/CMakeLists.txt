add_library(codegen-lib STATIC Common.cpp GraphvizGen.cpp HalideGen.cpp Kernel.cpp PyTorchGen.cpp)
target_link_libraries(codegen-lib PRIVATE Halide::Halide)
