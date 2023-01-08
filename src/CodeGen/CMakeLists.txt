add_library(codegen-lib STATIC HalideGen.cpp)
target_link_libraries(codegen-lib PRIVATE Halide::Halide)
