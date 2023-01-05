add_library(codegen-lib STATIC CodeGen.cpp)
target_link_libraries(codegen-lib PRIVATE Halide::Halide)
