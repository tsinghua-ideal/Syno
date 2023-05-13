#include <Halide.h>

#include "KAS/CodeGen/Common.hpp"


// This simply generates the Halide runtime.
int main(int argc, char **argv) {
    Halide::Target target = kas::GetHostTarget(true, true);
    if (argc == 2) {
        Halide::compile_standalone_runtime(argv[1], target);
    }
    return 0;
}
