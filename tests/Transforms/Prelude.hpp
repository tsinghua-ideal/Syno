#include <array>
#include <map>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Functional.hpp"

#ifdef KAS_USE_HALIDE
#include "KAS/CodeGen/HalideGen.hpp"
#endif


namespace kas {

// Here, we verify the shape, iterator and codegen semantics of each transform.
class transforms_tests: public ::testing::Test {
protected:
    BindingContext ctx = BindingContext({"H=4", "W=4"}, {"c=2"});
    Size sizeH = ctx.getSize("H");
    Size sizeW = ctx.getSize("W");
    Size sizeC = ctx.getSize("c");
    Iterator itH { 0, sizeH }, itW { 1, sizeW }, itCH { 2, sizeC * sizeH };
    Dimension dimH { &itH }, dimW { &itW }, dimCH { &itCH };
    transforms_tests() {
        ctx.debug();
    }
};

} // namespace kas
