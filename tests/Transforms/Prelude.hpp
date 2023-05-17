#include <array>
#include <map>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Functional.hpp"


namespace kas {

// Here, we verify the shape, iterator and codegen semantics of each transform.
class transforms_tests: public ::testing::Test {
protected:
    using SizeName = BindingContext::Metadata;
    BindingContext ctx { std::vector<SizeName> { SizeName("H", 128), SizeName("W", 128) }, std::vector<SizeName> { SizeName("c", 5) } };
    Size sizeH = ctx.getSinglePrimaryVariableSize(0);
    Size sizeW = ctx.getSinglePrimaryVariableSize(1);
    Size sizeC = ctx.getSingleCoefficientVariableSize(0);
    Iterator itH { 0, sizeH }, itW { 1, sizeW }, itCH { 2, sizeC * sizeH };
    Dimension dimH { &itH }, dimW { &itW }, dimCH { &itCH };
};

} // namespace kas
