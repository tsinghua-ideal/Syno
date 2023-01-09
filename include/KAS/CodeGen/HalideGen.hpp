#pragma once

#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>
#include "Halide.h"

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class HalideGen: public IteratorValueVisitor {
    FRIEND_TEST(codegen_tests, func);

    static bool AutoSchedulerLoaded;

protected:
    const BindingContext& ctx;
    const CodeGenContext& cgCtx;
    const TensorView& tensorView;

    std::vector<Halide::Param<int>> primaryConsts;
    std::vector<Halide::Param<int>> coefficientConsts;
    std::vector<Halide::Var> vars;

    mutable std::stack<Halide::Expr> stack;
    void visit(VariableValueNode& value) override;
    void visit(ConstValueNode& value) override;
    void visit(ImmediateValueNode& value) override;
    void visit(BinaryOpValueNode& value) override;
    Halide::Expr evaluate(IteratorValue& value) const;

    // Returns the (input, func) pair.
    std::pair<std::vector<Halide::ImageParam>, Halide::Func> createFunc() const;

public:
    HalideGen(const BindingContext& ctx, const TensorView& tensorView);
};

} // namespace kas
