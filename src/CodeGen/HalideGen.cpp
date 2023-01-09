#include <Halide.h>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

bool HalideGen::AutoSchedulerLoaded = false;

void HalideGen::visit(VariableValueNode& value) {
    stack.emplace(vars[value.variableId]);
}
void HalideGen::visit(ConstValueNode& value) {
    const auto& primary = value.value->primary;
    const auto& coefficient = value.value->coefficient;
    const std::size_t primaryCnt = ctx.getPrimaryCount(), coefficientCnt = ctx.getCoefficientCount();
    Halide::Expr size = 1;
    for (std::size_t i = 0; i < primaryCnt; ++i) {
        size *= Halide::pow(primaryConsts[i], primary[i]);
    }
    for (std::size_t i = 0; i < coefficientCnt; ++i) {
        size *= Halide::pow(coefficientConsts[i], coefficient[i]);
    }
    stack.emplace(std::move(size));
}
void HalideGen::visit(ImmediateValueNode& value) {
    stack.emplace(value.value);
}
void HalideGen::visit(BinaryOpValueNode& value) {
    value.op1->accept(*this);
    value.op2->accept(*this);
    Halide::Expr rhs = std::move(stack.top());
    stack.pop();
    Halide::Expr lhs = std::move(stack.top());
    stack.pop();
    using Type = BinaryOpValueNode::Type;
    switch (value.type) {
    case Type::Add: stack.emplace(lhs + rhs); break;
    case Type::Sub: stack.emplace(lhs - rhs); break;
    case Type::Mul: stack.emplace(lhs * rhs); break;
    case Type::Mod: stack.emplace(lhs % rhs); break;
    case Type::Div: stack.emplace(lhs / rhs); break;
    }
}

Halide::Expr HalideGen::evaluate(IteratorValue& value) const {
    value.accept(const_cast<HalideGen&>(*this));
    KAS_ASSERT(stack.size() == 1);
    Halide::Expr result = std::move(stack.top());
    stack.pop();
    return result;
}

std::pair<std::vector<Halide::ImageParam>, Halide::Func> HalideGen::createFunc() const {
    std::vector<Halide::ImageParam> inputs;
    for (std::size_t inputId = 0; inputId < tensorView.tensors.size(); ++inputId) {
        auto tensor = tensorView.tensors[inputId];
        inputs.emplace_back(Halide::type_of<float>(), tensor->shape.size(), std::string(cgCtx.getTensorName(tensor->tensorId)));
    }
    // Make the names parameterized. TODO
    Halide::Func func("output");

    std::vector<Halide::Var> outerLoops;
    for (std::size_t i: cgCtx.outerLoopIterators) {
        outerLoops.emplace_back(vars[i]);
    }

    Halide::Expr rhs = 1.0f;
    for (std::size_t inputId = 0; inputId < tensorView.tensors.size(); ++inputId) {
        const auto& tensor = tensorView.tensors[inputId];
        std::vector<Halide::Expr> indices;
        for (std::size_t i = 0; i < tensor->shape.size(); ++i) {
            indices.emplace_back(evaluate(*tensor->access[i]));
        }
        // Add support for other arithmetic operations. TODO
        rhs *= inputs[inputId](indices);
    }

    func(outerLoops) = rhs;

    return { std::move(inputs), std::move(func) };
}

HalideGen::HalideGen(const BindingContext& ctx, const TensorView& tensorView):
    ctx { ctx },
    cgCtx { *tensorView.cgCtx },
    tensorView { tensorView }
{
    for (const auto& metadata: ctx.primaryMetadata) {
        primaryConsts.emplace_back(metadata.alias);
    }
    for (const auto& metadata: ctx.coefficientMetadata) {
        coefficientConsts.emplace_back(metadata.alias);
    }

    for (const auto& [_, metadata]: cgCtx.iteratorVariableMetadata) {
        vars.emplace_back(metadata.name);
    }
}

} // namespace kas
