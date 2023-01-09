#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Manipulation.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

bool HalideGen::AutoSchedulerLoaded = false;

Halide::Target HalideGen::GetGPUTarget() {
    auto t = Halide::get_host_target()
        .with_feature(Halide::Target::CUDA)
        .with_feature(Halide::Target::UserContext);
    KAS_ASSERT(Halide::host_supports_target_device(t));
    return t;
}

void HalideGen::visit(VariableValueNode& value) {
    stack.emplace(vars[value.variableId]);
}
void HalideGen::visit(ConstValueNode& value) {
    auto factor = [](std::size_t cnt, const std::vector<Halide::Param<int>>& bases, const Size::ExprType& powers) -> std::pair<Halide::Expr, Halide::Expr> {
        Halide::Expr nominator = 1;
        Halide::Expr denominator = 1;
        for (std::size_t i = 0; i < cnt; ++i) {
            if (powers[i] > 0) {
                for (std::size_t j = 0; j < powers[i]; ++j) {
                    nominator *= bases[i];
                }
            } else if (powers[i] < 0) {
                for (std::size_t j = 0; j < -powers[i]; ++j) {
                    denominator *= bases[i];
                }
            }
        }
        return { nominator,  denominator};
    };
    auto [nP, dP] = factor(ctx.getPrimaryCount(), primaryConsts, value.value->primary);
    auto [nC, dC] = factor(ctx.getCoefficientCount(), coefficientConsts, value.value->coefficient);
    stack.emplace(nP * nC / dP / dC);
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

Halide::Expr HalideGen::evaluate(IteratorValue& value) {
    value.accept(*this);
    KAS_ASSERT(stack.size() == 1);
    Halide::Expr result = std::move(stack.top());
    stack.pop();
    return result;
}

Halide::Expr HalideGen::evaluate(std::shared_ptr<Size> value) {
    ConstValueNode node(std::move(value));
    return evaluate(node);
}

void HalideGen::evaluateAccess() {
    if (accessEvaluated) {
        return;
    }
    for (std::size_t i: cgCtx.outerLoopIterators) {
        outerLoops.emplace_back(vars[i]);
    }
    for (const auto& m: tensorView.manipulations) {
        reduceLoops.emplace_back(vars[m.iteratorVariableId]);
    }
    const auto& tensors = tensorView.getUnderlyingTensors();
    for (std::size_t inputId = 0; inputId < tensors.size(); ++inputId) {
        const auto& tensor = tensors[inputId];
        std::vector<Halide::Expr> indices;
        for (std::size_t i = 0; i < tensor->shape.size(); ++i) {
            indices.emplace_back(evaluate(*tensor->access[i]));
        }
        tensorIndices.emplace_back(std::move(indices));
    }
    accessEvaluated = true;
}

std::pair<std::vector<Halide::ImageParam>, Halide::Func> HalideGen::createFunc(std::string_view funcName) {
    std::vector<Halide::ImageParam> inputs;
    Halide::Func func { std::string(funcName) };

    evaluateAccess();

    const auto& tensors = tensorView.getUnderlyingTensors();
    for (std::size_t inputId = 0; inputId < tensors.size(); ++inputId) {
        auto tensor = tensors[inputId];
        inputs.emplace_back(Halide::type_of<float>(), tensor->shape.size(), std::string(cgCtx.getTensorName(tensor->tensorId)));
    }
    Halide::Expr rhs = 1.0f;
    for (std::size_t inputId = 0; inputId < tensors.size(); ++inputId) {
        // Add support for other arithmetic operations. TODO
        rhs *= inputs[inputId](tensorIndices.at(inputId));
    }

    std::vector<Halide::Var> tempAccess(outerLoops);
    std::copy(reduceLoops.rbegin(), reduceLoops.rend(), std::back_inserter(tempAccess));
    Halide::Func temp;
    temp(tempAccess) = rhs;
    for(const auto& m: tensorView.manipulations) {
        Halide::Func newTemp;
        Halide::RDom r(0, evaluate(m.getIterator()->getSize()));
        tempAccess.pop_back();
        std::vector<Halide::Expr> reduceAccess;
        std::copy(tempAccess.begin(), tempAccess.end(), std::back_inserter(reduceAccess));
        reduceAccess.emplace_back(r);
        Halide::Expr tempValue = temp(reduceAccess);
        using MapType = Manipulation::MapType;
        switch (m.mapType) {
        case MapType::Absolute: tempValue = Halide::abs(tempValue); break;
        case MapType::ArcTan:   tempValue = Halide::atan(tempValue); break;
        case MapType::Exp:      tempValue = Halide::exp(tempValue); break;
        case MapType::Log:      tempValue = Halide::log(tempValue); break;
        case MapType::Identity: break;
        case MapType::Inverse:  tempValue = 1.0f / tempValue; break;
        case MapType::Negative: tempValue = -tempValue; break;
        case MapType::ReLU:     tempValue = Halide::max(0.0f, tempValue); break;
        case MapType::Sigmoid:  tempValue = 1.0f / (1.0f + Halide::exp(-tempValue)); break;
        case MapType::Sign:     tempValue = Halide::select(tempValue > 0.0f, 1.0f, Halide::select(tempValue < 0.0f, -1.0f, 0.0f)); break;
        }
        using ReduceType = Manipulation::ReduceType;
        switch (m.reduceType) {
        case Manipulation::ReduceType::Sum:     newTemp(tempAccess) = Halide::sum(tempValue); break;
        case Manipulation::ReduceType::Max:     newTemp(tempAccess) = Halide::maximum(tempValue); break;
        case Manipulation::ReduceType::Mean:    newTemp(tempAccess) = Halide::sum(tempValue) / evaluate(m.getIterator()->getSize()); break;
        case Manipulation::ReduceType::Min:     newTemp(tempAccess) = Halide::minimum(tempValue); break;
        case Manipulation::ReduceType::Product: newTemp(tempAccess) = Halide::product(tempValue); break;
        }
        temp = std::move(newTemp);
    }

    func(outerLoops) = temp(outerLoops);

    return { std::move(inputs), std::move(func) };
}

Halide::Region HalideGen::ShapeEstimateToRegion(const std::vector<std::size_t>& estimate) {
    Halide::Region region;
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        region.emplace_back(0, static_cast<int>(estimate[i]));
    }
    return region;
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

    // outer loops + reduced loops == total loops
    KAS_ASSERT(cgCtx.outerLoopIterators.size() + tensorView.manipulations.size() == cgCtx.iteratorVariableMetadata.size());
    for (const auto& [_, metadata]: cgCtx.iteratorVariableMetadata) {
        vars.emplace_back(metadata.name);
    }
}

void HalideGen::generate(std::filesystem::path outputPath, std::string_view funcName) {
    auto [inputs, func] = createFunc(funcName);
    for (std::size_t i = 0; i < primaryConsts.size(); ++i) {
        primaryConsts[i].set_estimate(ctx.getPrimaryEstimate(i));
    }
    for (std::size_t i = 0; i < coefficientConsts.size(); ++i) {
        coefficientConsts[i].set_estimate(ctx.getCoefficientEstimate(i));
    }
    const auto& tensors = tensorView.getUnderlyingTensors();
    KAS_ASSERT(inputs.size() == tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        inputs[i].set_estimates(ShapeEstimateToRegion(tensor->shape.estimate(ctx)));
    }
    func.set_estimates(ShapeEstimateToRegion(tensorView.getShape().estimate(ctx)));

    std::vector<Halide::Argument> args;
    std::copy(primaryConsts.begin(), primaryConsts.end(), std::back_inserter(args));
    std::copy(coefficientConsts.begin(), coefficientConsts.end(), std::back_inserter(args));
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(args));

    if (!AutoSchedulerLoaded) {
        Halide::load_plugin("autoschedule_li2018");
        AutoSchedulerLoaded = true;
    }

    auto target = GetGPUTarget();
    auto ext = Halide::Internal::get_output_info(target);
    Halide::Pipeline pipeline(func);
    pipeline.auto_schedule("Li2018", target);

    const auto flagsForModule = [&ext](std::filesystem::path filename) -> std::map<Halide::OutputFileType, std::string> { 
        using FileType = Halide::OutputFileType;
        return {
            {FileType::stmt, filename.replace_extension(ext.at(FileType::stmt).extension)},
            {FileType::pytorch_wrapper, filename.replace_extension(ext.at(FileType::pytorch_wrapper).extension)},
            {FileType::static_library, filename.replace_extension(ext.at(FileType::static_library).extension)}
        };
    };

    std::filesystem::create_directories(outputPath);
    auto filename = outputPath / funcName;

    pipeline.compile_to(flagsForModule(filename), args, std::string(funcName), target);
}

} // namespace kas
