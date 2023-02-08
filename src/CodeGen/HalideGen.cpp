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

Halide::Target HalideGen::GetTarget(bool useGPU) {
    auto t = Halide::get_host_target();
    if (useGPU) {
        t = t
            .with_feature(Halide::Target::CUDA)
            .with_feature(Halide::Target::UserContext);
    }
    KAS_ASSERT(Halide::host_supports_target_device(t));
    return t;
}

void HalideGen::visit(VariableValueNode& value) {
    stack.emplace(vars[value.variableId]);
}
void HalideGen::visit(ConstValueNode& value) {
    stack.emplace(evaluate(value.value));
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
    return value->eval<Halide::Expr>(
        [this](std::size_t i) { return primaryConsts[i]; },
        [this](std::size_t i) { return coefficientConsts[i]; }
    );
}

Halide::Region HalideGen::evaluate(const Shape& shape) {
    Halide::Region region;
    for (const auto& s: shape.getSizes()) {
        region.emplace_back(0, evaluate(s));
    }
    // Because Halide uses column-major, we need to reverse the order of indices.
    std::ranges::reverse(region);
    return region;
}

void HalideGen::evaluateAccess() {
    if (accessEvaluated) {
        return;
    }
    for (std::size_t i: cgCtx.outerLoopIterators) {
        outerLoops.emplace_back(vars.at(i));
    }
    // Adapt to column-major layout.
    std::ranges::reverse(outerLoops);
    for (const auto& m: tensorView.manipulations) {
        reduceLoops.emplace_back(vars.at(m.iteratorVariableId));
    }
    // Adapt to column-major layout.
    std::ranges::reverse(reduceLoops);
    const auto& tensors = tensorView.getUnderlyingTensors();
    for (std::size_t inputId = 0; inputId < tensors.size(); ++inputId) {
        const auto& tensor = tensors[inputId];
        std::vector<Halide::Expr> indices;
        for (std::size_t i = 0; i < tensor->shape.size(); ++i) {
            indices.emplace_back(evaluate(*tensor->access.at(i)));
        }
        // Adapt to column-major layout.
        std::ranges::reverse(indices);
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
        // Here for simplicity and to avoid out of bound errors (still not sure why this happens), use zero padding. Need better solution. TODO
        Halide::Func wrappedInput = Halide::BoundaryConditions::constant_exterior(inputs[inputId], 0.0f, evaluate(tensors[inputId]->getShapeRef()));
        // Add support for other arithmetic operations. TODO
        rhs *= wrappedInput(tensorIndices.at(inputId));
    }

    // tempAccess = [...reversed reduce loops, ...reversed outer loops], to adapt to column major Halide.
    std::vector<Halide::Var> tempAccess(reduceLoops);
    std::ranges::copy(outerLoops, std::back_inserter(tempAccess));
    Halide::Func temp;
    temp(tempAccess) = rhs;
    for(const auto& m: tensorView.manipulations) {
        Halide::Func newTemp;
        Halide::RDom r(0, evaluate(m.getIterator()->getSize()));
        tempAccess.erase(tempAccess.begin());
        std::vector<Halide::Expr> reduceAccess { r };
        std::ranges::copy(tempAccess, std::back_inserter(reduceAccess));
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
        case MapType::MapTypeCount: KAS_CRITICAL("Invalid map type"); break;
        }
        using ReduceType = Manipulation::ReduceType;
        switch (m.reduceType) {
        case ReduceType::Sum:     newTemp(tempAccess) = Halide::sum(tempValue); break;
        case ReduceType::Max:     newTemp(tempAccess) = Halide::maximum(tempValue); break;
        case ReduceType::Mean:    newTemp(tempAccess) = Halide::sum(tempValue) / Halide::cast<float>(evaluate(m.getIterator()->getSize())); break;
        case ReduceType::Min:     newTemp(tempAccess) = Halide::minimum(tempValue); break;
        case ReduceType::Product: newTemp(tempAccess) = Halide::product(tempValue); break;
        case ReduceType::ReduceTypeCount: KAS_CRITICAL("Invalid reduce type"); break;
        }
        temp = std::move(newTemp);
    }

    func(outerLoops) = temp(outerLoops);

    return { std::move(inputs), std::move(func) };
}

std::pair<std::vector<Halide::ImageParam>, std::vector<Halide::Func>> HalideGen::createFuncGrad(std::string_view funcName) {
    auto [input, func] = createFunc(funcName);
    Shape outputShape = tensorView.getShape();
    Halide::Region outputRegion = evaluate(outputShape);
    Halide::ImageParam outputGrad(Halide::type_of<float>(), outputShape.size(), "output_grad");
    Halide::Func wrappedOutputGrad = Halide::BoundaryConditions::constant_exterior(outputGrad, 0.0f, outputRegion);
    Halide::Derivative d = Halide::propagate_adjoints(func, wrappedOutputGrad, outputRegion);
    std::vector<Halide::Func> gradFuncs;
    for (std::size_t i = 0; i < input.size(); ++i) {
        Halide::Func dInput(input[i].name() + "_grad");
        dInput(Halide::_) = d(input[i])(Halide::_);
        gradFuncs.emplace_back(std::move(dInput));
    }
    input.emplace_back(std::move(outputGrad));
    return { std::move(input), std::move(gradFuncs) };
}

Halide::Region HalideGen::ConcreteShapeToRegion(const std::vector<std::size_t>& shape) {
    Halide::Region region;
    for (auto dim: shape) {
        region.emplace_back(0, static_cast<int>(dim));
    }
    // Because Halide uses column-major, we need to reverse the order of indices.
    std::ranges::reverse(region);
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

void HalideGen::generate(std::filesystem::path outputPath, std::string_view funcName, Options options) {
    for (std::size_t i = 0; i < primaryConsts.size(); ++i) {
        primaryConsts[i].set_estimate(static_cast<int>(ctx.getPrimaryEstimate(i)));
    }
    for (std::size_t i = 0; i < coefficientConsts.size(); ++i) {
        coefficientConsts[i].set_estimate(static_cast<int>(ctx.getCoefficientEstimate(i)));
    }
    const auto& tensors = tensorView.getUnderlyingTensors();

    auto target = GetTarget(options.useGPU);
    auto ext = Halide::Internal::get_output_info(target);
    if (!AutoSchedulerLoaded) {
        Halide::load_plugin("autoschedule_mullapudi2016");
        Halide::load_plugin("autoschedule_li2018");
        Halide::load_plugin("autoschedule_adams2019");
        AutoSchedulerLoaded = true;
    }
    std::optional<Halide::AutoschedulerParams> params;
    using Scheduler = Options::AutoScheduler;
    const bool computeRoot = options.scheduler == Scheduler::ComputeRoot;
    if (!computeRoot) {
        std::string scheduler;
        switch (options.scheduler) {
        case Scheduler::Mullapudi2016:  scheduler = "Mullapudi2016";    break;
        case Scheduler::Li2018:         scheduler = "Li2018";           break;
        case Scheduler::Adams2019:      scheduler = "Adams2019";        break;
        case Scheduler::ComputeRoot:    KAS_UNREACHABLE();
        }
        params = { scheduler };
    }

    auto [forwardInputs, forwardFunc] = createFunc(funcName);
    auto [backwardInputs, backwardFuncs] = createFuncGrad(funcName);
    KAS_ASSERT(forwardInputs.size() == tensors.size());
    KAS_ASSERT(backwardInputs.size() == tensors.size() + 1);
    KAS_ASSERT(backwardFuncs.size() == tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        auto est = ConcreteShapeToRegion(tensor->shape.estimate(ctx));
        forwardInputs[i].set_estimates(est);
        backwardInputs[i].set_estimates(est);
        backwardFuncs[i].set_estimates(est);
    }
    auto outputShapeEst = ConcreteShapeToRegion(tensorView.getShape().estimate(ctx));
    forwardFunc.set_estimates(outputShapeEst);
    backwardInputs.back().set_estimates(outputShapeEst);

    std::vector<Halide::Argument> forwardArgs;
    std::vector<Halide::Argument> backwardArgs;
    std::ranges::copy(primaryConsts, std::back_inserter(forwardArgs));
    std::ranges::copy(primaryConsts, std::back_inserter(backwardArgs));
    std::ranges::copy(coefficientConsts, std::back_inserter(forwardArgs));
    std::ranges::copy(coefficientConsts, std::back_inserter(backwardArgs));
    std::ranges::copy(forwardInputs, std::back_inserter(forwardArgs));
    std::ranges::copy(backwardInputs, std::back_inserter(backwardArgs));

    if (computeRoot) {
        forwardFunc.compute_root();
        for (auto& func: backwardFuncs) {
            func.compute_root();
        }
    }

    const auto flagsForModule = [&ext](std::filesystem::path filename) -> std::map<Halide::OutputFileType, std::string> {
        using FileType = Halide::OutputFileType;
        return {
            {FileType::stmt, filename.replace_extension(ext.at(FileType::stmt).extension)},
            {FileType::static_library, filename.replace_extension(ext.at(FileType::static_library).extension)},
            {FileType::pytorch_wrapper, filename.replace_extension(ext.at(FileType::pytorch_wrapper).extension)},
        };
    };

    Halide::Pipeline forwardPipeline { forwardFunc };
    if (!computeRoot) {
        forwardPipeline.apply_autoscheduler(target, params.value());
    }
    auto forwardModule = forwardPipeline.compile_to_module(forwardArgs, std::string(funcName), target);
    for (auto& f: forwardModule.functions()) {
        // This is to solve the pytorch codegen bug, which generates internal functions as c codegen does not.
        f.linkage = Halide::LinkageType::External;
    }
    std::filesystem::create_directories(outputPath);
    forwardModule.compile(flagsForModule(outputPath / funcName));

    Halide::Pipeline backwardPipeline(backwardFuncs);
    if (!computeRoot) {
        backwardPipeline.apply_autoscheduler(target, params.value());
    }
    std::string backwardName = std::string(funcName) + "_grad";
    auto backwardModule = backwardPipeline.compile_to_module(backwardArgs, backwardName, target);
    for (auto& f: backwardModule.functions()) {
        f.linkage = Halide::LinkageType::External;
    }
    backwardModule.compile(flagsForModule(outputPath / backwardName));
}

Halide::ParamMap HalideGen::getParamMap(const std::map<std::string, std::size_t>& mappings) const {
    Halide::ParamMap params;
    for (std::size_t i = 0; i < primaryConsts.size(); ++i) {
        auto alias = ctx.getPrimaryAlias(i);
        params.set(primaryConsts[i], static_cast<int>(mappings.at(std::string(alias))));
    }
    for (std::size_t i = 0; i < coefficientConsts.size(); ++i) {
        auto alias = ctx.getCoefficientAlias(i);
        params.set(coefficientConsts[i], static_cast<int>(mappings.at(std::string(alias))));
    }
    return params;
}

} // namespace kas
