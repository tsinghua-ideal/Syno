#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

void HalideGen::GuardAutoSchedulers() {
    static bool loaded = false;
    if (!loaded) {
        Halide::load_plugin("autoschedule_mullapudi2016");
        Halide::load_plugin("autoschedule_li2018");
        Halide::load_plugin("autoschedule_adams2019");
    }
    loaded = true;
}

Halide::Target HalideGen::GetHostTarget(bool useGPU) {
    auto t = Halide::get_host_target();
    if (useGPU) {
        t = t
            .with_feature(Halide::Target::CUDA)
            .with_feature(Halide::Target::UserContext);
    }
    KAS_ASSERT(Halide::host_supports_target_device(t));
    return t;
}

void HalideGen::HalideExprEvaluator::visit(VariableValueNode& value) {
    if (!value.isReduce) {
        if (!useRVar) {
            evaluator = parent.getOuterIterator(value.index);
        } else {
            evaluator = getOuterIteratorAsRVar(value.index);
        }
    } else {
        evaluator = getInnerIterator(value.index);
    }
}
void HalideGen::HalideExprEvaluator::visit(ConstValueNode& value) {
    evaluator = parent.evaluate(consts, value.value);
}
void HalideGen::HalideExprEvaluator::visit(ImmediateValueNode& value) {
    evaluator = value.value;
}
void HalideGen::HalideExprEvaluator::visit(BinaryOpValueNode& value) {
    Halide::Expr lhs = evaluate(value.op1);
    Halide::Expr rhs = evaluate(value.op2);
    using Type = BinaryOpValueNode::Type;
    switch (value.type) {
    case Type::Add: evaluator = lhs + rhs; break;
    case Type::Sub: evaluator = lhs - rhs; break;
    case Type::Mul: evaluator = lhs * rhs; break;
    case Type::Mod: evaluator = lhs % rhs; break;
    case Type::Div: evaluator = lhs / rhs; break;
    }
}
void HalideGen::HalideExprEvaluator::visit(IntervalBoundValueNode& value) {
    Halide::Expr input = evaluate(value.input);
    Halide::Expr min = evaluate(value.min), max = evaluate(value.max);
    evaluator = Halide::clamp(input, min, max - 1);
    if (parent.options.zeroPadding) {
        // Add this constraint to the constraints list.
        constraintsBounds.emplace_back(evaluate(value));
    }
}

Halide::Expr HalideGen::HalideExprEvaluator::evaluate(const IteratorValue& value) {
    auto it = cache.find(value);
    if (it != cache.end()) {
        return it->second;
    }
    value.accept(*this);
    Halide::Expr result = std::move(evaluator);
    evaluator = 0;
    cache.emplace(value, result);
    return result;
}

Halide::Expr HalideGen::HalideExprEvaluator::evaluate(const IntervalBoundValueNode& value) {
    auto input = evaluate(value.input);
    auto min = evaluate(value.min), max = evaluate(value.max);
    return min <= input && input < max;
}

Halide::Expr HalideGen::evaluate(const ConcreteConsts& consts, HalideExprEvaluator::Cache& cache, std::vector<Halide::Expr>& constraintsBounds, const std::vector<Halide::RVar>& outerIteratorsAsRVars, bool useRVar, const std::vector<Halide::RVar>& innerIterators, const IteratorValue& value) const {
    HalideExprEvaluator evaluator { consts, cache, constraintsBounds, outerIteratorsAsRVars, useRVar, innerIterators, *this };
    return evaluator.evaluate(value);
}

Halide::Expr HalideGen::evaluate(const ConcreteConsts& consts, const Size& value) const {
    return value.eval<int>(consts.primaryWrapper(), consts.coefficientWrapper());
}

ConcreteConsts HalideGen::realizeConsts(const std::map<std::string, std::size_t>& mappings) const {
    return ctx.realizeConsts(mappings);
}

HalideGen::EvaluatedAccess HalideGen::evaluateAccess(const ConcreteConsts& consts, bool useRVar) const {
    auto fusedRegion = evaluate(consts, tensorView.getReduceShape(), false);
    const std::size_t reduceCnt = fusedRegion.size();
    if (useRVar) {
        auto outerRegion = evaluate(consts, tensorView.getShape());
        std::ranges::move(outerRegion, std::back_inserter(fusedRegion));
    }

    Halide::RDom rdom(fusedRegion);

    std::vector<Halide::RVar> outerIteratorsAsRVars; // If we do not use RVar, this is empty.
    for (std::size_t i = reduceCnt; i < fusedRegion.size(); ++i) {
        outerIteratorsAsRVars.emplace_back(rdom[i]);
    }
    std::vector<Halide::RVar> innerIterators;
    for (std::size_t i = 0; i < reduceCnt; ++i) {
        innerIterators.emplace_back(rdom[i]);
    }

    HalideExprEvaluator::Cache cache;
    std::vector<Halide::Expr> constraintsBounds;

    std::vector<std::vector<Halide::Expr>> tensorIndices;
    for (auto&& tensor: tensorView.getUnderlyingTensors()) {
        std::vector<Halide::Expr> indices;
        for (auto&& access: tensor.getAccess()) {
            indices.emplace_back(evaluate(consts, cache, constraintsBounds, outerIteratorsAsRVars, useRVar, innerIterators, access));
        }
        // Adapt to column-major layout.
        std::ranges::reverse(indices);
        tensorIndices.emplace_back(std::move(indices));
    }

    return { std::move(outerIteratorsAsRVars), std::move(innerIterators), std::move(tensorIndices), std::move(constraintsBounds) };
}

HalideGen::ForwardArgsAndFunc HalideGen::createFunc(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName, bool zeroBoundary, bool useRVars) {
    std::vector<Halide::ImageParam> inputs;

    Halide::Expr rhs = 1.0f;
    const auto& tensors = tensorView.getUnderlyingTensors();
    for (std::size_t inputId = 0; auto&& tensor: tensors) {
        inputs.emplace_back(Halide::type_of<float>(), tensor.getShape().size(), tensor.getName());
        auto est = evaluate(consts, tensor.getShape());
        inputs.back().set_estimates(est);
        // Now that the bug in autodiff is handled, we no longer need zero boundary.
        if (zeroBoundary) {
            Halide::Func wrappedInput = Halide::BoundaryConditions::constant_exterior(inputs.back(), 0.0f, est);
            // Add support for other arithmetic operations. TODO
            rhs *= wrappedInput(access.tensorIndices.at(inputId));
        } else {
            rhs *= inputs.back()(access.tensorIndices.at(inputId));
        }
        ++inputId;
    }

    Halide::Expr guardedRhs;
    if (options.zeroPadding) {
        // To enforce zero padding, we need to compute constraints of the region.
        Halide::Expr guardCond = std::accumulate(
            access.constraintsBounds.begin(),
            access.constraintsBounds.end(),
            Halide::cast<bool>(true),
            [](auto&& lhs, auto&& rhs) { return lhs && rhs; }
        );
        // Enforce zero-padding.
        guardedRhs = Halide::select(guardCond, Halide::likely(rhs), 0.0f);
    } else {
        guardedRhs = rhs;
    }

    Halide::Func func { std::string(funcName) };
    // Here we ignore all the `Map`s and `Reduce`s, and just sum up the entries for simplicity. TODO: Add other operations, and consider fusions of reductions.
    if (!useRVars) {
        if (access.innerIterators.empty()) { // If there is no inner loops, do not use sum.
            func(outerIterators) = guardedRhs;
        } else {
            func(outerIterators) = Halide::sum(guardedRhs); // Here the outer loop iterators are Halide::Var.
        }
    } else {
        func(outerIterators) = Halide::cast<float>(0);
        std::vector<Halide::Expr> lhs;
        for (auto&& rvar: access.outerIteratorsAsRVars) {
            lhs.emplace_back(rvar);
        }
        func(lhs) += guardedRhs; // Here the outer loop iterators are Halide::RVar.
    }

    auto concreteOutputShape = evaluate(consts, tensorView.getShape());
    func.set_estimates(concreteOutputShape);
    return { std::move(inputs), std::move(func) };
}

HalideGen::BackwardArgsAndFuncs HalideGen::createFuncGrad(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName) {
    constexpr bool zeroBoundary = false; // Now that bug in autodiff is handled, we no longer need to use zero boundary.

    // We must use RVars, because Halide's autodiff comes with bugs that make ShareOp malfunctions!
    auto [input, func] = createFunc(consts, access, funcName, zeroBoundary, true);

    auto outputShape = tensorView.getShape();
    Halide::Region outputRegion = evaluate(consts, outputShape);
    Halide::ImageParam outputGrad(Halide::type_of<float>(), outputShape.size(), "output_grad");
    outputGrad.set_estimates(outputRegion);

    Halide::Derivative d = Halide::propagate_adjoints(func, 
        zeroBoundary ?
        Halide::BoundaryConditions::constant_exterior(outputGrad, 0.0f, outputRegion) :
        outputGrad,
    outputRegion);

    const auto& inputTensors = tensorView.getUnderlyingTensors();
    std::vector<Halide::Func> gradFuncs;
    for (std::size_t i = 0; i < input.size(); ++i) {
        Halide::Func dInput(input[i].name() + "_grad");
        dInput(Halide::_) = d(input[i])(Halide::_);
        gradFuncs.emplace_back(std::move(dInput));
        gradFuncs.back().set_estimates(evaluate(consts, inputTensors[i].getShape()));
    }
    input.emplace_back(std::move(outputGrad));
    return { std::move(input), std::move(gradFuncs) };
}

HalideGen::ForwardAndBackwardFuncs HalideGen::createPipelines(const std::map<std::string, std::size_t>& mappings, std::string_view funcName, bool useRVar) {
    const auto& tensors = tensorView.getUnderlyingTensors();

    auto consts = realizeConsts(mappings);
    if (tensorView.getReduceShape().size() == 0) { // No need to use RVars. Override.
        useRVar = false;
    }
    auto access = evaluateAccess(consts, useRVar);
    try {
        auto [forwardInputs, forwardFunc] = createFunc(consts, access, funcName, false, useRVar);
        auto [backwardInputs, backwardFuncs] = createFuncGrad(
            consts,
            // We must use RVar, because bugs in Halide's autodiff.
            useRVar ? access : evaluateAccess(consts, true),
            funcName
        );
        KAS_ASSERT(forwardInputs.size() == tensors.size());
        KAS_ASSERT(backwardInputs.size() == tensors.size() + 1);
        KAS_ASSERT(backwardFuncs.size() == tensors.size());

        return {
            std::move(forwardInputs), std::move(forwardFunc),
            std::move(backwardInputs), std::move(backwardFuncs),
        };
    } catch (const Halide::Error& e) {
        std::cerr << "Halide Error: " << e.what() << std::endl;
        throw;
    }
}

HalideGen::HalideGen(const BindingContext& ctx, const TensorView& tensorView, Options options):
    ctx { ctx },
    tensorView { tensorView },
    options { std::move(options) }
{
    for (auto&& it: tensorView.interface) {
        outerIterators.emplace_back(it->getName());
    }
    // Adapt to column-major layout.
    std::ranges::reverse(outerIterators);
}

void HalideGen::generate(std::filesystem::path outputPath, std::string_view funcName, const std::map<std::string, std::size_t>& mappings) {
    std::string backwardName = std::string(funcName) + "_grad";

    // Create Halide Functions.
    auto [forwardInputs, forwardFunc,
        backwardInputs, backwardFuncs
    ] = createPipelines(mappings, funcName);

    // Prepare auto schedulers.
    using Scheduler = Options::AutoScheduler;
    const bool computeRoot = options.scheduler == Scheduler::ComputeRoot;
    auto target = GetHostTarget(options.useGPU);
    GuardAutoSchedulers();
    std::optional<Halide::AutoschedulerParams> params;
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

    // Apply auto schedulers to pipelines.
    if (computeRoot) {
        forwardFunc.compute_root();
        for (auto& func: backwardFuncs) {
            func.compute_root();
        }
    }
    Halide::Pipeline forwardPipeline { forwardFunc };
    Halide::Pipeline backwardPipeline { backwardFuncs };
    if (!computeRoot) {
        forwardPipeline.apply_autoscheduler(target, params.value());
        backwardPipeline.apply_autoscheduler(target, params.value());
    }

    // Compile to Halide modules.
    std::vector<Halide::Argument> forwardArgs;
    std::vector<Halide::Argument> backwardArgs;
    std::ranges::copy(forwardInputs, std::back_inserter(forwardArgs));
    std::ranges::copy(backwardInputs, std::back_inserter(backwardArgs));
    auto forwardModule = forwardPipeline.compile_to_module(forwardArgs, std::string(funcName), target);
    auto backwardModule = backwardPipeline.compile_to_module(backwardArgs, backwardName, target);

    // Mitigate Halide bugs.
    for (auto& f: forwardModule.functions()) {
        // This is to solve the pytorch codegen bug, which generates internal functions as c codegen does not.
        f.linkage = Halide::LinkageType::External;
    }
    for (auto& f: backwardModule.functions()) {
        // Same as above.
        f.linkage = Halide::LinkageType::External;
    }

    // Write to output.
    auto ext = Halide::Internal::get_output_info(target);
    const auto flagsForModule = [&ext](std::filesystem::path filename) -> std::map<Halide::OutputFileType, std::string> {
        using FileType = Halide::OutputFileType;
        return {
            {FileType::stmt, filename.replace_extension(ext.at(FileType::stmt).extension)},
            {FileType::static_library, filename.replace_extension(ext.at(FileType::static_library).extension)},
            {FileType::pytorch_wrapper, filename.replace_extension(ext.at(FileType::pytorch_wrapper).extension)},
        };
    };
    std::filesystem::create_directories(outputPath);
    forwardModule.compile(flagsForModule(outputPath / funcName));
    backwardModule.compile(flagsForModule(outputPath / backwardName));
}

} // namespace kas
