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

HalideAccess::HalideAccess(const ConcreteConsts& consts, const AbstractAccess& access):
    position { access.position }
{
    // Prepare iterators.
    for (auto& outerLoop: access.outerLoops) {
        outerLoops.emplace_back(outerLoop.as<VariableValueNode>().name);
    }
    auto reductionShape = ConcretizeShape(consts, access.innerLoopsShape, false); // The order of reduction does not matter, because interchanging them does not change the result.
    std::vector<Halide::RVar> innerLoops;
    if (reductionShape.size() > 0) {
        reductionDomain = Halide::RDom(reductionShape);
        for (std::size_t i = 0; i < reductionDomain.dimensions(); ++i) {
            innerLoops.emplace_back(reductionDomain[i]);
        }
    }

    struct Evaluator: public IteratorValueVisitor {
        const ConcreteConsts& consts;
        const std::vector<Halide::Var>& outerLoops;
        const std::vector<Halide::RVar>& innerLoops;
        std::vector<Halide::Expr>& constraints;
        std::map<IteratorValue, Halide::Expr> cache;
        Evaluator(const ConcreteConsts& consts, const std::vector<Halide::Var>& outerLoops, const std::vector<Halide::RVar>& innerLoops, std::vector<Halide::Expr>& constraints):
            consts { consts },
            outerLoops { outerLoops },
            innerLoops { innerLoops },
            constraints { constraints }
        {}
        Halide::Expr result;
        void visit(VariableValueNode& value) override {
            if (!value.isReduce) {
                result = outerLoops[value.index];
            } else {
                result = innerLoops[value.index];
            }
        }
        void visit(ConstValueNode& value) override {
            result = ConcretizeSize(consts, value.value);
        }
        void visit(ImmediateValueNode& value) override {
            result = value.value;
        }
        void visit(BinaryOpValueNode& value) override {
            Halide::Expr lhs = lower(value.op1);
            Halide::Expr rhs = lower(value.op2);
            using Type = BinaryOpValueNode::Type;
            switch (value.type) {
            case Type::Add: result = lhs + rhs; break;
            case Type::Sub: result = lhs - rhs; break;
            case Type::Mul: result = lhs * rhs; break;
            case Type::Mod: result = lhs % rhs; break;
            case Type::Div: result = lhs / rhs; break;
            }
        }
        // This `visit` produces a `clamp`, along with constraints.
        void visit(IntervalBoundValueNode& value) override {
            Halide::Expr input = lower(value.input);
            Halide::Expr min = lower(value.min), max = lower(value.max);
            constraints.emplace_back(min <= input && input < max);
            result = Halide::clamp(input, min, max - 1);
        }
        Halide::Expr lower(const IteratorValue& value) {
            auto it = cache.find(value);
            if (it != cache.end()) {
                return it->second;
            }
            value.accept(*this);
            cache.emplace(value, result);
            return std::move(result);
        }
    };
    Evaluator eval { consts, outerLoops, innerLoops, constraints };
    for (auto&& out: access.output) {
        output.emplace_back(eval.lower(out));
    }
    std::ranges::reverse(output); // To adapt to column-major Halide.
    for (auto&& input: access.inputs) {
        std::vector<Halide::Expr> lowered;
        for (auto&& in: input) {
            lowered.emplace_back(eval.lower(in));
        }
        std::ranges::reverse(lowered); // To adapt to column-major Halide.
        inputs.emplace_back(std::move(lowered));
    }
    std::ranges::reverse(outerLoops); // To adapt to column-major Halide.
}

void HalideGen::GuardAutoSchedulers() {
    static bool loaded = false;
    if (!loaded) {
        Halide::load_plugin("autoschedule_mullapudi2016");
        Halide::load_plugin("autoschedule_li2018");
        Halide::load_plugin("autoschedule_adams2019");
        Halide::load_plugin("autoschedule_anderson2021");
    }
    loaded = true;
}

Halide::Func HalideGen::lower(std::vector<Halide::ImageParam>& inputTensors, Halide::ImageParam *outputTensor, const HalideAccess& access, std::string_view funcName) {
    Halide::Func func { std::string(funcName) };
    Halide::Expr rhs;
    auto mult = [&](Halide::Expr what) {
        if (rhs.defined()) {
            rhs = rhs * what;
        } else {
            rhs = what;
        }
    };
    // out[...] += in_0[...] * in_1[...] * ... * in_n[...], or
    // in_i_grad[...] += in_0[...] * in_1[...] * ...  * out_grad[...] * ... * in_n[...]
    for (std::size_t inputId = 0; auto&& tensor: inputTensors) {
        if (inputId != access.position) {
            mult(tensor(access.inputs[inputId]));
        } else {
            mult((*outputTensor)(access.output));
        }
        ++inputId;
    }

    Halide::Expr guardedRhs;
    if (access.constraints.empty()) {
        guardedRhs = rhs;
    } else {
        Halide::Expr conditions = access.constraints[0];
        for (std::size_t i = 1; i < access.constraints.size(); ++i) {
            conditions = conditions && access.constraints[i];
        }
        guardedRhs = Halide::select(conditions, Halide::likely(rhs), 0.0f);
    }

    if (!access.reductionDomain.defined()) {
        func(access.outerLoops) = guardedRhs;
    } else {
        // TODO: In autodiff, guardedRhs may contain no RVar at all, in which case this throws!
        func(access.outerLoops) = Halide::sum(guardedRhs);
    }
    return func;
}

HalideGen::ConcreteShapes HalideGen::concretizeShapes(const ConcreteConsts& consts) const {
    std::vector<Halide::Region> inputShapes;
    for (auto&& tensor: tensorView.getUnderlyingTensors()) {
        inputShapes.emplace_back(ConcretizeShape(consts, tensor.getShape()));
    }
    return {
        .inputShapes = std::move(inputShapes),
        .outputShape = ConcretizeShape(consts, tensorView.getInterfaceShape()),
    };
}

HalideGen::ForwardArgsAndFunc HalideGen::createFunc(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName) {
    std::vector<Halide::ImageParam> inputs;

    const auto& tensors = tensorView.getUnderlyingTensors();
    for (std::size_t inputId = 0; auto&& tensor: tensors) {
        inputs.emplace_back(Halide::type_of<float>(), tensor.getShape().size(), tensor.getName());
        inputs.back().set_estimates(shapes.inputShapes[inputId]);
        ++inputId;
    }

    Halide::Func out = lower(inputs, nullptr, HalideAccess(consts, tensorView.getForwardAccess()), funcName);
    out.set_estimates(shapes.outputShape);
    return { std::move(inputs), std::move(out) };
}

HalideGen::BackwardArgsAndFuncs HalideGen::createFuncGrad(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName) {
    std::vector<Halide::ImageParam> inputs;
    std::vector<Halide::Func> inputsGrads;

    for (std::size_t inputId = 0; auto&& tensor: tensorView.getUnderlyingTensors()) {
        inputs.emplace_back(Halide::type_of<float>(), tensor.getShape().size(), tensor.getName());
        inputs.back().set_estimates(shapes.inputShapes[inputId]);
        ++inputId;
    }

    Halide::ImageParam outputGrad(Halide::type_of<float>(), tensorView.getInterfaceShape().size(), "output_grad");
    outputGrad.set_estimates(shapes.outputShape);

    for (auto&& backward: tensorView.getBackwardAccesses()) {
        inputsGrads.emplace_back(lower(inputs, &outputGrad, HalideAccess(consts, backward), funcName));
        inputsGrads.back().set_estimates(shapes.inputShapes[backward.position]);
    }

    inputs.emplace_back(std::move(outputGrad));
    return { std::move(inputs), std::move(inputsGrads) };
}

HalideGen::ForwardAndBackwardFuncs HalideGen::createPipelines(const ConcreteConsts& consts, std::string_view funcName) {
    const auto& tensors = tensorView.getUnderlyingTensors();

    auto shapes = concretizeShapes(consts);
    try {
        auto [forwardInputs, forwardFunc] = createFunc(consts, shapes, funcName);
        auto [backwardInputs, backwardFuncs] = createFuncGrad(consts, shapes, funcName);
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

HalideGen::ScheduledPipelins HalideGen::ApplyAutoScheduler(Halide::Func& forwardFunc, std::vector<Halide::Func>& backwardFuncs, const Halide::Target& target, Options::AutoScheduler scheduler, bool verbose) {
    // Prepare auto schedulers.
    using Scheduler = Options::AutoScheduler;
    const bool computeRoot = scheduler == Scheduler::ComputeRoot;
    GuardAutoSchedulers();
    std::optional<Halide::AutoschedulerParams> params;
    if (!computeRoot) {
        std::string schedulerName;
        switch (scheduler) {
        case Scheduler::Mullapudi2016:  schedulerName = "Mullapudi2016";    break;
        case Scheduler::Li2018:         schedulerName = "Li2018";           break;
        case Scheduler::Adams2019:      schedulerName = "Adams2019";        break;
        case Scheduler::Anderson2021:   schedulerName = "Anderson2021";     break;
        case Scheduler::ComputeRoot:    KAS_UNREACHABLE();
        }
        params = { schedulerName };
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
        auto forwardResult = forwardPipeline.apply_autoscheduler(target, params.value());
        if (verbose)
            fmt::print(stderr, "Forward pipeline:\n{}\n", forwardResult.schedule_source);
        auto backwardResult = backwardPipeline.apply_autoscheduler(target, params.value());
        if (verbose)
            fmt::print(stderr, "Backward pipeline:\n{}\n", backwardResult.schedule_source);
    } else {
        if (verbose) {
            fmt::print(stderr, "Forward pipeline:\ncompute_root()\n");
            fmt::print(stderr, "Backward pipeline:\ncompute_root()\n");
        }
    }

    return { std::move(forwardPipeline), std::move(backwardPipeline) };
}

void HalideGen::GenerateFromPipelines(std::vector<Halide::ImageParam>& forwardInputs, std::vector<Halide::ImageParam>& backwardInputs, Halide::Pipeline& forwardPipeline, Halide::Pipeline& backwardPipeline, std::filesystem::path outputPath, std::string_view funcName, const Halide::Target& target) {
    std::string backwardName = std::string(funcName) + "_grad";

    // Compile to Halide modules.
    std::vector<Halide::Argument> forwardArgs;
    std::vector<Halide::Argument> backwardArgs;
    std::ranges::copy(forwardInputs, std::back_inserter(forwardArgs));
    std::ranges::copy(backwardInputs, std::back_inserter(backwardArgs));
    auto forwardModule = forwardPipeline.compile_to_module(forwardArgs, std::string(funcName), target);
    auto backwardModule = backwardPipeline.compile_to_module(backwardArgs, backwardName, target);

    // Write to output.
    auto ext = Halide::Internal::get_output_info(target);
    const auto flagsForModule = [&ext](std::filesystem::path filename) -> std::map<Halide::OutputFileType, std::string> {
        using FileType = Halide::OutputFileType;
        return {
            {FileType::stmt, filename.replace_extension(ext.at(FileType::stmt).extension)},
            {FileType::object, filename.replace_extension(ext.at(FileType::object).extension)},
            {FileType::pytorch_wrapper, filename.replace_extension(ext.at(FileType::pytorch_wrapper).extension)},
        };
    };
    std::filesystem::create_directories(outputPath);
    forwardModule.compile(flagsForModule(outputPath / funcName));
    backwardModule.compile(flagsForModule(outputPath / backwardName));
}

void HalideGen::generate(std::filesystem::path outputDir, std::string_view funcName, const ConcreteConsts& consts) {
    // Create Halide Functions.
    auto [forwardInputs, forwardFunc,
        backwardInputs, backwardFuncs
    ] = createPipelines(consts, funcName);

    auto target = GetHostTarget(options.useGPU, false);
    auto [forwardPipeline, backwardPipeline] = ApplyAutoScheduler(forwardFunc, backwardFuncs, target, options.scheduler, false);

    GenerateFromPipelines(forwardInputs, backwardInputs, forwardPipeline, backwardPipeline, outputDir, funcName, target);
}

} // namespace kas
