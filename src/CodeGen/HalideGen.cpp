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
#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Algorithm.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

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

namespace {
    struct Evaluator: public IteratorValueVisitor {
        const ConcreteConsts& consts;
        const std::vector<Halide::Var>& outerLoops;
        const std::vector<Halide::VarOrRVar>& innerLoops; // Manual rfactor requires Var.
        std::vector<HalideAccess::Constraint>& constraints;
        float likelyThreshold;
        std::map<IteratorValue, Halide::Expr> cache;
        Evaluator(const ConcreteConsts& consts, const std::vector<Halide::Var>& outerLoops, const std::vector<Halide::VarOrRVar>& innerLoops, std::vector<HalideAccess::Constraint>& constraints, float likelyThreshold):
            consts { consts },
            outerLoops { outerLoops },
            innerLoops { innerLoops },
            constraints { constraints },
            likelyThreshold { likelyThreshold }
        {}
        Halide::Expr result;
        void visit(VariableValueNode& value) override {
            if (!value.isReduce) {
                result = outerLoops[value.index];
            } else {
                const auto& var = innerLoops[value.index];
                if (var.is_rvar) {
                    result = var.rvar;
                } else {
                    result = var.var;
                }
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
            Halide::Expr min = 0, max = ConcretizeSize(consts, value.max);
            // If out-of-bounds access is frequent, do not add `likely` tag to original branch.
            bool likely = value.outOfBoundFraction.eval<float>(consts) < likelyThreshold;
            constraints.emplace_back(min <= input && input < max, likely);
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
}

HalideAccess HalideGen::lowerToAccess(const ConcreteConsts& consts, const AbstractAccess& access) const {
    HalideAccess ret;
    auto& [
        position,
        outerLoops,
        rfactoredInnerLoops,
        rfactoredDomain,
        reductionDomain,
        constraints,
        inputs,
        output,
        expression,
        divBy
    ] = ret;

    position = access.position;

    // Prepare iterators.
    for (auto& outerLoop: access.outerLoops) {
        outerLoops.emplace_back(outerLoop.as<VariableValueNode>().name);
    }
    // The order of reduction does not matter, because interchanging them does not change the result.
    auto reductionShape = ConcretizeShape(consts, access.innerLoopsShape, false);
    std::vector<Halide::VarOrRVar> innerLoops;
    if (reductionShape.size() > 0) {
        // Check for rfactor.
        bool rfactor = false;
        // We only rfactor the outermost dimension, which is usually batch size. TODO: make this more flexible.
        std::size_t rfactorCandidate = access.innerLoopsShape.sizes.back().eval<std::size_t>(consts);
        // At least 2 reductions, otherwise rfactor is meaningless.
        if (reductionShape.size() >= 2) {
            // We must not rfactor too small a size.
            if (rfactorCandidate >= options.rfactorThreshold) {
                rfactor = true;
                reductionShape.pop_back();
            }
        }
        reductionDomain = Halide::RDom(reductionShape);
        for (std::size_t i = 0; i < reductionDomain.dimensions(); ++i) {
            innerLoops.emplace_back(reductionDomain[i]);
        }
        if (rfactor) {
            Halide::Var rfactoredVar("rfactoredVar");
            rfactoredInnerLoops.emplace_back(rfactoredVar);
            innerLoops.emplace_back(rfactoredVar);
            rfactoredDomain = Halide::RDom(0, Halide::Expr(rfactorCandidate));
        }
    }

    Evaluator eval { consts, outerLoops, innerLoops, constraints, options.inBoundsLikelyThreshold };
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

    // Handle the expression.
    expression = access.expression;
    if (access.divBy) {
        divBy = ConcretizeSize(consts, *access.divBy);
    }
    return ret;
}

Halide::Func HalideGen::lowerAccessToFunc(std::vector<Halide::ImageParam>& inputTensors, Halide::ImageParam *outputTensor, const HalideAccess& access, std::string_view originalFuncName) const {
    std::string funcName { originalFuncName };
    if (access.position != TensorExpression::Output) {
        funcName += std::to_string(access.position);
    }

    class HalideExpressionLower final: public ValuedTensorExpressionVisitor<HalideExpressionLower, Halide::Expr> {
        std::vector<Halide::ImageParam>& inputTensors;
        Halide::ImageParam *outputTensor;
        const std::vector<std::vector<Halide::Expr>>& inputAccesses;
        const std::vector<Halide::Expr>& outputAccess;
    public:
        HalideExpressionLower(std::vector<Halide::ImageParam>& inputTensors, Halide::ImageParam *outputTensor, const std::vector<std::vector<Halide::Expr>>& inputAccesses, const std::vector<Halide::Expr>& outputAccess):
            inputTensors { inputTensors },
            outputTensor { outputTensor },
            inputAccesses { inputAccesses },
            outputAccess { outputAccess }
        {}
        Halide::Expr visits(IntegerTensorExpression& expr) {
            return Halide::Expr(static_cast<float>(expr.value));
        }
        Halide::Expr visits(TensorTensorExpression& expr) {
            if (expr.position == TensorExpression::Output) {
                return (*outputTensor)(outputAccess);
            } else {
                return inputTensors.at(expr.position)(inputAccesses.at(expr.position));
            }
        }
        Halide::Expr visits(BinaryOpTensorExpression& expr) {
            switch (expr.op) {
            case BinaryOpTensorExpression::Op::Add:
                return lower(expr.lhs) + lower(expr.rhs);
            case BinaryOpTensorExpression::Op::Mul:
                return lower(expr.lhs) * lower(expr.rhs);
            default:
                KAS_UNREACHABLE();
            }
        }
        Halide::Expr lower(TensorExpression& expr) {
            expr.accept(*this);
            return result();
        }
    };

    HalideExpressionLower exprLower { inputTensors, outputTensor, access.inputs, access.output };

    Halide::Expr rhs = exprLower.lower(access.expression);
    if (access.position != TensorExpression::Output) {
        rhs *= (*outputTensor)(access.output);
    }
    if (access.divBy) {
        // Handle the expression.
        rhs /= *access.divBy;
    }

    // Guard the boundary.
    if (!access.constraints.empty()) {
        auto getConstraints = [&](bool likely) {
            return FoldLeftFirst(
                access.constraints
                | std::views::filter([&](const HalideAccess::Constraint& c) {
                    return likely == c.likely;
                })
                | std::views::transform(&HalideAccess::Constraint::inequality),
                std::logical_and<>{}
            );
        };
        std::optional<Halide::Expr> likelyCondition = getConstraints(true), unlikelyCondition = getConstraints(false);
        if (unlikelyCondition) {
            rhs = Halide::select(*unlikelyCondition, rhs, 0.0f);
        }
        if (likelyCondition) {
            rhs = Halide::select(*likelyCondition, Halide::likely(rhs), 0.0f);
        }
    }

    // Perform reduction.
    if (access.reductionDomain.defined()) {
        // Check for rfactor.
        if (!access.rfactoredInnerLoops.empty()) {
            KAS_ASSERT(access.rfactoredInnerLoops.size() == access.rfactoredDomain.dimensions());
            // Create intermediate.
            Halide::Func intermediate { funcName + "_intermediate_reduction" };
            {
                std::vector<Halide::Var> indices = access.outerLoops;
                std::ranges::copy(access.rfactoredInnerLoops, std::back_inserter(indices));
                intermediate(indices) = Halide::sum(access.reductionDomain, rhs, funcName + "_intermediate_reduction_helper");
            }

            // Reduce the intermediate func.
            {
                std::vector<Halide::Expr> indices;
                std::ranges::copy(access.outerLoops, std::back_inserter(indices));
                for (std::size_t rvarId = 0; rvarId < access.rfactoredDomain.dimensions(); ++rvarId) {
                    indices.emplace_back(access.rfactoredDomain[rvarId]);
                }
                rhs = Halide::sum(access.rfactoredDomain, intermediate(indices), funcName + "_rfactor_reduction");
            }
        } else {
            rhs = Halide::sum(access.reductionDomain, rhs, funcName + "_direct_reduction");
        }
    }

    Halide::Func func { funcName };
    func(access.outerLoops) = rhs;
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

HalideGen::ForwardArgsAndFunc HalideGen::createFunc(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName) const {
    std::vector<Halide::ImageParam> inputs;

    const auto& tensors = tensorView.getUnderlyingTensors();
    for (std::size_t inputId = 0; auto&& tensor: tensors) {
        inputs.emplace_back(Halide::type_of<float>(), tensor.getShape().size(), TensorExpression::PositionToString(tensor.getPosition()));
        inputs.back().set_estimates(shapes.inputShapes[inputId]);
        ++inputId;
    }

    Halide::Func out = lowerAccessToFunc(inputs, nullptr, lowerToAccess(consts, tensorView.getForwardAccess()), funcName);
    out.set_estimates(shapes.outputShape);
    return { std::move(inputs), std::move(out) };
}

HalideGen::BackwardArgsAndFuncs HalideGen::createFuncGrad(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName) const {
    std::vector<Halide::ImageParam> inputs;
    std::vector<Halide::Func> inputsGrads;

    for (std::size_t inputId = 0; auto&& tensor: tensorView.getUnderlyingTensors()) {
        inputs.emplace_back(Halide::type_of<float>(), tensor.getShape().size(), TensorExpression::PositionToString(tensor.getPosition()));
        inputs.back().set_estimates(shapes.inputShapes[inputId]);
        ++inputId;
    }

    Halide::ImageParam outputGrad(Halide::type_of<float>(), tensorView.getInterfaceShape().size(), "output_grad");
    outputGrad.set_estimates(shapes.outputShape);

    for (auto&& backward: tensorView.getBackwardAccesses()) {
        inputsGrads.emplace_back(lowerAccessToFunc(inputs, &outputGrad, lowerToAccess(consts, backward), funcName));
        inputsGrads.back().set_estimates(shapes.inputShapes[backward.position]);
    }

    inputs.emplace_back(std::move(outputGrad));
    return { std::move(inputs), std::move(inputsGrads) };
}

HalideGen::ForwardAndBackwardFuncs HalideGen::createPipelines(const ConcreteConsts& consts, std::string_view forwardFuncName, std::string_view backwardFuncName) const {
    const auto& tensors = tensorView.getUnderlyingTensors();

    auto shapes = concretizeShapes(consts);
    try {
        auto [forwardInputs, forwardFunc] = createFunc(consts, shapes, forwardFuncName);
        auto [backwardInputs, backwardFuncs] = createFuncGrad(consts, shapes, backwardFuncName);
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

HalideGen::ScheduledPipelins HalideGen::ApplyAutoScheduler(Halide::Func& forwardFunc, std::vector<Halide::Func>& backwardFuncs, const Halide::Target& target, Options::AutoScheduler scheduler, const std::map<std::string, std::string>& extraOptions, std::ostream *verbose) {
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
        params = { schedulerName, extraOptions };
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
            fmt::format_to(std::ostreambuf_iterator(*verbose), "// Forward pipeline:\n{}\n", forwardResult.schedule_source);
        auto backwardResult = backwardPipeline.apply_autoscheduler(target, params.value());
        if (verbose)
            fmt::format_to(std::ostreambuf_iterator(*verbose), "// Backward pipeline:\n{}\n", backwardResult.schedule_source);
    } else {
        if (verbose) {
            fmt::format_to(std::ostreambuf_iterator(*verbose), "// Forward pipeline:\nfunc.compute_root();\n");
            fmt::format_to(std::ostreambuf_iterator(*verbose), "// Backward pipeline:\nfunc.compute_root();\n");
        }
    }

    return { std::move(forwardPipeline), std::move(backwardPipeline) };
}

void HalideGen::GenerateFromPipelines(std::vector<Halide::ImageParam>& forwardInputs, std::vector<Halide::ImageParam>& backwardInputs, Halide::Pipeline& forwardPipeline, Halide::Pipeline& backwardPipeline, const std::filesystem::path& forwardOutputPath, const std::filesystem::path& backwardOutputPath, std::string_view forwardFuncName, std::string_view backwardFuncName, const Halide::Target& target) {
    // Compile to Halide modules.
    std::vector<Halide::Argument> forwardArgs;
    std::vector<Halide::Argument> backwardArgs;
    std::ranges::copy(forwardInputs, std::back_inserter(forwardArgs));
    std::ranges::copy(backwardInputs, std::back_inserter(backwardArgs));
    auto forwardModule = forwardPipeline.compile_to_module(forwardArgs, std::string(forwardFuncName), target);
    auto backwardModule = backwardPipeline.compile_to_module(backwardArgs, std::string(backwardFuncName), target);

    // Write to output.
    auto ext = Halide::Internal::get_output_info(target);
    const auto flagsForModule = [&ext](const std::filesystem::path& filename) -> std::map<Halide::OutputFileType, std::string> {
        using enum Halide::OutputFileType;
        std::map<Halide::OutputFileType, std::string> files;
        for (auto what: {
            object,
            stmt,
            // stmt_html,
            pytorch_wrapper
        }) {
            auto alteredFilename = filename;
            alteredFilename.replace_extension(ext.at(what).extension);
            files[what] = alteredFilename;
        }
        return files;
    };
    std::filesystem::create_directories(forwardOutputPath.parent_path());
    std::filesystem::create_directories(backwardOutputPath.parent_path());
    forwardModule.compile(flagsForModule(forwardOutputPath));
    backwardModule.compile(flagsForModule(backwardOutputPath));
}

void HalideGen::generate(const std::filesystem::path& forwardOutputPath, const std::filesystem::path& backwardOutputPath, std::string_view forwardFuncName, std::string_view backwardFuncName, const ConcreteConsts& consts, std::ostream *verbose) const {
    // Create Halide Functions.
    auto [forwardInputs, forwardFunc,
        backwardInputs, backwardFuncs
    ] = createPipelines(consts, forwardFuncName, backwardFuncName);

    auto target = GetHostTarget(options.useGPU, false);
    auto [forwardPipeline, backwardPipeline] = ApplyAutoScheduler(forwardFunc, backwardFuncs, target, options.scheduler, options.extraOptions, verbose);

    GenerateFromPipelines(forwardInputs, backwardInputs, forwardPipeline, backwardPipeline, forwardOutputPath, backwardOutputPath, forwardFuncName, backwardFuncName, target);
}

} // namespace kas
