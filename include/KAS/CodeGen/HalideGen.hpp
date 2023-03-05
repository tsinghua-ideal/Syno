#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>
#include "Halide.h"

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

class HalideGen {
    friend class forward_tests;
public:
    struct Options {
        enum class AutoScheduler {
            ComputeRoot, Mullapudi2016, Li2018, Adams2019, Anderson2021,
        };
        bool useGPU = true;
        AutoScheduler scheduler = AutoScheduler::Li2018;
        // bool zeroPadding = false; // If set to false, use replicate padding, which reduces some computation.
        // Here we use replicate padding. Because zero padding is just so hard a case.
    };

private:
    const BindingContext& ctx;
    const TensorView& tensorView;
    Options options;

    std::vector<Halide::Var> outerIterators;
    inline Halide::Var getOuterIterator(std::size_t index) const {
        // Due to the column-major order of Halide, we need to reverse the order of indices.
        return outerIterators.at(outerIterators.size() - 1 - index);
    }

    struct HalideExprEvaluator: public IteratorValueVisitor {
        using Cache = std::map<IteratorValue, Halide::Expr>;
        Halide::Expr evaluator;
        const ConcreteConsts& consts;
        Cache& cache;
        // When computing gradient, to mitigate Halide bugs, we need to use RVar.
        const std::vector<Halide::RVar>& outerIteratorsAsRVars; // In reverse order.
        bool useRVar;
        inline Halide::RVar getOuterIteratorAsRVar(std::size_t index) const {
            // Due to the column-major order of Halide, we need to revert the order of indices.
            return outerIteratorsAsRVars.at(outerIteratorsAsRVars.size() - 1 - index);
        }
        const std::vector<Halide::RVar>& innerIterators; // In original order.
        inline Halide::RVar getInnerIterator(std::size_t index) const {
            // Inner iterators are in order, so we don't need to reverse the order of indices.
            return innerIterators.at(index);
        }
        const HalideGen& parent;
        inline HalideExprEvaluator(const ConcreteConsts& consts, Cache& cache, const std::vector<Halide::RVar>& outerIteratorsAsRVars, bool useRVar, const std::vector<Halide::RVar>& innerIterators, const HalideGen& parent):
            consts { consts },
            cache { cache },
            outerIteratorsAsRVars { outerIteratorsAsRVars },
            useRVar { useRVar },
            innerIterators { innerIterators },
            parent { parent }
        {}
        void visit(VariableValueNode& value) override;
        void visit(ConstValueNode& value) override;
        void visit(ImmediateValueNode& value) override;
        void visit(BinaryOpValueNode& value) override;
        // This `visit` produces a `clamp`.
        void visit(IntervalBoundValueNode& value) override;
        Halide::Expr evaluate(const IteratorValue& value);
    };
    Halide::Expr evaluate(const ConcreteConsts& consts, HalideExprEvaluator::Cache& cache, const std::vector<Halide::RVar>& outerIteratorsAsRVars, bool useRVar, const std::vector<Halide::RVar>& innerIterators, const IteratorValue& value) const;
    Halide::Expr evaluate(const ConcreteConsts& consts, const Size& value) const;
    template<typename Storage, auto Mapping>
    Halide::Region evaluate(const ConcreteConsts& consts, const AbstractShape<Storage, Mapping>& shape, bool reverse = true) const {
        Halide::Region region;
        for (const auto& s: shape) {
            region.emplace_back(0, evaluate(consts, s));
        }
        if (reverse) {
            // Because Halide uses column-major, we need to reverse the order of indices.
            std::ranges::reverse(region);
        }
        return region;
    }

    struct EvaluatedAccess {
        std::vector<Halide::RVar> outerIteratorsAsRVars; // Reverse order!
        std::vector<Halide::RVar> innerIterators; // Since reduce loops are in order, we should not reverse them.
        std::vector<std::vector<Halide::Expr>> tensorIndices;
        // The conditions that must be satisfied in order that no out-of-bound error occurs. This is used to enforce zero-padding semantics.
        // std::vector<Halide::Expr> constraintsBounds;
        inline std::vector<Halide::Expr> outerIteratorsAsRVarsToExprs() const {
            std::vector<Halide::Expr> result;
            for (const auto& rvar: outerIteratorsAsRVars) {
                result.emplace_back(rvar);
            }
            return result;
        }
    };
    ConcreteConsts realizeConsts(const std::map<std::string, std::size_t>& mappings) const;
    EvaluatedAccess evaluateAccess(const ConcreteConsts& consts, bool useRVar) const;

public:
    static Halide::Target GetHostTarget(bool useGPU);

    static void GuardAutoSchedulers(); // Load the Halide auto scheduler plugins.

    using ForwardArgsAndFunc = std::pair<std::vector<Halide::ImageParam>, Halide::Func>;
    // Returns the (input, func) pair.
    ForwardArgsAndFunc createFunc(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName, bool zeroBoundary = false, bool useRVars = false);
    using BackwardArgsAndFuncs = std::pair<std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the (input (including the gradient of output), gradient funcs) pair.
    BackwardArgsAndFuncs createFuncGrad(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName);
    using ForwardAndBackwardFuncs = std::tuple<std::vector<Halide::ImageParam>, Halide::Func, std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the forward and backward funcs.
    ForwardAndBackwardFuncs createPipelines(const std::map<std::string, std::size_t>& mappings, std::string_view funcName, bool useRVar = false);
    // Applys auto-schedulers on the funcs.
    using ScheduledPipelins = std::pair<Halide::Pipeline, Halide::Pipeline>;
    static ScheduledPipelins ApplyAutoScheduler(Halide::Func& forwardFunc, std::vector<Halide::Func>& backwardFuncs, const Halide::Target& target, Options::AutoScheduler scheduler, bool verbose);
    static void GenerateFromPipelines(std::vector<Halide::ImageParam>& forwardInputs, std::vector<Halide::ImageParam>& backwardInputs, Halide::Pipeline& forwardPipeline, Halide::Pipeline& backwardPipeline, std::filesystem::path outputPath, std::string_view funcName, const Halide::Target& target);

    HalideGen(const BindingContext& ctx, const TensorView& tensorView, Options options);

    void generate(std::filesystem::path outputPath, std::string_view funcName, const std::map<std::string, std::size_t>& mappings);

    // Do the reversal for users.
    std::vector<int> getInputBufferShape(const ConcreteConsts& consts, std::size_t index) const;
    std::vector<std::vector<int>> getInputBuffersShapes(const ConcreteConsts& consts) const;
    std::vector<int> getOutputBufferShape(const ConcreteConsts& consts) const;

    // This is a workaround for reverse indexing caused by Halide's column-major buffers.
    template<typename BufferType>
    struct ReverseAdaptor {
        BufferType content;
        auto operator()(const std::vector<int>& indices) -> decltype(auto) {
            std::vector<int> indicesCopy(indices.rbegin(), indices.rend());
            return content(indicesCopy);
        }
        template<typename... Args>
        requires(Halide::Runtime::AllInts<Args...>::value)
        auto operator()(Args... args) -> decltype(auto) {
            auto helper = [this]<typename... Params>(Params&&... params) -> decltype(auto) {
                return content(std::forward<Params>(params)...);
            };
            return std::apply(helper, ReverseTuple(std::forward_as_tuple(std::forward<Args>(args)...)));
        }
    };

    template<typename T = void, int Dims = Halide::AnyDims>
    using BufferAdaptor = ReverseAdaptor<Halide::Buffer<T, Dims>>;

    template<typename T = void, int Dims = Halide::AnyDims>
    using BufferRefAdaptor = ReverseAdaptor<Halide::Buffer<T, Dims>&>;

};

} // namespace kas
