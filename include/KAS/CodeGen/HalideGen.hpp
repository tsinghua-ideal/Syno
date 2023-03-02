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
    FRIEND_TEST(codegen_tests, func);
    FRIEND_TEST(forward_tests, pooling);
    FRIEND_TEST(search_tests, sampler);
    friend class transforms_tests;
public:
    struct Options {
        enum class AutoScheduler {
            ComputeRoot, Mullapudi2016, Li2018, Adams2019, Anderson2021,
        };
        bool useGPU = true;
        AutoScheduler scheduler = AutoScheduler::Li2018;
        bool zeroPadding = false; // If set to false, use replicate padding, which reduces some computation.
    };

private:
    static void GuardAutoSchedulers(); // Load the Halide auto scheduler plugins.

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
        std::vector<Halide::Expr>& constraintsBounds;
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
        inline HalideExprEvaluator(const ConcreteConsts& consts, Cache& cache, std::vector<Halide::Expr>& constraintsBounds, const std::vector<Halide::RVar>& outerIteratorsAsRVars, bool useRVar, const std::vector<Halide::RVar>& innerIterators, const HalideGen& parent):
            consts { consts },
            cache { cache },
            constraintsBounds { constraintsBounds },
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
        // This `evaluate` produces a condition expression.
        Halide::Expr evaluate(const IntervalBoundValueNode& value);
    };
    Halide::Expr evaluate(const ConcreteConsts& consts, HalideExprEvaluator::Cache& cache, std::vector<Halide::Expr>& constraintsBounds, const std::vector<Halide::RVar>& outerIteratorsAsRVars, bool useRVar, const std::vector<Halide::RVar>& innerIterators, const IteratorValue& value) const;
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
        std::vector<Halide::Expr> constraintsBounds;
    };
    ConcreteConsts realizeConsts(const std::map<std::string, std::size_t>& mappings) const;
    EvaluatedAccess evaluateAccess(const ConcreteConsts& consts, bool useRVar) const;

    using ForwardArgsAndFunc = std::pair<std::vector<Halide::ImageParam>, Halide::Func>;
    // Returns the (input, func) pair.
    ForwardArgsAndFunc createFunc(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName, bool zeroBoundary = false, bool useRVars = false);
    using BackwardArgsAndFuncs = std::pair<std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the (input (including the gradient of output), gradient funcs) pair.
    BackwardArgsAndFuncs createFuncGrad(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName);
    using ForwardAndBackwardFuncs = std::tuple<std::vector<Halide::ImageParam>, Halide::Func, std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the forward and backward funcs.
    ForwardAndBackwardFuncs createPipelines(const std::map<std::string, std::size_t>& mappings, std::string_view funcName, bool useRVar = false);

    static Halide::Target GetHostTarget(bool useGPU);

public:
    HalideGen(const BindingContext& ctx, const TensorView& tensorView, Options options);

    void generate(std::filesystem::path outputPath, std::string_view funcName, const std::map<std::string, std::size_t>& mappings);

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
