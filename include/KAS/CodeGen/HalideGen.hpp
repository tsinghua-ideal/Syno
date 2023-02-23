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
    FRIEND_TEST(search_tests, sample);
    friend class transforms_tests;

    static bool AutoSchedulerLoaded;

protected:
    const BindingContext& ctx;
    const TensorView& tensorView;

    std::vector<Halide::Var> vars;
    std::vector<Halide::Var> outerIterators;
    std::vector<Halide::Var> innerIterators;

    struct HalideExprEvaluator: public IteratorValueVisitor {
        using Cache = std::map<IteratorValue, Halide::Expr>;
        Halide::Expr evaluator;
        const ConcreteConsts& consts;
        Cache& cache;
        std::vector<Halide::Expr>& constraintsBounds;
        const HalideGen& parent;
        HalideExprEvaluator(const ConcreteConsts& consts, Cache& cache, std::vector<Halide::Expr>& constraintsBounds, const HalideGen& parent): consts { consts }, cache { cache }, constraintsBounds { constraintsBounds }, parent { parent } {}
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
    Halide::Expr evaluate(const ConcreteConsts& consts, HalideExprEvaluator::Cache& cache, std::vector<Halide::Expr>& constraintsBounds, const IteratorValue& value) const;
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
        std::vector<std::vector<Halide::Expr>> tensorIndices;
        // The conditions that must be satisfied in order that no out-of-bound error occurs. This is used to enforce zero-padding semantics.
        std::vector<Halide::Expr> constraintsBounds;
    };
    ConcreteConsts realizeConsts(const std::map<std::string, std::size_t>& mappings) const;
    EvaluatedAccess evaluateAccess(const ConcreteConsts& consts) const;

    // Returns the (input, func) pair.
    std::pair<std::vector<Halide::ImageParam>, Halide::Func> createFunc(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName, bool zeroBoundary = false);
    // Returns the (input (including the gradient of output), gradient funcs) pair.
    std::pair<std::vector<Halide::ImageParam>, std::vector<Halide::Func>> createFuncGrad(const ConcreteConsts& consts, const EvaluatedAccess& access, std::string_view funcName);

    static Halide::Target GetHostTarget(bool useGPU);

public:
    HalideGen(const BindingContext& ctx, const TensorView& tensorView);

    struct Options {
        enum class AutoScheduler {
            ComputeRoot, Mullapudi2016, Li2018, Adams2019
        };
        bool useGPU = true;
        AutoScheduler scheduler = AutoScheduler::Li2018;
    };

    void generate(std::filesystem::path outputPath, std::string_view funcName, const std::map<std::string, std::size_t>& mappings, Options options);

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
