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

class HalideGen: public IteratorValueVisitor {
    FRIEND_TEST(codegen_tests, func);
    FRIEND_TEST(search_tests, sample);
    friend class transforms_tests;

    static bool AutoSchedulerLoaded;

protected:
    const BindingContext& ctx;
    const CodeGenContext& cgCtx;
    const TensorView& tensorView;

    std::vector<Halide::Param<int>> primaryConsts;
    std::vector<Halide::Param<int>> coefficientConsts;
    std::vector<Halide::Var> vars;

    std::stack<Halide::Expr> stack;
    void visit(VariableValueNode& value) override;
    void visit(ConstValueNode& value) override;
    void visit(ImmediateValueNode& value) override;
    void visit(BinaryOpValueNode& value) override;
    Halide::Expr evaluate(const IteratorValue& value);
    Halide::Expr evaluate(std::shared_ptr<Size> value);
    Halide::Region evaluate(const Shape& shape);

    bool accessEvaluated = false;
    std::vector<Halide::Var> outerLoops;
    std::vector<Halide::Var> reduceLoops;
    std::vector<std::vector<Halide::Expr>> tensorIndices;
    void evaluateAccess();

    // Returns the (input, func) pair.
    std::pair<std::vector<Halide::ImageParam>, Halide::Func> createFunc(std::string_view funcName);
    // Returns the (input (including the gradient of output), gradient funcs) pair.
    std::pair<std::vector<Halide::ImageParam>, std::vector<Halide::Func>> createFuncGrad(std::string_view funcName);

    static Halide::Region ConcreteShapeToRegion(const std::vector<std::size_t>& estimate);

    static Halide::Target GetTarget(bool useGPU);

public:
    HalideGen(const BindingContext& ctx, const TensorView& tensorView);

    struct Options {
        enum class AutoScheduler {
            ComputeRoot, Mullapudi2016, Li2018, Adams2019
        };
        bool useGPU = true;
        AutoScheduler scheduler = AutoScheduler::Li2018;
    };

    void generate(std::filesystem::path outputPath, std::string_view funcName, Options options);
    // Requires all variables to be assigned.
    Halide::ParamMap getParamMap(const std::map<std::string, std::size_t>& mappings) const;

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
