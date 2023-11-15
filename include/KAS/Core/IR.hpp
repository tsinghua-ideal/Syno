#pragma once

#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

struct IR {
    std::vector<std::vector<const Expand *>> expansions;
    std::vector<Tensor> inputTensors;
    Tensor outputTensor;

    explicit operator bool() const { return static_cast<bool>(outputTensor); }

    // Helper function.
    static IR Build(const std::vector<Topmost>& tensors, const BindingContext& ctx, bool constractOneTensorEachTime = false);

    template<bool Const = true, bool PostOrder = false, typename F>
    void forEachHelper(F&& f) const {
        std::set<Tensor> visited;
        auto dfs = [&](const auto& self, const Tensor& tensor) {
            auto [_, inserted] = visited.insert(tensor);
            if (!inserted) return;
            if constexpr (PostOrder) {
                for (const Tensor& input: tensor.inputs()) {
                    self(self, input);
                }
            }
            if constexpr (Const) {
                std::invoke(std::forward<F>(f), tensor);
            } else {
                std::invoke(std::forward<F>(f), const_cast<Tensor&>(tensor));
            }
            if constexpr (!PostOrder) {
                for (const Tensor& input: tensor.inputs()) {
                    self(self, input);
                }
            }
        };
        dfs(dfs, outputTensor);
    }
    void bottomTopForEach(std::invocable<const Tensor&> auto&& f) const { forEachHelper<true, false>(std::forward<decltype(f)>(f)); }
    void bottomTopForEach(std::invocable<Tensor&> auto&& f) { forEachHelper<false, false>(std::forward<decltype(f)>(f)); }
    void topBottomForEach(std::invocable<const Tensor&> auto&& f) const { forEachHelper<true, true>(std::forward<decltype(f)>(f)); }
    void topBottomForEach(std::invocable<Tensor&> auto&& f) { forEachHelper<false, true>(std::forward<decltype(f)>(f)); }

    IR copy() const;

    Graph buildGraph() const;
    std::size_t getFLOPs(const BindingContext& ctx, const ConcreteConsts& consts) const;
    std::size_t getFLOPs(const BindingContext& ctx) const;
    std::size_t numStages() const;

    KAS_STATISTICS_DEF(
        EqualFLOPs,
        WithInterdependentShares,
    )
};

struct ContractionScheme {
    // Each item is a contraction group. Moreover, each contraction group corresponds to one Tensor.
    // {{}, ...} if we can do early reduction, and {{...}, ...} is we cannot do early reduction, in which case we perform contraction right away.
    std::vector<std::vector<std::size_t>> contractions;
    ContractionScheme() = default;
    template<std::convertible_to<std::vector<std::vector<std::size_t>>> T>
    explicit ContractionScheme(T&& contractions): contractions(std::forward<T>(contractions)) {}
    ContractionScheme(std::initializer_list<std::vector<std::size_t>> contractions): contractions(contractions) {}
    ContractionScheme& cons(const std::vector<std::size_t>& contraction) {
        contractions.insert(contractions.begin(), contraction);
        return *this;
    }
    bool contractMoreThanOneTensorEachTime() const;
};

// There are multiple passes when building the IR.
// initial pass: extract computation graph (i.e., the IR) from the graph, based on a given contraction order.
// rfactor pass: search for rfactor chances to reduce FLOPs.
// layout pass: alter the layout of the tensors to improve locality.
// view pass: move the views to computation graph level, so PyTorchGen can use einsum to contract tensors. This is done by PyTorchGen.
class IRBuilder {
    const std::vector<Topmost>& inputTensors;
    Graph graph;

    Generator<ContractionScheme> plausibleContractionSchemes(const std::vector<std::vector<bool>>& laterThan, std::vector<std::size_t> remaining) const;

public:
    IRBuilder(const std::vector<Topmost>& tensors);
    Generator<ContractionScheme> plausibleContractionSchemes(bool constractOneTensorEachTime) const;

    // This cuts the graph into subgraphs, based on contraction order.
    // Each subgraph is a Tensor. We here only do this minimally. That is, in each subgraph, all reductions that can be done are done, and only the dependent Dimension's and Op's are kept.
    // Only `outputTensor` can be a view. No views for other `Tensor`s. Other `Tensor`s are contractions or have reductions.
    IR initial(const ContractionScheme& scheme) const;
    // Enumerate rfactor schemes for all subgraphs, and apply the best. Note that the final (possible) view and all the input tensors are not rfactored.
    void rfactor(IR& ir, const BindingContext& ctx) const;
    // Optimize locality.
    void optimizeLayout(IR& ir) const;

    // Perform all the passes.
    IR build(const ContractionScheme& scheme, const BindingContext& ctx) const;
};

} // namespace kas
