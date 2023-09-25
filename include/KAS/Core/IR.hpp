#pragma once

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class DependentCutSetDiscoverer {
protected:
    const Graph& graph;
    Graph::CutSet cutSet;

    bool tryInsert(const Dimension& dim) { return cutSet.insert(dim).second; }
    bool tryErase(const Dimension& dim) { return cutSet.erase(dim) > 0; }
    void assertInsert(const Dimension& dim) { auto res = tryInsert(dim); KAS_ASSERT(res); }
    void assertErase(const Dimension& dim) { auto res = tryErase(dim); KAS_ASSERT(res); }

    // Call this if you are sure the dimensions has no dependency with the existing cut set.
    // Returns the number of dimensions inserted.
    template<DimensionRange R>
    std::size_t includeUnchecked(R&& dimensions) {
        const auto originalSize = cutSet.size();
        std::ranges::copy(std::forward<R>(dimensions), std::inserter(cutSet, cutSet.begin()));
        return cutSet.size() - originalSize;
    }

    // Helper function for `include`.
    void excludeUpwards(const Dimension& dimension);
    // This is called whenever `excludeUpwards` handles an Op, before DFS..
    virtual void beforeExclusionHook(const PrimitiveOp *op) {}
    // This is called whenever `excludeUpwards` handles an Op, after DFS.
    virtual void afterExclusionHook(const PrimitiveOp *op) {}

public:
    DependentCutSetDiscoverer(const Graph& graph): graph(graph) {}
    // Use the param as the initial cut set, without checking.
    template<DimensionRange R>
    DependentCutSetDiscoverer(const Graph& graph, R&& cutSet): DependentCutSetDiscoverer(graph) {
        includeUnchecked(std::forward<R>(cutSet));
    }

    // Introduce a new dimension as a dependency.
    DependentCutSetDiscoverer& include(const Dimension& dimension);
    // Introduce new dimensions as dependencies.
    template<DimensionRange R>
    DependentCutSetDiscoverer& include(R&& dimensions) {
        for (auto&& dimension: dimensions) {
            include(std::forward<decltype(dimension)>(dimension));
        }
        return *this;
    }

    // Returns the number of dimensions removed.
    std::size_t removeReductions();

    std::vector<Dimension> build() const;
};

class TensorContractor: protected DependentCutSetDiscoverer {
protected:
    Graph::CompactIndices collected;
    std::set<const Reduce *> doneReductions;

    // Add a tensor, and return its ancestors.
    Graph::CompactIndices add(const std::vector<Dimension>& tensorOutput);
    // Store the remaining ShareOp's to be contracted.
    std::set<const MergeLikeOp *> allowedShareOps;
    // Find Share's where `collected` < shareOp.output.ancestors <= `collected + targets`.
    void performContractions(Graph::CompactIndices targets);
    // This checks for ShareOp's.
    void beforeExclusionHook(const PrimitiveOp *op) override;

public:
    // Mark all expansions as collected as well.
    TensorContractor(const Graph& graph, const std::vector<Dimension>& current);

    // Mark these as new tensors, and perform all possible contractions available so far.
    template<std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, std::vector<Dimension>>
    TensorContractor& contract(R&& tensors) {
        auto features = Graph::CompactIndices::None();
        for (auto&& tensor: std::forward<R>(tensors)) {
            features.merges(add(std::forward<decltype(tensor)>(tensor)));
        }
        performContractions(features);
        return *this;
    }

    using DependentCutSetDiscoverer::build;

    // The below 3 functions are used by IRBuilder.
    // Perform all possible reductions available so far.
    // To use this function, you had better start from the input.
    // Because we need to collect all the reductions.
    TensorContractor& reduce();
    // Go all the way down, but do not allow any Share.
    TensorContractor& fill();
    // Remove reductions from current cut set.
    TensorContractor& removeReductions();

    // Helper function.
    template<std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, std::vector<Dimension>>
    static std::vector<Dimension> Contract(const Graph& graph, R&& tensors) {
        using std::begin;
        using std::end;
        auto b = begin(tensors);
        auto e = end(tensors);
        KAS_ASSERT(b != e, "Cannot contract 0 tensors.");
        auto contractor = TensorContractor(graph, *b);
        ++b;
        auto features = Graph::CompactIndices::None();
        for (; b != e; ++b) {
            features.merges(contractor.add(*b));
        }
        contractor.performContractions(features);
        return contractor.build();
    }
};

class RFactorSolver {
public:
    struct Scheme {
        // Empty reduction group in the front means no reductions. This happens only for contraction.
        std::vector<std::vector<const Reduce *>> reductions;
        Scheme() = default;
        template<std::convertible_to<std::vector<std::vector<const Reduce *>>> T>
        explicit Scheme(T&& reductions): reductions(std::forward<T>(reductions)) {}
        Scheme(std::initializer_list<std::vector<const Reduce *>> reductions): reductions(reductions) {}
        Scheme& cons(const std::vector<const Reduce *>& reduction) {
            reductions.insert(reductions.begin(), reduction);
            return *this;
        }
    };

private:
    Tensor tensor;
    const Graph& graph;
    const BindingContext& ctx;
    const std::vector<Dimension> contractedInterface;

    static Generator<Scheme> PlausibleRFactorSchemes(std::vector<const Reduce *> remaining, bool allowEmpty);
    Generator<Scheme> plausibleRFactorSchemes() const;
    // If > overflow, treated as infinity.
    static constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();
    std::size_t getFLOPs(const Scheme& scheme, std::size_t overflow = Infinity) const;

public:
    // Never pass a view here!
    RFactorSolver(Tensor& tensor, const Graph& graph, const BindingContext& ctx);
    std::optional<Scheme> optimalRFactorScheme(std::size_t overflow) const;
    Scheme optimalRFactorScheme() const { return optimalRFactorScheme(Infinity).value(); }
    void apply(const Scheme& scheme);
};

struct IR {
    std::vector<std::vector<const Expand *>> expansions;
    std::vector<Tensor> inputTensors;
    Tensor outputTensor;

    // Helper function.
    static IR Build(const std::vector<Topmost>& tensors, const BindingContext& ctx);

    template<bool isConst = true, typename F>
    void forEachHelper(F&& f) const {
        std::set<Tensor> visited;
        auto dfs = [&](const auto& self, const Tensor& tensor) {
            auto [_, inserted] = visited.insert(tensor);
            if (!inserted) return;
            if constexpr (isConst) {
                std::invoke(std::forward<F>(f), tensor);
            } else {
                std::invoke(std::forward<F>(f), const_cast<Tensor&>(tensor));
            }
            for (const Tensor& input: tensor.inputs()) {
                self(self, input);
            }
        };
        dfs(dfs, outputTensor);
    }
    void forEach(std::invocable<const Tensor&> auto&& f) const { forEachHelper<true>(std::forward<decltype(f)>(f)); }
    void forEach(std::invocable<Tensor&> auto&& f) { forEachHelper<false>(std::forward<decltype(f)>(f)); }

    IR copy() const;

    Graph buildGraph() const;
    std::size_t getFLOPs(const BindingContext& ctx) const;
    std::size_t numStages() const;
};

struct ContractionScheme {
    // Each item is a contraction group. Moreover, each contraction group corresponds to one Tensor.
    // {{0}, ...} if we can do early reduction, and {{0, ...}, ...} is we cannot do early reduction, in which case we perform contraction right away.
    std::vector<std::vector<std::size_t>> contractions;
    ContractionScheme() = default;
    template<std::convertible_to<std::vector<std::vector<std::size_t>>> T>
    explicit ContractionScheme(T&& contractions): contractions(std::forward<T>(contractions)) {}
    ContractionScheme(std::initializer_list<std::vector<std::size_t>> contractions): contractions(contractions) {}
    ContractionScheme& cons(const std::vector<std::size_t>& contraction) {
        contractions.insert(contractions.begin(), contraction);
        return *this;
    }
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
    Generator<ContractionScheme> plausibleContractionSchemes() const;

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

class LayoutOptimizer {

};

} // namespace kas
