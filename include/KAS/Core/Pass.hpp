#pragma once

#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

struct IR;

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
    // This is called whenever `excludeUpwards` handles an Op, before DFS.
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
    void removeSingleReduction(const Reduce *reduction);

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
    bool singleReductionPerStage;
    const std::vector<Dimension> contractedInterface;

    static Generator<Scheme> PlausibleRFactorSchemes(std::vector<const Reduce *> remaining, bool allowEmpty);
    static Generator<Scheme> PlausibleSingleReductionRFactorSchemes(std::vector<const Reduce *> remaining, bool firstEmpty);
    Generator<Scheme> plausibleRFactorSchemes() const;
    // If > overflow, treated as infinity.
    static constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();
    std::size_t getFLOPs(const Scheme& scheme, std::size_t overflow = Infinity) const;

public:
    // Never pass a view here!
    RFactorSolver(Tensor& tensor, const Graph& graph, const BindingContext& ctx, bool singleReductionPerStage);
    std::optional<Scheme> optimalRFactorScheme() const;
    void apply(const Scheme& scheme);
};

class RFactorIRPass {
    const BindingContext& ctx;
    const Graph& graph;
    bool singleReductionPerStage;
public:
    RFactorIRPass(const BindingContext& ctx, const Graph& graph, bool singleReductionPerStage = false);
    void operator()(IR& ir) const;
};

// Optimize the layout of all the Tensor's, except the input and output tensors.
class OptimizeLayoutIRPass {
    const Graph& graph;
    void optimize(Tensor& tensor) const;
public:
    OptimizeLayoutIRPass(const Graph& graph);
    void operator()(IR& ir) const;
};

} // namespace kas
