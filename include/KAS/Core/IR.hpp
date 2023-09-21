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

    // Call this before any other call!
    template<DimensionRange R>
    DependentCutSetDiscoverer& includeUnchecked(R&& dimensions) {
        using std::begin;
        using std::end;
        cutSet.insert(begin(std::forward<R>(dimensions)), end(std::forward<R>(dimensions)));
        return *this;
    }

    // Helper function for `include`.
    void excludeUpwards(const Dimension& dimension);
    // Helper function for `fill`.
    bool pushDownwards(const Dimension& dimension);

public:
    // Use the param as the initial cut set, without checking.
    template<DimensionRange R>
    DependentCutSetDiscoverer(const Graph& graph, R&& cutSet): graph(graph) {
        includeUnchecked(std::forward<R>(cutSet));
    }

    // Introduce the new dimension as a dependency.
    DependentCutSetDiscoverer& include(const Dimension& dimension);
    // Go all the way down.
    DependentCutSetDiscoverer& fill();

    // Returns the number of dimensions removed.
    std::size_t removeReductions();

    std::vector<Dimension> build() const;
};

class TensorContractor: protected DependentCutSetDiscoverer {
public:
    // TODO!!! mark all expansions as done.
    template<TensorRange R>
    TensorContractor(const Graph& graph, R&& tensors);

    // Mark this as a new tensor, and perform all possible contractions available so far.
    TensorContractor& contract(const std::vector<Dimension>& tensorOutput);
    template<TensorRange R>
    TensorContractor& contract(R&& tensors) {
        for (auto&& tensor: tensors) {
            contract(std::forward<decltype(tensor)>(tensor));
        }
        return *this;
    }
    // Perform all possible reductions available so far.
    TensorContractor& reduce();
    // Go all the way down.
    TensorContractor& fill() { DependentCutSetDiscoverer::fill(); return *this; }

    // Remove reductions from current cut set.
    TensorContractor& removeReductions() { DependentCutSetDiscoverer::removeReductions(); return *this; }

    std::vector<Dimension> build() const;

    // Helper function.
    template<TensorRange R>
    static std::vector<Dimension> Contract(const Graph& graph, R&& tensors) {
        return TensorContractor(graph, std::forward<R>(tensors)).build();
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
    Tensor& tensor;
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
    std::vector<Tensor> inputTensors;
    Tensor outputTensor;

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
    std::size_t getFLOPs(const BindingContext& ctx) const;
};

struct ContractionScheme {
    // Each item is a contraction group. Moreover, each contraction group corresponds to one Tensor.
    // {{0}, ...} if we can do early reduction, and {{0, ...}, ...} is we cannot do early reduction, in which case we perform contraction right away.
    std::vector<std::vector<std::size_t>> contractions;
};

// There are multiple passes when building the IR.
// initial pass: extract computation graph (i.e., the IR) from the graph, based on a given contraction order.
// rfactor pass: search for rfactor chances to reduce FLOPs.
// layout pass: alter the layout of the tensors to improve locality.
// view pass: move the views to computation graph level, so PyTorchGen can use einsum to contract tensors.
class IRBuilder {
    const std::vector<Topmost>& inputTensors;
    Graph graph;
    // This cuts the graph into subgraphs, based on contraction order.
    // Each subgraph is a Tensor. We here only do this minimally. That is, in each subgraph, all reductions that can be done are done, and only the dependent Dimension's and Op's are kept.
    // Only `outputTensor` can be a view. No views for other `Tensor`s. Other `Tensor`s are contractions or have reductions.
    IR initial(const ContractionScheme& scheme) const;
    // Enumerate rfactor schemes for all subgraphs, and apply the best. Note that the final (possible) view and all the input tensors are not rfactored.
    void rfactor(IR& ir, const BindingContext& ctx) const;
    // Optimize locality.
    void optimizeLayout(IR& ir) const;
    // For PyTorch codegen, further split Tensor's apart so that contractions are apparent, that is, ShareOp's are above any other type of Op's in each Tensor.
    void performViews(IR& ir) const;
public:
    IRBuilder(const std::vector<Topmost>& tensors);
    Generator<ContractionScheme> plausibleContractionSchemes() const;
    // Perform all the passes.
    IR build(const ContractionScheme& scheme, const BindingContext& ctx) const;
    // Helper function.
    static IR Build(const std::vector<Topmost>& tensors, const BindingContext& ctx);
};

} // namespace kas
