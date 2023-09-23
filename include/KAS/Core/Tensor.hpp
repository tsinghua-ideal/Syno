#pragma once

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Graph.hpp"


namespace kas {

template<typename F>
requires std::invocable<F, const Dimension&>
class ShareBlockDiscoverer {
    const Graph& graph;
    Dimension bottommost;
    F f;
public:
    void operator()(const RepeatLikeVertex&, auto) {}
    void operator()(const SplitLikeVertex&, auto) {}
    void operator()(const MergeLikeVertex& vertex, MergeLikeOp::Branch) {
        if (vertex.op.getType() != DimensionType::Share) {
            return;
        }
        // This is a ShareOp. First collect.
        f(vertex[MergeLikeOp::Branch::InputLhs]);
        f(vertex[MergeLikeOp::Branch::InputRhs]);
        // Then propagate.
        propagateTo(vertex.visitAdjacent(MergeLikeOp::Branch::InputLhs));
        propagateTo(vertex.visitAdjacent(MergeLikeOp::Branch::InputRhs));
    }
    void operator()(const ExpandVertex&, auto) {}
    void propagateTo(VisitedVertex vertex) {
        vertex.match(*this);
    }
    ShareBlockDiscoverer(const Graph& graph, Dimension dim, F&& f):
        graph { graph },
        bottommost { dim }, // temporary.
        f { std::forward<F>(f) }
    {
        // First find the bottom-most Share dimension.
        while (dim.type() == DimensionType::Share) {
            dim = dim.as<MergeLikeOp::Input>().getOp()->output;
        }
        bottommost = dim;
    }
    Dimension getBottommost() const { return bottommost; }
    Dimension traverse() {
        f(bottommost);
        propagateTo(graph.visitAlong(bottommost, Direction::Up));
        return bottommost;
    }
};

class TensorImpl;

class Tensor {
    friend class TensorImpl;

    std::shared_ptr<TensorImpl> inner;
    Tensor(std::shared_ptr<TensorImpl> inner): inner(std::move(inner)) {}
public:
    Tensor() = default;

    // Check if empty.
    explicit operator bool() const { return static_cast<bool>(inner); }

    // So that Tensor can be stored in an std::map.
    std::strong_ordering operator<=>(const Tensor& rhs) const = default;

    const std::vector<Tensor>& inputs() const;
    const std::vector<Dimension>& output() const;
    const std::vector<const Reduce *>& reductions() const;

    std::vector<Tensor>& getInputs() { return const_cast<std::vector<Tensor>&>(inputs()); }
    std::vector<Dimension>& getOutput() { return const_cast<std::vector<Dimension>&>(output()); }
    std::vector<const Reduce *>& getReductions() { return const_cast<std::vector<const Reduce *>&>(reductions()); }

    void adjustLayout(const std::vector<Dimension> *expectedOutput, const std::vector<const Reduce *> *expectedReductions);

    std::size_t getFLOPs(const BindingContext& ctx) const;

    // No input tensors. Only when this is the case, output can contain Reduce's.
    bool isInputTensor() const { return inputs().empty(); }
    // >= 2 input tensors.
    bool hasContraction() const { return inputs().size() >= 2; }
    // >= 1 reductions.
    bool hasReduction() const { return !reductions().empty(); }
    // Exactly one input tensor, and no reductions.
    bool isView() const { return inputs().size() == 1 && !hasReduction(); }

    std::string toString(const BindingContext& ctx) const;

    // FOR DEBUG USAGE ONLY!
    std::string debugToString() const;
};

class IRBuilder;

// Basically a subgraph in Kernel Graph.
// This processes the input tensor in 3 stages.
// 1. Share. The dimensions in `inputs` are shared. That is, `i, i -> i`. In some cases, the output can be reduced, with `Reduce` originating from `inputs` or ShareOp::output, so we can have something like `ij, i ->`.
// 2. Transform. Apply some views to the tensor.
// 3. Reduce. The new `Reduce`s originating from the views are are reduced.
// Note that the `Reduce`s present in `reductions` can be from Share and Transform. CodeGen needs to handle that. Simply put, loop-level codegen do not need to bother with this, but PyTorch codgen must handle this with care. 
class TensorImpl {
    friend class Tensor;
    friend class IRBuilder;

    // The inputs need to be contracted. Note that `output` may contain `ShareOp::Input`, and `Reduce` if this is an input tensor.
    // CodeGen needs to figure out how to contract the inputs.
    // An input tensor has no input.
    std::vector<Tensor> inputs;

    // The output. This is the representation of this Tensor, from which we can read the shape.
    // this can contain `Reduce` if this is an input tensor, and is the "anchor" of the graph. CodeGen cannot freely mutate the anchors.
    std::vector<Dimension> output;

    // The reductions. If this is an input tensor, there is no reduction.
    std::vector<const Reduce *> reductions;

    // Input tensor.
    template<std::convertible_to<std::vector<Dimension>> O>
    TensorImpl(O&& output): output(std::forward<O>(output)) {}
    template<
        std::convertible_to<std::vector<Tensor>> I,
        std::convertible_to<std::vector<Dimension>> O,
        std::convertible_to<std::vector<const Reduce *>> R
    >
    TensorImpl(I&& inputs, O&& output, R&& reductions):
        inputs(std::forward<I>(inputs)),
        output(std::forward<O>(output)),
        reductions(std::forward<R>(reductions)) {}
    // Tensor view.
    template<std::convertible_to<std::vector<Tensor>> I>
    TensorImpl(I&& inputs, const Bottommost& bottommost):
        inputs(std::forward<I>(inputs)),
        output(bottommost.getOutput()),
        reductions(bottommost.getReductions()) {}

public:
    template<std::convertible_to<std::vector<Dimension>> O>
    static Tensor CreateInput(O&& output) {
        return Tensor(std::shared_ptr<TensorImpl>(new TensorImpl(std::forward<O>(output))));
    }
    template<
        std::convertible_to<std::vector<Tensor>> I,
        std::convertible_to<std::vector<Dimension>> O,
        std::convertible_to<std::vector<const Reduce *>> R
    >
    static Tensor CreateView(I&& inputs, O&& output, R&& reductions) {
        return Tensor(std::shared_ptr<TensorImpl>(new TensorImpl(std::forward<I>(inputs), std::forward<O>(output), std::forward<R>(reductions))));
    }
    template<std::convertible_to<std::vector<Tensor>> I>
    static Tensor CreateView(I&& inputs, const Bottommost& bottommost) {
        return Tensor(std::shared_ptr<TensorImpl>(new TensorImpl(std::forward<I>(inputs), bottommost)));
    }
};

} // namespace kas
