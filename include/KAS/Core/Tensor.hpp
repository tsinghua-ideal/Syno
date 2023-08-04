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
    void propagateTo(VisitedVertex vertex) {
        vertex.match(*this, *this, *this);
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
struct Subgraphs;

class Tensor {
    friend class TensorImpl;

    std::shared_ptr<TensorImpl> inner;
    Tensor(std::shared_ptr<TensorImpl> inner): inner(std::move(inner)) {}
public:
    Tensor() = default;

    class Builder {
        friend class TensorImpl;

        const Graph& graph;
        std::map<Tensor, std::vector<const Expand *>> expansions;
        std::map<Dimension, Tensor, Dimension::AddressLessThan> owner;
        // Discover Share blocks to determine the contractions. Return the contracted interface as well.
        std::pair<std::vector<Tensor>, std::vector<Dimension>> findTensorsWhichWeNeedToContract(const Tensor& tensor) const;

    public:
        Builder(const Graph& graph): graph(graph) {}
        Subgraphs build(const std::vector<Topmost>& rawInputTensors);
    };

    // So that Tensor can be stored in an std::map.
    std::strong_ordering operator<=>(const Tensor& rhs) const = default;

    const std::vector<Tensor>& inputs() const;
    const std::vector<Dimension>& output() const;
    bool isInputTensor() const;

    std::string toString(const BindingContext& ctx) const;

    // FOR DEBUG USAGE ONLY!
    std::string debugToString() const;
};

struct Subgraphs {
    std::vector<std::vector<const Expand *>> expansions;
    std::vector<Tensor> inputTensors;
    Tensor outputTensor;
};

// Basically a subgraph in Kernel Graph.
// This processes the input tensor in 2 stages.
// 1. Share. The dimensions in `inputs` are shared. That is, `i, i -> i`. In some cases, the output can be reduced, with `MapReduce` originating from `inputs` or ShareOp::output, so we can have something like `ij, i ->`.
// 2. Transform. Apply some views to the tensor.
// Any view that is later reduced, is left to the next Tensor.
class TensorImpl {
    friend class Tensor;
protected:
    // The inputs need to be contracted. Note that `output` may contain `ShareOp::Input` and `MapReduce`.
    // CodeGen needs to figure out how to contract the inputs.
    // An input tensor has no input.
    std::vector<Tensor> inputs;

    // The output. This is the representation of this Tensor, from which we can read the shape.
    // this can contain `MapReduce`, and is the "anchor" of the graph. CodeGen cannot freely mutate the anchors.
    std::vector<Dimension> output;

    // Input tensor.
    template<typename O>
    TensorImpl(O&& output): output { std::forward<O>(output) } {}
    // Tensor view.
    template<typename I, typename O>
    TensorImpl(I&& inputs, O&& output): inputs { std::forward<I>(inputs) }, output { std::forward<O>(output) } {}

public:
    static Tensor CreateInputTensor(Tensor::Builder& builder, const Topmost& dimensions);
    static Tensor CreateTensorView(Tensor::Builder& builder, const std::vector<Tensor>& inputs, std::vector<Dimension> contracted);

    bool isInputTensor() const { return inputs.empty(); }
    void adjustOutputOrder(const std::vector<Dimension>& expectedOutput);
};

} // namespace kas
