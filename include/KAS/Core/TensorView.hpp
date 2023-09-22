#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Expand.hpp"
#include "KAS/Core/IR.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class PureTensor {
protected:
    TensorExpression::Position position;
    Topmost content;
public:
    PureTensor(TensorExpression::Position position, auto&& dims, auto&&expands):
        position { position },
        content(
            std::forward<decltype(dims)>(dims),
            std::forward<decltype(expands)>(expands)
        )
    {}
    PureTensor(TensorExpression::Position position, auto&& topmost):
        position { position },
        content { std::forward<decltype(topmost)>(topmost) }
    {}
    bool operator==(const PureTensor& rhs) const {
        return position == rhs.position && content == rhs.content;
    }
    const TensorExpression::Position& getPosition() const { return position; }
    const Topmost& getContent() const { return content; }
    const std::vector<Dimension>& getDims() const { return content.getDimensions(); }
    const std::vector<const Expand *>& getExpands() const { return content.getExpansions(); }
    ShapeView getShape() const { return content.getShape(); }
    std::string shapeToString(const BindingContext& ctx) const;
    std::string description(const BindingContext& ctx) const;
};

struct AbstractAccess {
    TensorExpression::Position position;
    std::vector<IteratorValue> outerLoops;
    std::vector<IteratorValue> innerLoops;
    Shape innerLoopsShape;
    std::vector<std::vector<IteratorValue>> inputs;
    std::vector<IteratorValue> output;

    // Description of the expression.
    TensorExpression expression; // The actual expression.
    std::optional<Size> divBy; // The divisor, if any.

    bool isDerivative() const { return position != TensorExpression::Output; }

    // The standard interface, i.e., the outer loops.
    std::string outerLoopsIteratorsToString() const;

    // The inner reduction loops.
    std::string innerLoopsIteratorsToString() const;

    // The access of the tensor at `pos`.
    std::string accessToString(const BindingContext& ctx, int pos) const;

    // The innermost statement.
    std::string statementToString(const BindingContext& ctx) const;

    // The result tensor.
    std::string targetEntryToString() const;
};

class TensorView {
protected:
    // The interface iterators.
    std::vector<const Iterator *> interface;
    // Define the corresponding shape view for interface.
    using IteratorShapeView = AbstractShape<const std::vector<const Iterator *>&, [](const Iterator * const& ptr) -> const Size& { return ptr->size(); }>;
    // The reduce iterators.
    std::vector<const Reduce *> manipulations;
    using ReduceIteratorShapeView = AbstractShape<const std::vector<const Reduce *>&, [](const Reduce * const& ptr) -> const Size& { return ptr->size(); }>;

    std::vector<PureTensor> tensors;
    AbstractAccess forwardAccess; // Iterators evaluated for the forward pipeline.
    std::vector<AbstractAccess> backwardAccesses; // Iterators evaluated for the backward pipeline.

    IR subgraphs;

public:
    // Build the tensor from iterator DAG.
    explicit TensorView(const std::vector<Topmost>& tensors, TensorExpression blending);

    const IR& getSubgraphs() const { return subgraphs; }

    IteratorShapeView getInterfaceShape() const { return IteratorShapeView(interface); }

    // Returns the underlying tensor underneath the view.
    const std::vector<PureTensor>& getUnderlyingTensors() const { return tensors; }

    Graph buildGraph() const {
        Graph::Builder builder;
        builder.addTopmosts(tensors | std::views::transform(&PureTensor::getContent));
        return builder.build();
    }

    bool operator==(const TensorView& rhs) const {
        return std::ranges::equal(tensors, rhs.tensors);
    }
    std::size_t hash() const {
        return std::hash<std::vector<Topmost>>{}(tensors | std::views::transform(&PureTensor::getContent));
    }

    const AbstractAccess& getForwardAccess() const { return forwardAccess; }

    const std::vector<AbstractAccess>& getBackwardAccesses() const { return backwardAccesses; }

    // Returns the interface of the view.
    const std::vector<const Iterator *>& getInterfaceIterators() const { return interface; }

    // Returns the reduce manipulations.
    const std::vector<const Reduce *>& getManipulations() const { return manipulations; }

    // Note that sometimes we need padding to make things work. Here we need to guarantee that all dimensions in the Graph are valid, so instead of padding tensors, we pad variables.
    ConcreteConsts computePadding(const BindingContext& ctx, const ConcreteConsts& consts) const;
    PaddedConsts computeConsts(const BindingContext& ctx, const std::map<std::string, std::size_t>& mappings) const {
        auto unpaddedConsts = ctx.realizeConsts(mappings);
        auto paddedConsts = computePadding(ctx, unpaddedConsts);
        return { std::move(unpaddedConsts), std::move(paddedConsts) };
    }

    // Observe that FLOPs is determined by outer loops and inner loops.
    std::size_t getFLOPs(const ConcreteConsts& consts) const;

    // Evaluate the full loops.
    std::string printNestedLoops(const BindingContext& ctx, int pos) const;
    std::string printNestedLoopsForAll(const BindingContext& ctx) const;

    std::string description(const BindingContext& ctx) const;
};

} // namespace kas
