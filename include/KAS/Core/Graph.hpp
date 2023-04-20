#pragma once

#include <optional>
#include <variant>

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class Graph;
class VisitedVertex;

// Here we want to provide a unified interface for visiting the Graph.
// To visit a vertex, we need to first obtain a VisitedVertex, via Graph::visitAlong or Vertex::visitAdjacent. The arguments of visitAlong is a Dimension and a Direction, and the arguments of visitAdjacent is a Branch.
// Then, pass 3 callback functions into VisitedVertex::match, which will be called depending on the type of the vertex. This is a bit like pattern matching.

template<typename V>
concept Vertex = requires(V v, V::OpType::Branch b) {
    typename V::OpType;
    typename V::OpType::Branch;
    typename V::OpType::Values;
    { V::OpType::BranchCount } -> std::convertible_to<std::size_t>;
    std::same_as<typename V::OpType::Values, Valuations<V::OpType::BranchCount>>;
    { v.op } -> std::same_as<const typename V::OpType&>;
    { v[b] } -> std::convertible_to<Dimension>;
    { v.outgoingDirection(b) } -> std::convertible_to<Direction>;
    { v.visitAdjacent(b) } -> std::convertible_to<VisitedVertex>;
};

class RepeatLikeVertex {
    const Graph& graph;

public:
    using OpType = RepeatLikeOp;
    const OpType& op;

    inline RepeatLikeVertex(const Graph& graph, const OpType& op):
        graph { graph }, op { op } {}
    Dimension operator[](OpType::Branch branch) const;
    Direction outgoingDirection(OpType::Branch branch) const;
    VisitedVertex visitAdjacent(OpType::Branch branch) const;
};
template<typename CaseRepeatLike>
concept RepeatLikeCase = std::invocable<CaseRepeatLike, RepeatLikeVertex, RepeatLikeOp::Branch>;
template<RepeatLikeCase CaseRepeatLike>
using RepeatLikeCaseResult = std::invoke_result_t<CaseRepeatLike, RepeatLikeVertex, RepeatLikeOp::Branch>;

class SplitLikeVertex {
    const Graph& graph;

public:
    using OpType = SplitLikeOp;
    const OpType& op;

    inline SplitLikeVertex(const Graph& graph, const OpType& op):
        graph { graph }, op { op } {}
    Dimension operator[](OpType::Branch branch) const;
    Direction outgoingDirection(OpType::Branch branch) const;
    VisitedVertex visitAdjacent(OpType::Branch branch) const;
};
template<typename CaseSplitLike>
concept SplitLikeCase = std::invocable<CaseSplitLike, SplitLikeVertex, SplitLikeOp::Branch>;
template<SplitLikeCase CaseSplitLike>
using SplitLikeCaseResult = std::invoke_result_t<CaseSplitLike, SplitLikeVertex, SplitLikeOp::Branch>;

class MergeLikeVertex {
    const Graph& graph;

public:
    using OpType = MergeLikeOp;
    const OpType& op;

    inline MergeLikeVertex(const Graph& graph, const OpType& op):
        graph { graph }, op { op } {}
    Dimension operator[](OpType::Branch branch) const;
    Direction outgoingDirection(OpType::Branch branch) const;
    VisitedVertex visitAdjacent(OpType::Branch branch) const;
};
template<typename CaseMergeLike>
concept MergeLikeCase = std::invocable<CaseMergeLike, MergeLikeVertex, MergeLikeOp::Branch>;
template<MergeLikeCase CaseMergeLike>
using MergeLikeCaseResult = std::invoke_result_t<CaseMergeLike, MergeLikeVertex, MergeLikeOp::Branch>;

class VisitedVertex {
    friend class Graph;

    using Inner = std::variant<
        std::pair<RepeatLikeVertex, RepeatLikeOp::Branch>,
        std::pair<SplitLikeVertex, SplitLikeOp::Branch>,
        std::pair<MergeLikeVertex, MergeLikeOp::Branch>
    >;
    std::optional<Inner> vertexAndSource;

    inline VisitedVertex(auto&& vertexAndSource):
        vertexAndSource { std::forward<decltype(vertexAndSource)>(vertexAndSource) }
    {}

    template<RepeatLikeCase CaseRepeatLike, SplitLikeCase CaseSplitLike, MergeLikeCase CaseMergeLike>
    struct Visitor {
        CaseRepeatLike&& caseRepeatLike;
        CaseSplitLike&& caseSplitLike;
        CaseMergeLike&& caseMergeLike;
        decltype(auto) operator()(std::pair<RepeatLikeVertex, RepeatLikeOp::Branch>& r) const {
            return std::invoke(std::forward<CaseRepeatLike>(caseRepeatLike), r.first, r.second);
        }
        decltype(auto) operator()(std::pair<SplitLikeVertex, SplitLikeOp::Branch>& s) const {
            return std::invoke(std::forward<CaseSplitLike>(caseSplitLike), s.first, s.second);
        }
        decltype(auto) operator()(std::pair<MergeLikeVertex, MergeLikeOp::Branch>& m) const {
            return std::invoke(std::forward<CaseMergeLike>(caseMergeLike), m.first, m.second);
        }
    };

public:
    template<RepeatLikeCase CaseRepeatLike, SplitLikeCase CaseSplitLike, MergeLikeCase CaseMergeLike, typename Result = RepeatLikeCaseResult<CaseRepeatLike>>
    requires
        std::same_as<Result, RepeatLikeCaseResult<CaseRepeatLike>> &&
        std::same_as<Result, SplitLikeCaseResult<CaseSplitLike>> &&
        std::same_as<Result, MergeLikeCaseResult<CaseMergeLike>>
    Result match(
        CaseRepeatLike&& caseRepeatLike,
        CaseSplitLike&& caseSplitLike,
        CaseMergeLike&& caseMergeLike
    ) {
        if (!vertexAndSource.has_value()) {
            if constexpr (std::is_void_v<Result>) {
                return;
            } else {
                return {};
            }
        }
        return std::visit(Visitor {
            std::forward<CaseRepeatLike>(caseRepeatLike),
            std::forward<CaseSplitLike>(caseSplitLike),
            std::forward<CaseMergeLike>(caseMergeLike)
        }, *vertexAndSource);
    }
};

static_assert(Vertex<RepeatLikeVertex>);
static_assert(Vertex<SplitLikeVertex>);
static_assert(Vertex<MergeLikeVertex>);

class Graph {
public:
    // In the original graph, each Dimension serves as an edge. It provides easy access for the Op below it (which has the Dimension as input), but cannot access the Op above it. This is used to store the Op above each Dimension.
    struct OpAbove {
        // std::monostate means the dimension is an input dimension, and has no Op above.
        std::variant<std::monostate, const RepeatLikeOp *, std::pair<const SplitLikeOp *, Order>, const MergeLikeOp *> op;
        template<typename F>
        void visit(F&& f) const {
            std::visit(std::forward<F>(f), op);
        }
    };

    // Use Builder to construct a Graph.
    class Builder final: public DimVisitor {
        std::map<Dimension, OpAbove, Dimension::AddressLessThan> theOtherEndOfEdge;
        std::vector<const Iterator *> outputIterators;
        std::vector<const MapReduceOp *> mapReduceIterators;

        OpAbove parent;
        void visit(const Iterator& dim) override;
        void visit(const MapReduceOp& dim) override;
        void visit(const RepeatLikeOp::Input& dim) override;
        void visit(const SplitLikeOp::Input& dim) override;
        void visit(const MergeLikeOp::Input& dim) override;
        void visit(const Dimension& dim);

    public:
        void add(const Dimension& dim);
        Graph build();
    };
private:
    // The Op's above the Dimensions.
    std::map<Dimension, OpAbove, Dimension::AddressLessThan> theOtherEndOfEdge;
    // And the output/reduce iterators as well.
    std::vector<const Iterator *> outputIterators;
    std::vector<const MapReduceOp *> mapReduceIterators;

    Graph(auto&& theOtherEndOfEdge, auto&& outputIterators, auto&& mapReduceIterators):
        theOtherEndOfEdge { std::forward<decltype(theOtherEndOfEdge)>(theOtherEndOfEdge) },
        outputIterators { std::forward<decltype(outputIterators)>(outputIterators) },
        mapReduceIterators { std::forward<decltype(mapReduceIterators)>(mapReduceIterators) }
    {}

    // Visitor that walks down along a dimension.
    struct WalkDownVisitor: public DimVisitor {
        const Graph& graph;
        std::optional<VisitedVertex::Inner> result;

        void visit(const RepeatLikeOp::Input& dim) override;
        void visit(const SplitLikeOp::Input& dim) override;
        void visit(const MergeLikeOp::Input& dim) override;
        using DimVisitor::visit;
        inline WalkDownVisitor(const Graph& graph): graph { graph } {}
    };

    // Visitor that walks up along a dimension.
    struct WalkUpVisitor {
        const Graph& graph;
        std::optional<VisitedVertex::Inner> result;

        inline void operator()(std::monostate) {}
        void operator()(const RepeatLikeOp *op);
        void operator()(std::pair<const SplitLikeOp *, Order> opAndOrder);
        void operator()(const MergeLikeOp *op);
        inline WalkUpVisitor(const Graph& graph): graph { graph } {}
    };

public:
    // Walk along a dimension in a direction to find a vertex.
    VisitedVertex visitAlong(const Dimension& dim, Direction dir) const;
};

} // namespace kas
