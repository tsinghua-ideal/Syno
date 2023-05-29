#pragma once

#include <bit>
#include <concepts>
#include <optional>
#include <set>
#include <type_traits>
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
    typename V::BranchType;
    typename V::OpType::Values;
    { V::OpType::BranchCount } -> std::convertible_to<std::size_t>;
    requires std::same_as<typename V::OpType::Values, Valuations<V::OpType::BranchCount>>;
    { v.op } -> std::same_as<const typename V::OpType&>;
    { v[b] } -> std::convertible_to<Dimension>;
    { v.outgoingDirection(b) } -> std::convertible_to<Direction>;
    { v.visitAdjacent(b) } -> std::convertible_to<VisitedVertex>;
};

class RepeatLikeVertex {
    const Graph& graph;

public:
    using OpType = RepeatLikeOp;
    using BranchType = OpType::Branch;
    const OpType& op;

    RepeatLikeVertex(const Graph& graph, const OpType& op):
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
    using BranchType = OpType::Branch;
    const OpType& op;

    SplitLikeVertex(const Graph& graph, const OpType& op):
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
    using BranchType = OpType::Branch;
    const OpType& op;

    MergeLikeVertex(const Graph& graph, const OpType& op):
        graph { graph }, op { op } {}
    Dimension operator[](OpType::Branch branch) const;
    Direction outgoingDirection(OpType::Branch branch) const;
    VisitedVertex visitAdjacent(OpType::Branch branch) const;
};
template<typename CaseMergeLike>
concept MergeLikeCase = std::invocable<CaseMergeLike, MergeLikeVertex, MergeLikeOp::Branch>;
template<MergeLikeCase CaseMergeLike>
using MergeLikeCaseResult = std::invoke_result_t<CaseMergeLike, MergeLikeVertex, MergeLikeOp::Branch>;

template<auto Val>
struct EmptyVertexCase {
    template<typename V, typename B>
    auto operator()(V&&, B&&) const {
        return Val;
    }
};

class VisitedVertex {
    friend class Graph;

    using Inner = std::variant<
        std::pair<RepeatLikeVertex, RepeatLikeOp::Branch>,
        std::pair<SplitLikeVertex, SplitLikeOp::Branch>,
        std::pair<MergeLikeVertex, MergeLikeOp::Branch>
    >;
    std::optional<Inner> vertexAndSource;

    VisitedVertex(auto&& vertexAndSource):
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
        return std::visit(Visitor<CaseRepeatLike, CaseSplitLike, CaseMergeLike> {
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

    // Use bits to record the Dimensions of which this Dimension is a descendant.
    class CompactIndices {
        std::size_t content;
        CompactIndices(std::size_t raw): content { raw } {}
    public:
        [[nodiscard]] static CompactIndices None() { return {0}; }
        [[nodiscard]] static CompactIndices Single(std::size_t index) {
            return { static_cast<std::size_t>(1) << index };
        }
        [[nodiscard]] static CompactIndices All(std::size_t count) {
            return { (static_cast<std::size_t>(1) << count) - 1 };
        }
        [[nodiscard]] CompactIndices merged(CompactIndices other) const {
            return { content | other.content };
        }
        CompactIndices& merges(CompactIndices other) {
            content |= other.content;
            return *this;
        }
        [[nodiscard]] bool contains(std::size_t index) const {
            return Single(index).content & content;
        }
        [[nodiscard]] CompactIndices excluded(CompactIndices other) const {
            return { content & ~other.content };
        }
        CompactIndices& excludes(CompactIndices other) {
            content &= ~other.content;
            return *this;
        }
        template<typename F>
        requires std::invocable<F, std::size_t>
        void foreach(F&& f) const {
            std::size_t tries = std::numeric_limits<std::size_t>::digits - std::countl_zero(content);
            for (std::size_t i = 0; i < tries; ++i) {
                if (content & (static_cast<std::size_t>(1) << i)) {
                    std::invoke(std::forward<F>(f), i);
                }
            }
        }
    };

    struct DimensionMetadata {
        OpAbove opAbove; // Op above each dimension.
        CompactIndices ancestors; // Dimensions of which this Dimension is a descendant.
    };

    // Use Builder to construct a Graph.
    class Builder final: public DimVisitor {
        std::vector<Dimension> topmost;
        std::map<Dimension, DimensionMetadata, Dimension::AddressLessThan> dimMeta;
        std::set<const Iterator *> outputIterators;
        std::set<const MapReduceOp *> mapReduceIterators;

        OpAbove parent;
        CompactIndices ancestor = CompactIndices::None();
        void visit(const Iterator& dim) override;
        void visit(const MapReduceOp& dim) override;
        void visit(const RepeatLikeOp::Input& dim) override;
        void visit(const SplitLikeOp::Input& dim) override;
        void visit(const MergeLikeOp::Input& dim) override;
        void visit(const Dimension& dim);

    public:
        void addTopmost(const Dimension& dim);
        template<DimensionRange R>
        void addTopmost(R&& dims) {
            for (const Dimension& dim: dims) {
                addTopmost(dim);
            }
        }
        Graph build();
    };
private:
    // The input dimensions;
    std::vector<Dimension> topmost;
    // The Op's above the Dimensions.
    std::map<Dimension, DimensionMetadata, Dimension::AddressLessThan> dimMeta;
    // And the output/reduce iterators as well.
    std::vector<const Iterator *> outputIterators;
    std::vector<const MapReduceOp *> mapReduceIterators;

    Graph(auto&& topmost, auto&& dimMeta, auto&& outputIterators, auto&& mapReduceIterators):
        topmost { std::forward<decltype(topmost)>(topmost) },
        dimMeta { std::forward<decltype(dimMeta)>(dimMeta) },
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
        WalkDownVisitor(const Graph& graph): graph { graph } {}
    };

    // Visitor that walks up along a dimension.
    struct WalkUpVisitor {
        const Graph& graph;
        std::optional<VisitedVertex::Inner> result;

        void operator()(std::monostate) {}
        void operator()(const RepeatLikeOp *op);
        void operator()(std::pair<const SplitLikeOp *, Order> opAndOrder);
        void operator()(const MergeLikeOp *op);
        WalkUpVisitor(const Graph& graph): graph { graph } {}
    };

public:
    // Walk along a dimension in a direction to find a vertex.
    VisitedVertex visitAlong(const Dimension& dim, Direction dir) const;

    const std::vector<Dimension>& getTopmost() const { return topmost; }
    decltype(auto) getDimensions() const {
        return dimMeta | std::views::transform([](auto&& pair) { return pair.first; });
    }
    std::vector<const Iterator *>& getOutputIterators() { return outputIterators; }
    const std::vector<const Iterator *>& getOutputIterators() const { return outputIterators; }
    std::vector<const MapReduceOp *>& getMapReduceIterators() { return mapReduceIterators; }
    const std::vector<const MapReduceOp *>& getMapReduceIterators() const { return mapReduceIterators; }

    struct ConnectedComponent {
        std::vector<Dimension> inputs;
        std::vector<Dimension> outputs; // Iterator's and MapReduceOp's
    };
    std::vector<ConnectedComponent> computeConnectedComponents() const;

    template<typename Value>
    class AttributeMap {
        std::map<const RepeatLikeVertex::OpType *, Value> rAttr;
        std::map<const SplitLikeVertex::OpType *, Value> sAttr;
        std::map<const MergeLikeVertex::OpType *, Value> mAttr;

    public:
        template<typename V>
        requires Vertex<std::remove_cvref_t<V>>
        Value& operator[](V&& v) {
            if constexpr (std::is_same_v<std::remove_cvref_t<V>, RepeatLikeVertex>) {
                return rAttr[&v.op];
            } else if constexpr (std::is_same_v<std::remove_cvref_t<V>, SplitLikeVertex>) {
                return sAttr[&v.op];
            } else if constexpr (std::is_same_v<std::remove_cvref_t<V>, MergeLikeVertex>) {
                return mAttr[&v.op];
            } else {
                static_assert(std::is_same_v<std::remove_cvref_t<V>, RepeatLikeVertex> ||
                              std::is_same_v<std::remove_cvref_t<V>, SplitLikeVertex> ||
                              std::is_same_v<std::remove_cvref_t<V>, MergeLikeVertex>);
            }
        }
    };
};

} // namespace kas
