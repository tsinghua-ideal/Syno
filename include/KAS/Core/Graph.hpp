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
concept Vertex = requires(std::remove_cvref_t<V> v, std::remove_cvref_t<V>::OpType::Branch b) {
    typename std::remove_cvref_t<V>::OpType;
    typename std::remove_cvref_t<V>::OpType::Branch;
    typename std::remove_cvref_t<V>::BranchType;
    typename std::remove_cvref_t<V>::OpType::Values;
    { std::remove_cvref_t<V>::OpType::BranchCount } -> std::convertible_to<std::size_t>;
    requires std::same_as<typename std::remove_cvref_t<V>::OpType::Values, Valuations<std::remove_cvref_t<V>::OpType::BranchCount>>;
    { v.op } -> std::same_as<const typename std::remove_cvref_t<V>::OpType&>;
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
    Dimension operator[](BranchType branch) const;
    Direction outgoingDirection(BranchType branch) const;
    VisitedVertex visitAdjacent(BranchType branch) const;
};
template<typename CaseRepeatLike>
concept RepeatLikeCase = std::invocable<CaseRepeatLike, const RepeatLikeVertex&, RepeatLikeOp::Branch>;
template<RepeatLikeCase CaseRepeatLike>
using RepeatLikeCaseResult = std::invoke_result_t<CaseRepeatLike, const RepeatLikeVertex&, RepeatLikeOp::Branch>;

class SplitLikeVertex {
    const Graph& graph;

public:
    using OpType = SplitLikeOp;
    using BranchType = OpType::Branch;
    const OpType& op;

    SplitLikeVertex(const Graph& graph, const OpType& op):
        graph { graph }, op { op } {}
    Dimension operator[](BranchType branch) const;
    Direction outgoingDirection(BranchType branch) const;
    VisitedVertex visitAdjacent(BranchType branch) const;
};
template<typename CaseSplitLike>
concept SplitLikeCase = std::invocable<CaseSplitLike, const SplitLikeVertex&, SplitLikeOp::Branch>;
template<SplitLikeCase CaseSplitLike>
using SplitLikeCaseResult = std::invoke_result_t<CaseSplitLike, const SplitLikeVertex&, SplitLikeOp::Branch>;

class MergeLikeVertex {
    const Graph& graph;

public:
    using OpType = MergeLikeOp;
    using BranchType = OpType::Branch;
    const OpType& op;

    MergeLikeVertex(const Graph& graph, const OpType& op):
        graph { graph }, op { op } {}
    Dimension operator[](BranchType branch) const;
    Direction outgoingDirection(BranchType branch) const;
    VisitedVertex visitAdjacent(BranchType branch) const;
};
template<typename CaseMergeLike>
concept MergeLikeCase = std::invocable<CaseMergeLike, const MergeLikeVertex&, MergeLikeOp::Branch>;
template<MergeLikeCase CaseMergeLike>
using MergeLikeCaseResult = std::invoke_result_t<CaseMergeLike, const MergeLikeVertex&, MergeLikeOp::Branch>;

class ExpandVertex {
public:
    using OpType = ExpandOp;
    using BranchType = std::monostate;
    const OpType& op;

    ExpandVertex(const OpType& op): op { op } {}
};
template<typename CaseExpand>
concept ExpandCase = std::invocable<CaseExpand, const ExpandVertex&, std::monostate>;
template<ExpandCase CaseExpand>
using ExpandCaseResult = std::invoke_result_t<CaseExpand, const ExpandVertex&, std::monostate>;

template<typename V>
concept GeneralizedVertex = Vertex<V> || std::same_as<std::remove_cvref_t<V>, ExpandVertex>;

class VisitedVertex {
    friend class Graph;

    using Inner = std::variant<
        std::pair<RepeatLikeVertex, RepeatLikeVertex::BranchType>,
        std::pair<SplitLikeVertex, SplitLikeVertex::BranchType>,
        std::pair<MergeLikeVertex, MergeLikeVertex::BranchType>,
        ExpandVertex
    >;
    std::optional<Inner> vertexAndSource;

    VisitedVertex(auto&& vertexAndSource):
        vertexAndSource { std::forward<decltype(vertexAndSource)>(vertexAndSource) }
    {}

    template<RepeatLikeCase CaseRepeatLike, SplitLikeCase CaseSplitLike, MergeLikeCase CaseMergeLike, ExpandCase CaseExpand>
    struct Visitor {
        CaseRepeatLike&& caseRepeatLike;
        CaseSplitLike&& caseSplitLike;
        CaseMergeLike&& caseMergeLike;
        CaseExpand&& caseExpand;
        decltype(auto) operator()(std::pair<RepeatLikeVertex, RepeatLikeVertex::BranchType>& r) const {
            return std::invoke(std::forward<CaseRepeatLike>(caseRepeatLike), r.first, r.second);
        }
        decltype(auto) operator()(std::pair<SplitLikeVertex, SplitLikeVertex::BranchType>& s) const {
            return std::invoke(std::forward<CaseSplitLike>(caseSplitLike), s.first, s.second);
        }
        decltype(auto) operator()(std::pair<MergeLikeVertex, MergeLikeVertex::BranchType>& m) const {
            return std::invoke(std::forward<CaseMergeLike>(caseMergeLike), m.first, m.second);
        }
        decltype(auto) operator()(ExpandVertex& e) const {
            return std::invoke(std::forward<CaseExpand>(caseExpand), e, std::monostate{});
        }
    };

public:
    template<RepeatLikeCase CaseRepeatLike, SplitLikeCase CaseSplitLike, MergeLikeCase CaseMergeLike, ExpandCase CaseExpand, typename Result = RepeatLikeCaseResult<CaseRepeatLike>>
    requires
        std::same_as<Result, RepeatLikeCaseResult<CaseRepeatLike>> &&
        std::same_as<Result, SplitLikeCaseResult<CaseSplitLike>> &&
        std::same_as<Result, MergeLikeCaseResult<CaseMergeLike>> &&
        std::same_as<Result, ExpandCaseResult<CaseExpand>>
    Result match(
        CaseRepeatLike&& caseRepeatLike,
        CaseSplitLike&& caseSplitLike,
        CaseMergeLike&& caseMergeLike,
        CaseExpand&& caseExpand
    ) {
        if (!vertexAndSource.has_value()) {
            if constexpr (std::is_void_v<Result>) {
                return;
            } else {
                return {};
            }
        }
        return std::visit(Visitor<CaseRepeatLike, CaseSplitLike, CaseMergeLike, CaseExpand> {
            std::forward<CaseRepeatLike>(caseRepeatLike),
            std::forward<CaseSplitLike>(caseSplitLike),
            std::forward<CaseMergeLike>(caseMergeLike),
            std::forward<CaseExpand>(caseExpand)
        }, *vertexAndSource);
    }
    template<typename F>
    decltype(auto) match(F&& f) {
        return match(std::forward<F>(f), std::forward<F>(f), std::forward<F>(f), std::forward<F>(f));
    }
};

static_assert(Vertex<RepeatLikeVertex>);
static_assert(Vertex<SplitLikeVertex>);
static_assert(Vertex<MergeLikeVertex>);
static_assert(GeneralizedVertex<ExpandVertex>);

class Graph {
public:
    using DimensionSet = std::set<Dimension, Dimension::AddressLessThan>;
    using CutSet = DimensionSet;

    // In the original graph, each Dimension serves as an edge. It provides easy access for the Op below it (which has the Dimension as input), but cannot access the Op above it. This is used to store the Op above each Dimension.
    struct OpAbove {
        // std::monostate means the dimension is an input dimension, and has no Op above.
        std::variant<std::monostate, const RepeatLikeOp *, std::pair<const SplitLikeOp *, Order>, const MergeLikeOp *, const ExpandOp *> op;
        template<typename F>
        decltype(auto) visit(F&& f) const {
            return std::visit(std::forward<F>(f), op);
        }
    };

    // Use bits to record the Dimension's of which this Dimension is a descendant.
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
        CompactIndices ancestors; // Dimension's of which this Dimension is a descendant.
    };

    // Use Builder to construct a Graph.
    class Builder final: public DimVisitor {
        Topmost topmost;
        std::map<Dimension, DimensionMetadata, Dimension::AddressLessThan> dimMeta;
        std::set<const Iterator *> outputIterators;
        std::set<const Reduce *> reduceIterators;
        std::set<const PrimitiveOp *> ops;

        OpAbove parent;
        CompactIndices ancestor = CompactIndices::None();
        void visit(const Iterator& dim) override;
        void visit(const Reduce& dim) override;
        void visit(const RepeatLikeOp::Input& dim) override;
        void visit(const SplitLikeOp::Input& dim) override;
        void visit(const MergeLikeOp::Input& dim) override;
        void match(const Dimension& dim);

    public:
        Builder& addDimension(const Dimension& dim);
        Builder& addExpansion(const Expand *exp);
        Builder& addTopmost(const Topmost& interface);
        static Graph BuildFromHandle(const GraphHandle& handle);
        template<std::ranges::input_range R>
        requires std::same_as<std::ranges::range_value_t<R>, Dimension>
        Builder& addDimensions(R&& dims) {
            for (const Dimension& dim: dims) {
                addDimension(dim);
            }
            return *this;
        }
        template<std::ranges::input_range R>
        requires std::same_as<std::ranges::range_value_t<R>, const Expand *>
        Builder& addExpansions(R&& exps) {
            for (const Expand *exp: exps) {
                addExpansion(exp);
            }
            return *this;
        }
        template<std::ranges::input_range R>
        requires std::same_as<std::ranges::range_value_t<R>, Topmost>
        Builder& addTopmosts(R&& interfaces) {
            for (const Topmost& interface: interfaces) {
                addTopmost(interface);
            }
            return *this;
        }
        Graph build();
    };
private:
    // The input dimensions;
    Topmost topmost;
    // The Op's above the Dimension's.
    std::map<Dimension, DimensionMetadata, Dimension::AddressLessThan> dimMeta;
    // And the output/reduce iterators as well.
    std::vector<const Iterator *> outputIterators;
    std::vector<const Reduce *> reduceIterators;
    std::set<const PrimitiveOp *> ops;

    Graph(auto&& topmost, auto&& dimMeta, auto&& outputIterators, auto&& reduceIterators, auto&& ops):
        topmost(std::forward<decltype(topmost)>(topmost)),
        dimMeta(std::forward<decltype(dimMeta)>(dimMeta)),
        outputIterators(std::forward<decltype(outputIterators)>(outputIterators)),
        reduceIterators(std::forward<decltype(reduceIterators)>(reduceIterators)),
        ops(std::forward<decltype(ops)>(ops))
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
        void operator()(const ExpandOp *op);
        WalkUpVisitor(const Graph& graph): graph { graph } {}
    };

public:
    // Walk along a dimension in a direction to find a vertex.
    VisitedVertex visitAlong(const Dimension& dim, Direction dir) const;

    template<typename Visitor, typename AttributeType>
    void accept(BottomTopDimVisitor<Visitor, AttributeType>& visitor) const { visitor.propagate(topmost.getAllDimensions()); }

    const Topmost& getTopmost() const { return topmost; }
    decltype(auto) getDimensions() const {
        return dimMeta | std::views::transform([](auto&& pair) -> const Dimension& { return pair.first; });
    }
    std::vector<const Iterator *>& getOutputIterators() { return outputIterators; }
    const std::vector<const Iterator *>& getOutputIterators() const { return outputIterators; }
    std::vector<const Reduce *>& getReduceIterators() { return reduceIterators; }
    const std::vector<const Reduce *>& getReduceIterators() const { return reduceIterators; }

    // Includes ExpandOp.
    const PrimitiveOp *getOpAbove(const Dimension& dim) const;
    // Every Op, including ExpandOp, excluding ReduceOp.
    const std::set<const PrimitiveOp *>& getOps() const { return ops; }

    template<typename Value>
    class AttributeMap {
        std::map<const RepeatLikeVertex::OpType *, Value> rAttr;
        std::map<const SplitLikeVertex::OpType *, Value> sAttr;
        std::map<const MergeLikeVertex::OpType *, Value> mAttr;

    public:
        template<Vertex V>
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
        template<Vertex V>
        Value *find(V&& v) {
            if constexpr (std::is_same_v<std::remove_cvref_t<V>, RepeatLikeVertex>) {
                auto it = rAttr.find(&v.op);
                return it == rAttr.end() ? nullptr : &it->second;
            } else if constexpr (std::is_same_v<std::remove_cvref_t<V>, SplitLikeVertex>) {
                auto it = sAttr.find(&v.op);
                return it == sAttr.end() ? nullptr : &it->second;
            } else if constexpr (std::is_same_v<std::remove_cvref_t<V>, MergeLikeVertex>) {
                auto it = mAttr.find(&v.op);
                return it == mAttr.end() ? nullptr : &it->second;
            } else {
                static_assert(std::is_same_v<std::remove_cvref_t<V>, RepeatLikeVertex> ||
                              std::is_same_v<std::remove_cvref_t<V>, SplitLikeVertex> ||
                              std::is_same_v<std::remove_cvref_t<V>, MergeLikeVertex>);
            }
        }
    };
};

class ConstrainedGraph {
    const Graph& graph;
    using DimensionSet = Graph::DimensionSet;
    using CutSet = Graph::CutSet;
    DimensionSet dimensions;
    std::set<const PrimitiveOp *> ops;
    std::optional<CutSet> top;
    std::optional<CutSet> bottom;
    ConstrainedGraph(const Graph& graph, DimensionSet&& dimensions, std::set<const PrimitiveOp *>&& ops, std::optional<CutSet>&& top, std::optional<CutSet>&& bottom):
        graph { graph }, dimensions(std::move(dimensions)), ops(std::move(ops)), top(std::move(top)), bottom(std::move(bottom)) {}

public:
    class Builder {
        const Graph& graph;
        std::optional<CutSet> top;
        std::optional<CutSet> bottom;
        template<DimensionRange R>
        Builder& set(std::optional<CutSet> Builder::*field, R&& r) {
            auto& cutSet = this->*field;
            KAS_ASSERT(!cutSet);
            using std::begin;
            using std::end;
            cutSet = CutSet();
            cutSet->insert(begin(std::forward<R>(r)), end(std::forward<R>(r)));
            return *this;
        }
    public:
        Builder(const Graph& graph): graph { graph } {}
        template<DimensionRange Top>
        Builder& setTop(Top&& top) {
            return set(&Builder::top, std::forward<Top>(top));
        }
        template<DimensionRange Bottom>
        Builder& setBottom(Bottom&& bottom) {
            return set(&Builder::bottom, std::forward<Bottom>(bottom));
        }
        ConstrainedGraph build();
    };
    struct VisitedSubgraphVertex {
        std::variant<VisitedVertex, std::pair<Direction, Dimension>> result;
        template<typename F>
        decltype(auto) match(F&& f) {
            return std::visit([&](auto&& x) -> decltype(auto) {
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(x)>, VisitedVertex>) {
                    return x.match(std::forward<F>(f));
                } else {
                    return f(x.first, x.second);
                }
            }, result);
        }
    };
    VisitedSubgraphVertex visitAlong(const Dimension& dim, Direction dir) const;
    const DimensionSet& getDimensions() const { return dimensions; }
    const std::set<const PrimitiveOp *>& getOps() const { return ops; }
};

} // namespace kas
