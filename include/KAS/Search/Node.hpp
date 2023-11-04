#pragma once

#include <cstddef>
#include <map>
#include <ranges>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <variant>

#include <fmt/core.h>

#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/Search/Common.hpp"
#include "KAS/Transforms/Transforms.hpp"
#include "KAS/Utils/Hash.hpp"
#include "KAS/Utils/Ranges.hpp"
#include "KAS/Utils/Threads.hpp"

namespace kas {

class ReductionStage;
class NormalStage;
class FinalStage;
struct LatticeTask;
struct ShapeDistance;

class Node {
    friend struct Next;

    Sampler *sampler;

    // A node has 3 types.
    enum class Type: std::uint8_t {
        Reducing = 0, // Generating reductions.
        Growing = 1, // Now we have generated Reduce's, repeatedly add PrimitiveOp's.
        Final = 2, // Finalization performed.
    };
    // This corresponds to the three types.
    std::variant<ReductionStage *, NormalStage *, FinalStage *> inner;
    Type type() const noexcept {
        return static_cast<Type>(inner.index());
    }
    template<typename R, typename FR, typename FN, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FR, ReductionStage *>, R> &&
        std::convertible_to<std::invoke_result_t<FN, NormalStage *>, R> &&
        std::convertible_to<std::invoke_result_t<FF, FinalStage *>, R>
    R match(FR&& fr, FN&& fn, FF&& ff) const {
        return std::visit([&](auto arg) -> R {
            if constexpr (std::is_same_v<decltype(arg), ReductionStage *>) {
                return fr(arg);
            } else if constexpr (std::is_same_v<decltype(arg), NormalStage *>) {
                return fn(arg);
            } else if constexpr (std::is_same_v<decltype(arg), FinalStage *>) {
                return ff(arg);
            } else {
                KAS_UNREACHABLE();
            }
        }, inner);
    }
    template<typename R, typename FS, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FS, AbstractStage *>, R> &&
        std::convertible_to<std::invoke_result_t<FF, FinalStage *>, R>
    R match(FS&& fs, FF&& ff) const {
        return std::visit([&](auto arg) -> R {
            if constexpr (std::is_same_v<decltype(arg), ReductionStage *>) {
                return fs(arg);
            } else if constexpr (std::is_same_v<decltype(arg), NormalStage *>) {
                return fs(arg);
            } else if constexpr (std::is_same_v<decltype(arg), FinalStage *>) {
                return ff(arg);
            } else {
                KAS_UNREACHABLE();
            }
        }, inner);
    }

public:
    Node(Sampler *sampler, ReductionStage *rStage):
        sampler { sampler }, inner { rStage } {}
    Node(Sampler *sampler, NormalStage *nStage):
        sampler { sampler }, inner { nStage } {}
    Node(Sampler *sampler, FinalStage *fStage):
        sampler { sampler }, inner { fStage } {}

    // For Python.
    bool operator==(const Node& rhs) const;
    // For Python.
    std::size_t hash() const;
    struct Hash {
        std::size_t operator()(const Node& node) const {
            return node.hash();
        }
    };

    // For convenience.
    std::strong_ordering operator<=>(const Node& rhs) const = default;

    AbstractStage *asNonFinalStage() const;
    FinalStage *asFinalStage() const;
    std::unique_ptr<Kernel> realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, CodeGenOptions options, const std::filesystem::path& directory, const std::string& name) const;
    // Obtain the mappings from Sampler, and do not solve the paddings. We only want to estimate the FLOPs.
    std::size_t estimateTotalFLOPsAsFinal() const;
    // No tensors!
    void generateGraphviz(const std::filesystem::path& path, const std::string& name) const;
    // With tensors!
    void generateGraphvizAsFinal(const std::filesystem::path& path, const std::string& name) const;
    std::string getNestedLoopsAsFinal() const;

    Node arbitraryParent() const;
    void recomputeShapeDistance() const;
    ShapeDistance getShapeDistance() const;
    // The count of children nodes.
    std::size_t countChildren() const;
    std::vector<Next> getChildrenHandles() const;
    std::vector<Arc> getChildrenArcs() const;
    std::optional<Arc> getArcFromHandle(Next next) const;
    std::optional<Node> getChild(Next next) const;
    std::vector<std::optional<Node>> getChildren(const std::vector<Next>& nexts) const;
    bool canAcceptArc(Arc arc) const;
    std::optional<Node> getChildFromArc(Arc arc) const;
    std::vector<std::optional<Node>> getChildrenFromArcs(const std::vector<Arc>& arcs) const;
    std::vector<Next> getPossiblePath() const;
    std::vector<Arc> getComposingArcs() const;
    void expandSync(int layers) const;
    void expandWithArcs(ThreadPool<LatticeTask>& expander, const LatticeTask& task) const;
    Node expandToSync(Node target) const;
    void expand(int layers) const;
    std::optional<std::string> getChildDescription(Next next) const;
    bool isReduction() const { return type() == Type::Reducing; }
    bool isNormal() const { return type() == Type::Growing; }
    bool isFinal() const { return type() == Type::Final; }
    bool isDeadEnd() const;
    bool discoveredFinalDescendant() const;
    std::string toString() const;

    // For debugging.
    std::string debugToGraphviz() const;
};

class LatticePool {
    std::size_t depth;
    std::vector<std::pair<std::mutex, std::unordered_set<Node, Node::Hash>>> nodesPools;
public:
    LatticePool(std::size_t depth);
    bool add(std::size_t remainingArcs, Node node);
};

struct LatticeTask {
    LatticePool& pool;
    Node node;
    std::vector<Arc> arcs;
};

} // namespace kas
