#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Colors.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Transforms/DimensionStore.hpp"


namespace kas {

struct Next {
    enum class Type {
        Finalize,
        RepeatLike,
        SplitLike,
        MergeLike,
    };
    Type type;
    std::size_t index;
    std::string toString() const;
};

struct NextBound {
    std::size_t finalizeCount;
    std::size_t repeatLikeCount;
    std::size_t splitLikeCount;
    std::size_t mergeLikeCount;
    std::size_t size() const;
    Next get(std::size_t index) const;
};

class StageStore;

class Stage {
    friend class StageStore;

    // The interface decides the hash. Other properties are computed.
    std::vector<Dimension> interface;

    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    std::optional<NextBound> nexts; // If `nexts == std::nullopt`, then all children are not evaluated. If `nexts` is evaluated, all children are evaluated, but the `Stage *` may be `nullptr`, i.e., remains to be evaluated.
    std::vector<std::pair<FinalizeOp, std::unique_ptr<TensorView>>> finalizes;
    std::vector<std::pair<const NextRepeatLike, Stage *>> nextRepeatLikes;
    std::vector<std::pair<const NextSplitLike, Stage *>> nextSplitLikes;
    std::vector<std::pair<const NextMergeLike, Stage *>> nextMergeLikes;

    // Metadata.
    Colors colors;
    std::vector<std::reference_wrapper<const Size>> missingSizes;

public:
    std::size_t size();
    bool isFinal(std::size_t index);
    std::variant<Stage *, TensorView *> next(std::size_t index);
    std::string opDescription(std::size_t index);
};

class StageStore {
    DimensionStore dimensionStore;
    struct Hash {
        std::size_t operator()(Interface * const interface) const noexcept {
            return std::hash<std::vector<Dimension>>{}(*interface);
        }
    };
    struct Equal {
        bool operator()(Interface * const lhs, Interface * const rhs) const noexcept {
            return *lhs == *rhs;
        }
    };
    std::unordered_set<Interface *, Hash, Equal> interfaces;
public:
    DimensionStore& dimStore() { return dimensionStore; }
    Stage *find(Interface * const interface) const {
        if (auto it = interfaces.find(interface); it != interfaces.end()) {
            return reinterpret_cast<Stage *>(*it - offsetof(Stage, interface));
        } else {
            return nullptr;
        }
    }
    bool insert(Stage * const stage) {
        return interfaces.insert(&stage->interface).second;
    }
};

} // namespace kas
