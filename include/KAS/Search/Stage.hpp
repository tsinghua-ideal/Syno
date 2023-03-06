#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Transforms/DimensionStore.hpp"


namespace kas {

class Sampler;

struct Next {
    enum class Type {
        Finalize,
        RepeatLike,
        SplitLike,
        MergeLike,
    };
    Type type;
    std::size_t index;
};

struct NextBound {
    std::size_t finalizeCount;
    std::size_t repeatLikeCount;
    std::size_t splitLikeCount;
    std::size_t mergeLikeCount;
    std::size_t size() const;
    Next get(std::size_t index) const;
};

class Stage;

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

    // This is just the `container_of()` in Linux kernel.
    static Stage *Convert(Interface *from);
public:
    inline DimensionStore& dimStore() { return dimensionStore; }
    Stage *find(Interface *interface) const;
    bool insert(Stage *stage);
    ~StageStore();
};

class Stage {
    friend class StageStore;
    FRIEND_TEST(search_tests, sampler);

    // The interface decides the hash. Other properties are computed.
    std::vector<Dimension> interface;

    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    std::optional<NextBound> nexts; // If `nexts == std::nullopt`, then all children are not evaluated. If `nexts` is evaluated, all children are evaluated, but the `Stage *` may be `nullptr`, i.e., remains to be evaluated.
    std::vector<std::pair<FinalizeOp, std::unique_ptr<TensorView>>> finalizes;
    std::vector<std::pair<const NextRepeatLike, Stage *>> nextRepeatLikes;
    std::vector<std::pair<const NextSplitLike, Stage *>> nextSplitLikes;
    std::vector<std::pair<const NextMergeLike, Stage *>> nextMergeLikes;

    // Metadata.
    Sampler& sampler;
    std::size_t depth; // Stages with identical interfaces must be the same depth.
    Colors colors;
    std::vector<std::reference_wrapper<const Size>> missingSizes;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guard();

    TensorView *getFinalize(std::size_t index);
    template<typename NextOp>
    requires (std::same_as<NextOp, NextRepeatLike> || std::same_as<NextOp, NextSplitLike> || std::same_as<NextOp, NextMergeLike>)
    auto& getNextBuffer() {
        if constexpr (std::same_as<NextOp, NextRepeatLike>) {
            return nextRepeatLikes;
        } else if constexpr (std::same_as<NextOp, NextSplitLike>) {
            return nextSplitLikes;
        } else if constexpr (std::same_as<NextOp, NextMergeLike>) {
            return nextMergeLikes;
        }
    }
    template<typename NextOp>
    Stage *getNext(StageStore& store, std::size_t index) {
        auto& [op, stage] = getNextBuffer<NextOp>()[index];
        if (!stage) {
            auto newInterface = op.applyTo(interface);
            if (Stage *found = store.find(&newInterface); found) {
                stage = found;
            } else {
                auto tempStage = std::unique_ptr<Stage> { new Stage { std::move(newInterface), sampler, depth + 1 } };
                if(store.insert(tempStage.get())) {
                    stage = tempStage.release();
                } else {
                    KAS_CRITICAL("StageStore::insert() failed.");
                }
            }
        }
        return stage;
    }

public:
    Stage(auto&& interface, Sampler& sampler, std::size_t depth):
        interface { std::forward<decltype(interface)>(interface) },
        sampler { sampler },
        depth { depth }
    {
        // Compute colors and missing sizes. TODO.
    }
    inline const std::vector<Dimension>& getInterface() const { return interface; }
    std::size_t countChildren();
    bool isFinal(std::size_t index);
    std::variant<Stage *, TensorView *> next(std::size_t index);
    std::string opType(std::size_t index);
    std::string opDescription(std::size_t index);
};

} // namespace kas
