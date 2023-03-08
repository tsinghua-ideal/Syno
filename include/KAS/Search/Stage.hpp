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
#include "KAS/Utils/Hash.hpp"


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
        std::size_t operator()(const ColoredInterface *interface) const noexcept {
            std::size_t h = interface->items.size();
            for (const auto& dim: interface->items) {
                HashCombine(h, dim.dimension.hash());
            }
            return h;
        }
    };
    struct Equal {
        bool operator()(const ColoredInterface *lhs, const ColoredInterface *rhs) const noexcept {
            return std::ranges::equal(lhs->items, rhs->items, std::equal_to<Dimension>{}, ColoredDimension::Projection{}, ColoredDimension::Projection{});
        }
    };
    std::unordered_set<ColoredInterface *, Hash, Equal> interfaces;

    // This is just the `container_of()` in Linux kernel.
    static Stage *Convert(ColoredInterface *from);
public:
    inline DimensionStore& dimStore() { return dimensionStore; }
    Stage *find(ColoredInterface *interface) const;
    bool insert(Stage *stage);
    ~StageStore();
};

class Stage {
    friend class StageStore;
    FRIEND_TEST(search_tests, sampler);

    // The interface decides the hash. Other properties are computed.
    ColoredInterface interface;

    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    std::optional<NextBound> nexts; // If `nexts == std::nullopt`, then all children are not evaluated. If `nexts` is evaluated, all children are evaluated, but the `Stage *` may be `nullptr`, i.e., remains to be evaluated.
    std::vector<std::pair<FinalizeOp, std::unique_ptr<TensorView>>> finalizes;
    std::vector<std::pair<const RepeatLikeOp * const, Stage *>> nextRepeatLikes;
    std::vector<std::pair<const SplitLikeOp * const, Stage *>> nextSplitLikes;
    std::vector<std::pair<const MergeLikeOp * const, Stage *>> nextMergeLikes;

    // Metadata.
    Sampler& sampler;
    Colors colors;
    std::vector<std::reference_wrapper<const Size>> missingSizes;
    std::size_t depth; // Stages with identical interfaces must be the same depth.

    StageStore& getStageStore();

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guard();

    TensorView *getFinalize(std::size_t index);

    static constexpr Colors::Options colorsOptions = {
        .maximumTensors = 2,
    };

    template<PrimitiveOp NextOp>
    Stage *getNext(const NextOp *op) {
        StageStore& store = getStageStore();
        auto newInterface = interface;
        auto newColors = colors;
        if (!op->transformInterface(newInterface, colors, colorsOptions)) {
            return nullptr; // This failed.
        }
        if (!newColors.isConsistent()) {
            return nullptr; // Inconsistent colors. This failed.
        }
        if (Stage *found = store.find(&newInterface); found) {
            return found;
        } else {
            auto tempStage = std::make_unique<Stage>(std::move(newInterface), std::move(newColors), sampler, depth + 1);
            if(store.insert(tempStage.get())) {
                return tempStage.release();
            } else {
                KAS_CRITICAL("StageStore::insert() failed.");
            }
        }
    }

    // Remove the Op from store, deleting its enclosing dimensions as well.
    template<PrimitiveOp Op>
    void removeOp(const Op *op) {
        // TODO
    }

public:
    Stage(auto&& interface, auto&& colors, Sampler& sampler, std::size_t depth):
        interface { std::forward<decltype(interface)>(interface) },
        sampler { sampler },
        colors { std::forward<decltype(colors)>(colors) },
        depth { depth }
    {
        // Compute colors and missing sizes. TODO.
    }
    inline const ColoredInterface& getInterface() const { return interface; }
    std::size_t countChildren();
    bool isFinal(std::size_t index);
    std::variant<Stage *, TensorView *> next(std::size_t index);
    std::string opType(std::size_t index);
    std::string opDescription(std::size_t index);
};

} // namespace kas
