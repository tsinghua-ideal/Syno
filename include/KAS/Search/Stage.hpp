#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
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
#include "KAS/Search/Node.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

class Sampler;
class Stage;

class StageStore {
    DimensionStore dimensionStore;
    struct Hash {
        using is_transparent = void;
        std::size_t operator()(const ColoredInterface& interface) const noexcept;
        std::size_t operator()(const Stage * stage) const noexcept;
    };
    struct Equal {
        using is_transparent = void;
        bool operator()(const ColoredInterface& lhs, const ColoredInterface& rhs) const noexcept;
        bool operator()(const ColoredInterface& lhs, const Stage *rhs) const noexcept;
        bool operator()(const Stage *lhs, const ColoredInterface& rhs) const noexcept;
        bool operator()(const Stage *lhs, const Stage *rhs) const noexcept;
    };
    std::unordered_set<Stage *, Hash, Equal> interfaces;

public:
    inline DimensionStore& dimStore() { return dimensionStore; }
    Stage *find(const ColoredInterface& interface) const;
    bool insert(Stage *stage);
    ~StageStore();
};

class Stage {
    // The interface decides the hash. Other properties are computed.
    ColoredInterface interface;

    struct NextFinalizeSlot {
        std::size_t key;
        FinalizeOp finalization;
        std::unique_ptr<TensorView> kernel;
        inline Next toNext() const {
            return Next { Next::Type::Finalize, key };
        }
    };

    template<typename Op>
    struct NextOpSlot {
        std::size_t key;
        const Op *op;
        Stage *nextStage;
        Next toNext() const {
            return Next { Next::TypeOf<Op>(), key };
        }
    };
    template<typename Op>
    using NextOpStore = std::vector<NextOpSlot<Op>>;
    template<typename... Ops>
    struct NextOpStores {
        std::tuple<NextOpStore<Ops>...> stores;
        template<typename Op>
        NextOpStore<Op>& get() {
            return std::get<NextOpStore<Op>>(stores);
        }
        template<typename Op>
        const NextOpStore<Op>& get() const {
            return std::get<NextOpStore<Op>>(stores);
        }
    };

    // Lazily generate children.
    bool childrenGenerated = false;
    // Children handles.
    std::vector<Next> nexts;
    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    std::vector<NextFinalizeSlot> nextFinalizations;
    NextOpStores<ShiftOp, StrideOp, SplitOp, UnfoldOp, MergeOp, ShareOp> nextOpStores;

    // Metadata.
    Sampler& sampler;
    StageStore& getStageStore();
    Colors colors;
    // std::vector<std::reference_wrapper<const Size>> missingSizes;
    std::size_t depth; // Stages with identical interfaces must be of the same depth.

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guard();

    // Execute the finalization to obtain TensorView.
    TensorView *getFinalize(std::size_t key);

    static constexpr Colors::Options colorsOptions = {
        .maximumTensors = 2,
    };

    // Apply the Op to obtain Stage.
    template<typename Op>
    Stage *getNextOp(const Op *op) {
        StageStore& store = getStageStore();
        auto newInterface = interface;
        auto newColors = colors;
        if (!op->transformInterface(newInterface, newColors, colorsOptions)) {
            return nullptr; // This failed.
        }
        if (!newColors.isConsistent()) {
            return nullptr; // Inconsistent colors. This failed.
        }
        if (Stage *found = store.find(newInterface); found) {
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
        sampler { std::forward<decltype(sampler)>(sampler) },
        colors { std::forward<decltype(colors)>(colors) },
        depth { depth }
    {
        // Compute missing sizes. TODO.
    }
    inline const ColoredInterface& getInterface() const { return interface; }
    std::size_t countChildren();
    inline const std::vector<Next>& getChildrenHandles() {
        guard();
        return nexts;
    }
    Node getChild(Next next);
    std::string shapeToString() const;
    std::string description() const;
};

} // namespace kas
