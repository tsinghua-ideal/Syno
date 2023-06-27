#pragma once

#include <unordered_set>

#include "KAS/Search/AbstractStage.hpp"


namespace kas {

class DimensionsStage;

class DimensionsStageStore {
public:
    struct Hash {
        using is_transparent = void;
        std::size_t operator()(const Dimensions& interface) const noexcept;
        std::size_t operator()(const std::unique_ptr<DimensionsStage>& stage) const noexcept;
    };
    struct Equal {
        using is_transparent = void;
        bool operator()(const Dimensions& lhs, const Dimensions& rhs) const noexcept;
        bool operator()(const Dimensions& lhs, const std::unique_ptr<DimensionsStage>& rhs) const noexcept;
        bool operator()(const std::unique_ptr<DimensionsStage>& lhs, const Dimensions& rhs) const noexcept;
        bool operator()(const std::unique_ptr<DimensionsStage>& lhs, const std::unique_ptr<DimensionsStage>& rhs) const noexcept;
    };

private:
    std::unordered_set<std::unique_ptr<DimensionsStage>, Hash, Equal> interfaces;

public:
    DimensionsStage *find(const Dimensions& interface) const;
    auto insert(std::unique_ptr<DimensionsStage> stage) -> std::pair<decltype(interfaces)::iterator, bool>;
};

class DimensionsStage: public AbstractStage {
protected:
    // The interface decides the hash. Other properties are computed.
    Dimensions interface;

    // Node pointers. We are searching bottom-up, so the children are actually closer to the input tensor.
    NextSlotStore nextSlotStore;

    DimensionsStageStore& getStageStore() const;

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    // Apply the Op to obtain NormalStage.
    template<std::derived_from<DimensionsStage> ChildStageType>
    ChildStageType *getNextOp(const PrimitiveOp *op) {
        DimensionsStageStore& store = getStageStore();
        auto newInterface = op->applyToInterface(interface);
        if (DimensionsStage *found = store.find(newInterface); found) {
            found->addParent(*this);
            return dynamic_cast<ChildStageType *>(found);
        } else {
            std::unique_ptr<DimensionsStage> tempStage = std::make_unique<ChildStageType>(std::move(newInterface), *this, Next::TypeOf(op->getType()));
            if(auto [it, inserted] = store.insert(std::move(tempStage)); inserted) {
                auto childStage = dynamic_cast<ChildStageType *>(it->get());
                KAS_ASSERT(childStage);
                return childStage;
            } else {
                KAS_CRITICAL("NormalStageStore::insert() failed.");
            }
        }
    }

public:
    DimensionsStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp);
    DimensionsStage(Sampler& sampler, Dimensions interface); // The root.

    const Dimensions& getInterface() const final override { return interface; }

    std::size_t hash() const final override;
    std::size_t uncheckedCountChildren() const;
    std::vector<Next> uncheckedGetChildrenHandles() const;
    std::vector<Arc> uncheckedGetChildrenArcs() const;
    std::optional<Arc> uncheckedGetArcFromHandle(Next next) const;
    template<std::derived_from<DimensionsStage> ChildStageType>
    std::optional<Node> uncheckedGetChild(Next next) const {
        return nextSlotStore.findTransform<Node>(next, [this](const NextDimensionsStageSlot& slot) -> Node {
            auto childStage = dynamic_cast<ChildStageType *>(slot.nextStage);
            KAS_ASSERT(childStage);
            return { &sampler, childStage };
        });
    }
    bool canAcceptArc(Arc arc) override;
    std::string description() const final override;
};

} // namespace kas
