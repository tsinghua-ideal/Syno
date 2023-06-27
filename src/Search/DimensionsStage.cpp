#include "KAS/Search/DimensionsStage.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

DimensionsStageStore& DimensionsStage::getStageStore() const {
    return sampler.getStageStore();
}

void DimensionsStage::removeDeadChildrenFromSlots() {
    nextSlotStore.remove([&](const NextDimensionsStageSlot& slot) {
        // Even if the stage is removed, we had better keep it in memory so we avoid redundant computation.
        return slot.nextStage->getFinalizability() == Finalizability::No;
    });
}

void DimensionsStage::removeAllChildrenFromSlots() {
    nextSlotStore.clear();
}

AbstractStage::Finalizability DimensionsStage::checkForFinalizableChildren() const {
    // Check children. Yes if any Yes, No if all No.
    bool allNo = true;
    bool foundYes = false;
    nextSlotStore.forEach([&](const NextDimensionsStageSlot& slot) {
        if (foundYes) {
            return;
        }
        DimensionsStage *child = slot.nextStage;
        if (child->getFinalizability() == Finalizability::Yes) {
            foundYes = true;
            allNo = false;
        } else if (child->getFinalizability() == Finalizability::Maybe) {
            allNo = false;
        }
    });
    if (foundYes) {
        return Finalizability::Yes;
    } else if (allNo) {
        return Finalizability::No;
    } else {
        return Finalizability::Maybe;
    }
}

DimensionsStage::DimensionsStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp):
    AbstractStage { creator, deltaOp },
    interface { std::move(interface) }
{}

DimensionsStage::DimensionsStage(Sampler& sampler, Dimensions interface):
    AbstractStage { sampler },
    interface { std::move(interface) }
{
    std::ranges::sort(this->interface, Dimension::HashLessThan{});
}

std::size_t DimensionsStage::hash() const {
    return DimensionsStageStore::Hash{}(interface);
}

std::size_t DimensionsStage::uncheckedCountChildren() const {
    return nextSlotStore.size();
}

std::vector<Next> DimensionsStage::uncheckedGetChildrenHandles() const {
    return nextSlotStore.toNexts();
}

std::vector<Arc> DimensionsStage::uncheckedGetChildrenArcs() const {
    return nextSlotStore.toArcs(&sampler);
}

std::optional<Arc> DimensionsStage::uncheckedGetArcFromHandle(Next next) const {
    return nextSlotStore.findTransform<Arc>(next, [this](const NextDimensionsStageSlot& slot) -> Arc {
        return slot.toArc(&sampler);
    });
}

bool DimensionsStage::canAcceptArc(Arc arc) {
    return arc.match<bool>(
        [&](const PrimitiveOp *op) -> bool {
            return op->canApplyToInterface(interface);
        },
        [&](const FinalizeOp *op) -> bool {
            return op->toDimensions() == interface;
        }
    );
}

std::string DimensionsStage::description() const {
    return DimensionArrayToString(interface, sampler.getBindingContext());
}

std::size_t DimensionsStageStore::Hash::operator()(const Dimensions& interface) const noexcept {
    return std::hash<Dimensions>{}(interface);
}

std::size_t DimensionsStageStore::Hash::operator()(const std::unique_ptr<DimensionsStage>& stage) const noexcept {
    return (*this)(stage->getInterface());
}

bool DimensionsStageStore::Equal::operator()(const Dimensions& lhs, const Dimensions& rhs) const noexcept {
    return std::ranges::equal(lhs, rhs);
}
bool DimensionsStageStore::Equal::operator()(const Dimensions& lhs, const std::unique_ptr<DimensionsStage>& rhs) const noexcept {
    return (*this)(lhs, rhs->getInterface());
}
bool DimensionsStageStore::Equal::operator()(const std::unique_ptr<DimensionsStage>& lhs, const Dimensions& rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs);
}
bool DimensionsStageStore::Equal::operator()(const std::unique_ptr<DimensionsStage>& lhs, const std::unique_ptr<DimensionsStage>& rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs->getInterface());
}

DimensionsStage *DimensionsStageStore::find(const Dimensions& interface) const {
    KAS_ASSERT(std::ranges::is_sorted(interface, Dimension::HashLessThan{}), "Interface is not sorted.");
    if (auto it = interfaces.find(interface); it != interfaces.end()) {
        return it->get();
    } else {
        return nullptr;
    }
}

auto DimensionsStageStore::insert(std::unique_ptr<DimensionsStage> stage) -> std::pair<decltype(interfaces)::iterator, bool> {
    return interfaces.insert(std::move(stage));
}

} // namespace kas
