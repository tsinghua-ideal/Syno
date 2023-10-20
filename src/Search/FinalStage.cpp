#include "KAS/Search/FinalStage.hpp"
#include "KAS/Search/NormalStage.hpp"


namespace kas {

const NextFinalizeSlot& FinalStage::getSlot() const {
    const auto& slots = parent.nextFinalizations.getRawSlots();
    auto it = std::ranges::find_if(slots, [this](const NextFinalizeSlot& slot) {
        return slot.nextStage.get() == this;
    });
    KAS_ASSERT(it != slots.end());
    return *it;
}

std::string FinalStage::description() const {
    return value.description(parent.sampler.getBindingContext());
}

} // namespace kas
