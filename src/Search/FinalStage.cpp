#include "KAS/CodeGen/PyTorchGen.hpp"
#include "KAS/Search/FinalStage.hpp"
#include "KAS/Search/NormalStage.hpp"


namespace kas {

const BindingContext& FinalStage::getBindingContext() const {
    return parent.sampler.getBindingContext();
}

FinalStage::FinalStage(NormalStage& parent, TensorView value):
    parent { parent }, value { std::move(value) },
    pyTorchSpecializedIR { PyTorchGen::SpecializeIR(
        getBindingContext(),
        this->value,
        parent.sampler.getOptions().maxVRAM
    ) }
{}

const NextFinalizeSlot& FinalStage::getSlot() const {
    const auto& slots = parent.nextFinalizations.getRawSlots();
    auto it = std::ranges::find_if(slots, [this](const NextFinalizeSlot& slot) {
        return slot.nextStage.get() == this;
    });
    KAS_ASSERT(it != slots.end());
    return *it;
}

std::size_t FinalStage::getVRAMUsage() const {
    return PyTorchGen::EstimateVRAMUsage(getBindingContext(), pyTorchSpecializedIR);
}

std::string FinalStage::description() const {
    return value.description(getBindingContext());
}

} // namespace kas
