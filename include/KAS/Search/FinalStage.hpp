#pragma once

#include "KAS/CodeGen/PyTorchGen.hpp"
#include "KAS/Core/TensorView.hpp"


namespace kas {

class NormalStage;
struct NextFinalizeSlot;

struct FinalStage {
    NormalStage& parent;
    TensorView value;
    PyTorchSpecializedIR pyTorchSpecializedIR;

    const BindingContext& getBindingContext() const;

    FinalStage(NormalStage& parent, TensorView value);
    FinalStage(const FinalStage&) = delete;
    FinalStage(FinalStage&&) = delete;

    const NextFinalizeSlot& getSlot() const;

    std::size_t getVRAMUsage() const;

    std::size_t hash() const { return value.hash(); }
    std::string description() const;
};

} // namespace kas
