#pragma once

#include "KAS/Core/TensorView.hpp"


namespace kas {

class NormalStage;

struct FinalStage {
    NormalStage& parent;
    TensorView value;
    template<std::convertible_to<TensorView> T>
    FinalStage(NormalStage& parent, T&& value): parent { parent }, value { std::forward<T>(value) } {}
    FinalStage(const FinalStage&) = delete;
    FinalStage(FinalStage&&) = delete;

    std::size_t hash() const { return value.hash(); }
    std::string description() const;
};

} // namespace kas
