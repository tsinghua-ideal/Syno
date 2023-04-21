#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class StrideOp final: public RepeatLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Stride;
    class Input final: public RepeatLikeOp::Input {
    public:
        inline Input(const StrideOp* op):
            RepeatLikeOp::Input { op }
        {}
        inline const Size& size() const noexcept override { return getDerivedOp<StrideOp>()->sz; }
        constexpr DimensionType type() const noexcept override { return Type; }
    };

protected:
    Size stride;
    Size sz;
    Input input;

public:
    StrideOp(auto&& output, auto&& stride):
        RepeatLikeOp { std::forward<decltype(output)>(output) },
        stride { stride },
        sz { this->output.size() * this->stride },
        input { this }
    {}
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    inline Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    static std::size_t CountColorTrials;
    static std::size_t CountColorSuccesses;
    bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const override;

    inline bool operator==(const StrideOp& other) const noexcept {
        return output == other.output && stride == other.stride;
    }

    static std::vector<const StrideOp *> Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors);
};

} // namespace kas
