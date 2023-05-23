#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShiftOp final: public RepeatLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Shift;
    class Input final: public RepeatLikeOp::Input {
    public:
        Input(const ShiftOp* op):
            RepeatLikeOp::Input { op }
        {}
        const Size& size() const noexcept override { return op->output.size(); }
        constexpr DimensionType type() const noexcept override { return Type; }
    };

protected:
    int shift;
    Input input;

public:
    ShiftOp(auto&& output, int shift):
        RepeatLikeOp { std::forward<decltype(output)>(output) },
        shift { shift },
        input { this }
    {}
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    bool operator==(const ShiftOp& other) const noexcept {
        return output == other.output && shift == other.shift;
    }

    static std::vector<const ShiftOp *> Generate(DimensionStore& store, const ColoredInterface& outputShape);
};

} // namespace kas
