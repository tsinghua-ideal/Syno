#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShiftOp final: public RepeatLikeOp {
public:
    class Input final: public RepeatLikeOp::Input {
    public:
        inline Input(const ShiftOp* op):
            RepeatLikeOp::Input { op }
        {}
        inline const Size& size() const noexcept override { return op->output.size(); }
        std::size_t hash() const noexcept override;
        constexpr DimensionType type() const noexcept override { return DimensionType::Shift; }
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
    inline Dimension getInput() const override { return &input; }
    IteratorValue value(const IteratorValue& output) const override;

    inline bool operator==(const ShiftOp& other) const noexcept {
        return output == other.output && shift == other.shift;
    }

    static std::vector<std::unique_ptr<ShiftOp>> Generate(DimensionStore& store, const Interface& outputShape);
};

} // namespace kas
