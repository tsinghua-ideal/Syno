#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class StrideOp final: public RepeatLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Stride;
    class Input final: public RepeatLikeOp::Input {
    public:
        Input(const StrideOp* op):
            RepeatLikeOp::Input { op }
        {}
        const Size& size() const noexcept override { return getDerivedOp<StrideOp>()->sz; }
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
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    bool operator==(const StrideOp& other) const noexcept {
        return output == other.output && stride == other.stride;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
    };
    static std::vector<const StrideOp *> Generate(DimensionStore& store, const ColoredInterface& outputShape, GenerateOptions options);
};

} // namespace kas
