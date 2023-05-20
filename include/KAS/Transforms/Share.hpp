#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareOp final: public MergeLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Share;
    class Input final: public MergeLikeOp::Input {
    public:
        Input(const ShareOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        const Size& size() const noexcept override { return op->output.size(); }
        constexpr DimensionType type() const noexcept override { return Type; }
        bool is(DimensionTypeWithOrder ty) const noexcept override {
            return (ty == DimensionTypeWithOrder::ShareL && order == Order::Left)
                || (ty == DimensionTypeWithOrder::ShareR && order == Order::Right);
        }
    };

protected:
    Input inputLhs, inputRhs;

public:
    ShareOp(auto&& output):
        MergeLikeOp { std::forward<decltype(output)>(output) },
        inputLhs { this, Order::Left },
        inputRhs { this, Order::Right }
    {}
    constexpr DimensionType getType() const noexcept override { return Type; }
    constexpr std::size_t initialHash() const noexcept override { return static_cast<std::size_t>(Type); }
    Dimension getInputL() const override { return &inputLhs; }
    Dimension getInputR() const override { return &inputRhs; }
    Values value(const Values& known) const override;

    std::pair<bool, CompactColor> transformColor(CompactColor fro1, CompactColor fro2) const override;
    // Add new constraints.
    ColoredInterface applyToInterface(const ColoredInterface& interface) const override;

    bool operator==(const ShareOp& other) const noexcept {
        return output == other.output;
    }

    // We require a total order of Op's above a chain of ShareOp's.
    // Just like this:
    //
    //     Split(345) Stride(234)
    //          └────┬────┘
    //             Share   Unfold(123)
    //               └────┬────┘
    //                  Share
    //
    // We simply sort by the opHash of Op's.
    // When building, we must build the Op's from right to left, and in increasing order of opHash.
    static bool IsSharedDimensionCanonical(const PrimitiveOp *op, const Graph& graph);

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
        std::size_t maximumTensors;
        std::size_t maxColorTags() {
            return maximumTensors - 1;
        }
    };
    static inline std::size_t CountGenerateInvocations = 0;
    static inline std::size_t CountGenerateAttempts = 0; // Equals the sum of below.
    static inline std::size_t CountDisallowedAttempts = 0;
    static inline std::size_t CountAllowanceExceeded = 0;
    static inline std::size_t CountMaximumTensorsExceeded = 0;
    static inline std::size_t CountSuccessfulGenerations = 0;
    static std::vector<const ShareOp *> Generate(DimensionStore& store, const ColoredInterface& interface, GenerateOptions options);
};

} // namespace kas
