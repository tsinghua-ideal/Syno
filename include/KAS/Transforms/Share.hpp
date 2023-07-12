#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


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
    ShareOp(const Dimension& output);
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return DimensionTypeHash(Type); }
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInputL() const override { return &inputLhs; }
    Dimension getInputR() const override { return &inputRhs; }
    Values value(const Values& known) const override;

    std::pair<bool, CompactColor> transformColor(CompactColor fro1, CompactColor fro2) const override;

    bool operator==(const ShareOp& other) const noexcept {
        return output == other.output;
    }

    // Due to canonicalization reasons, we require ShareOp's to be chained, and RHS to be from weight.
    // Just like this:
    //
    //     Split(345)   Weight
    //          └────┬────┘
    //             Share     Weight
    //               └────┬────┘
    //                  Share
    //
    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t maximumTensors;
        std::size_t maxColorTags() const {
            return maximumTensors - 1;
        }
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        AllowanceExceeded,
        MaximumTensorsExceeded,
        SuccessfulGenerations,
    )
    static std::vector<const ShareOp *> Generate(PrimitiveOpStore& store, const Dimensions& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<ShareOp>);

} // namespace kas
