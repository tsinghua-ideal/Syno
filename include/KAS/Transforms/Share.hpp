#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class ShareOp final: public MergeLikeOp {
    bool isEqual(const Operation& other) const override;
public:
    static constexpr DimensionType Type = DimensionType::Share;
    class Input final: public MergeLikeOp::Input {
    public:
        Input(const ShareOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        int getRhsOrigin() const {
            KAS_ASSERT(order == Order::Right);
            return getDerivedOp<ShareOp>()->getRhsOrigin();
        }
        const Size& size() const override { return op->output.size(); }
        constexpr DimensionType type() const noexcept override { return Type; }
        bool is(DimensionTypeWithOrder ty) const noexcept override {
            return (ty == DimensionTypeWithOrder::ShareL && order == Order::Left)
                || (ty == DimensionTypeWithOrder::ShareR && order == Order::Right);
        }
        Color computeColor(const GraphBuilder& graphBuilder) const override;
    };

protected:
    int rhsOrigin;
    Input inputLhs, inputRhs;

public:
    ShareOp(const Dimension& output, int rhsOrigin);
    int getRhsOrigin() const noexcept { return rhsOrigin; }
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInputL() const override { return &inputLhs; }
    Dimension getInputR() const override { return &inputRhs; }
    Values value(const Values& known) const override;

    std::pair<bool, CompactColor> transformColor(CompactColor fro1, CompactColor fro2) const override;

    static std::set<int> GetRhsOrigins(const Graph& graph);

    // Due to canonicalization reasons, we require ShareOp's to be chained, and RHS to be from weight.
    // Just like this:
    //
    //     Split(345)   Weight
    //          └────┬────┘
    //             Share     Weight
    //               └────┬────┘
    //                  Share
    //
};

static_assert(PrimitiveOpImpl<ShareOp>);

} // namespace kas
