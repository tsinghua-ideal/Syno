#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class ShiftOp final: public RepeatLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Shift;
    class Input final: public RepeatLikeOp::Input {
    public:
        Input(const ShiftOp* op):
            RepeatLikeOp::Input { op }
        {}
        const Size& size() const override { return op->output.size(); }
        constexpr DimensionType type() const noexcept override { return Type; }
    };

protected:
    int shift;
    Input input;

public:
    ShiftOp(const Dimension& output, int shift);
    int getShift() const { return shift; }
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    bool operator==(const ShiftOp& other) const noexcept {
        return output == other.output && shift == other.shift;
    }

    static bool ExceedsMaxValidReshapeShiftPattern(const Size& block, int shift, const BindingContext& ctx, float maximumValidReshapeShiftPattern);

    struct GenerateOptions {
        const BindingContext& ctx;
        const Graph& graph;
        bool disallowShiftAboveUnfold;
        float maximumValidReshapeShiftPattern;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        ExceedsMaxValidReshapeShiftPattern,
        SuccessfulGenerations,
    )
    static std::vector<const ShiftOp *> Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<ShiftOp>);

} // namespace kas
