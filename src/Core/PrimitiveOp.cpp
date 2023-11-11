#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

void Valuation::assertCanBeConvertedFrom(const Valuation& other) const {
    KAS_ASSERT(type() >= other.type(), "Valuation of a dimension is decaying from {} to {}!", static_cast<int>(other.type()), static_cast<int>(type()));
    if (type() == Type::Oriented && other.type() == Type::Oriented) {
        auto dir = std::get<Direction>(value);
        auto otherDir = std::get<Direction>(other.value);
        KAS_ASSERT(dir == otherDir, "Valuation of a dimension is changing direction from {} to {}!", otherDir, dir);
    }
}

bool Valuation::isRefined(const Valuation& other) const noexcept {
    return type() > other.type();
}

Direction Valuation::extractOrientation() const {
    KAS_ASSERT(type() == Type::Oriented, "Extracting a direction from a dimension which is not oriented!");
    return std::get<Direction>(value);
}

std::optional<Direction> Valuation::tryOrientation() const {
    if (type() == Type::Oriented) {
        return std::get<Direction>(value);
    } else {
        return std::nullopt;
    }
}

const IteratorValue& Valuation::extractValue() const {
    KAS_ASSERT(type() == Type::Valued, "Extracting a dimension which is not yet valued!");
    return std::get<IteratorValue>(value);
}

IteratorValue Valuation::tryValue() const {
    if (type() == Type::Valued) {
        return std::get<IteratorValue>(value);
    } else {
        return {};
    }
}

GraphHandle Operation::appliedToInterface(const GraphHandle& interface) const {
    auto newInterface = interface;
    applyToInterface(newInterface);
    return newInterface;
}

Color RepeatLikeOp::Input::computeColor(const GraphBuilder& graphBuilder) const {
    return Color::Repeat(graphBuilder.colorOf(op->output));
}

std::pair<bool, CompactColor> RepeatLikeOp::transformColor(CompactColor fro) const {
    return { true, fro }; 
}

bool RepeatLikeOp::canApplyToInterface(const GraphHandle& interface) const {
    return interface.contains(output);
}

void RepeatLikeOp::applyToInterface(GraphHandle& interface) const {
    interface.substitute1to1(output, getInput());
}

std::string RepeatLikeOp::description(const BindingContext& ctx) const {
    return fmt::format("{} -> {}", getInput().description(ctx), output.description(ctx));
}

std::string RepeatLikeOp::descendantsDescription(const BindingContext& ctx) const {
    return fmt::format("{} -> {}", getInput().description(ctx), output.descendantsDescription(ctx));
}

Color SplitLikeOp::Input::computeColor(const GraphBuilder& graphBuilder) const {
    return Color::Merge(graphBuilder.colorOf(op->outputLhs), graphBuilder.colorOf(op->outputRhs));
}

std::tuple<bool, CompactColor, CompactColor> SplitLikeOp::transformColor(CompactColor fro) const {
    return {true, fro, fro};
}

bool SplitLikeOp::canApplyToInterface(const GraphHandle &interface) const {
    return interface.contains(outputLhs) && interface.contains(outputRhs);
}

void SplitLikeOp::applyToInterface(GraphHandle &interface) const {
    interface.substitute2to1(outputLhs, outputRhs, getInput());
}

std::string SplitLikeOp::description(const BindingContext &ctx) const {
    return fmt::format("{} -> {}, {}", getInput().description(ctx), outputLhs.description(ctx), outputRhs.description(ctx));
}

std::string SplitLikeOp::descendantsDescription(const BindingContext &ctx) const {
    return fmt::format("{} -> {}, {}", getInput().description(ctx), outputLhs.descendantsDescription(ctx), outputRhs.descendantsDescription(ctx));
}

Color MergeLikeOp::Input::computeColor(const GraphBuilder& graphBuilder) const {
    return Color::Repeat(graphBuilder.colorOf(op->output));
}

std::pair<bool, CompactColor> MergeLikeOp::transformColor(CompactColor fro1, CompactColor fro2) const {
    return {true, fro1 | fro2};
}

bool MergeLikeOp::canApplyToInterface(const GraphHandle &interface) const {
    return interface.contains(output);
}

void MergeLikeOp::applyToInterface(GraphHandle &interface) const {
    interface.substitute1to2(output, getInputL(), getInputR());
}

std::string MergeLikeOp::description(const BindingContext &ctx) const {
    return fmt::format("{}, {} -> {}", getInputL().description(ctx), getInputR().description(ctx), output.description(ctx));
}

std::string MergeLikeOp::descendantsDescription(const BindingContext &ctx) const {
    return fmt::format("{}, {} -> {}", getInputL().description(ctx), getInputR().description(ctx), output.descendantsDescription(ctx));
}

} // namespace kas
