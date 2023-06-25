#include <algorithm>

#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Dimension::Origin Dimension::deduceOrigin() const {
    const auto& color = getColor();
    if (is(DimensionTypeWithOrder::ShareR)) {
        KAS_ASSERT(!color.isDataDiscarding(), "A weight dimension must not be data discarding!");
        // As required by canonicalization, rhs of ShareOp is not allowed to be further transformed and must be weight.
        return Origin::Weight;
    } else if (color.isDataDiscarding()) {
        return Origin::Unfold;
    } else {
        switch (type()) {
        case DimensionType::Iterator:
            return Origin::BothPossible;
        case DimensionType::MapReduce:
        case DimensionType::Shift:
        case DimensionType::Stride:
        case DimensionType::Split:
        case DimensionType::Unfold:
        case DimensionType::Merge:
        case DimensionType::Share: // In this case, always ShareL.
            return Origin::Input;
        default:
            KAS_UNREACHABLE("Unknown DimensionType");
        }
    }

}

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("[{}]@{}{:x}", size().toString(ctx), type(), hash());
}

std::string Dimension::descendantsDescription(const BindingContext& ctx) const {
    struct visitor: public DimVisitor {
        const BindingContext& ctx;
        std::string result;
        void visit(const Iterator& dim) override {
            result = std::to_string(dim.getIndex());
        }
        void visit(const MapReduce& dim) override {
            result = std::to_string(dim.getPriority());
        }
        void visit(const RepeatLikeOp::Input& dim) override {
            result = dim.getOp()->output.descendantsDescription(ctx);
        }
        void visit(const SplitLikeOp::Input& dim) override {
            auto op = dim.getOp();
            result = fmt::format("{}, {}", op->outputLhs.descendantsDescription(ctx), op->outputRhs.descendantsDescription(ctx));
        }
        void visit(const MergeLikeOp::Input& dim) override {
            result = dim.getOp()->output.descendantsDescription(ctx);
        }
        visitor(const BindingContext& ctx): ctx(ctx) {}
        using DimVisitor::visit;
    };
    visitor v { ctx };
    v.visit(*this);
    return fmt::format("{}({})", description(ctx), std::move(v.result));
}

std::string Dimension::debugDescription() const {
    return BindingContext::ApplyDebugPublicCtx(&Dimension::description, *this);
}
std::string Dimension::debugDescendantsDescription() const {
    return BindingContext::ApplyDebugPublicCtx(&Dimension::descendantsDescription, *this);
}

Dimensions Dimensions::substitute1to1(const Dimension& fro, const Dimension& to) const {
    auto newInterface = *this;
    bool res = WeakOrderedSubstituteVector1To1IfAny(newInterface, fro, to, Dimension::HashLessThan{});
    KAS_ASSERT(res);
    return newInterface;
}

Dimensions Dimensions::substitute1to2(const Dimension& fro, const Dimension& to1, const Dimension& to2) const {
    auto newInterface = *this;
    bool res = WeakOrderedSubstituteVector1To2IfAny(newInterface, fro, to1, to2, Dimension::HashLessThan{});
    KAS_ASSERT(res);
    return newInterface;
}

Dimensions Dimensions::substitute2to1(const Dimension& fro1, const Dimension& fro2, const Dimension& to) const {
    auto newInterface = *this;
    bool res = WeakOrderedSubstituteVector2To1IfAny(newInterface, fro1, fro2, to, Dimension::HashLessThan{});
    KAS_ASSERT(res);
    return newInterface;
}

Graph Dimensions::buildGraph() const {
    Graph::Builder builder;
    builder.addTopmost(*this);
    return builder.build();
}

std::ostream& operator<<(std::ostream& os, kas::DimensionType t) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", t);
    return os;
}

std::ostream& operator<<(std::ostream& os, kas::Direction d) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", d);
    return os;
}

std::ostream& operator<<(std::ostream& os, kas::Order o) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", o);
    return os;
}

} // namespace kas
