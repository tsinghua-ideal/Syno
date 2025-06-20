#include <algorithm>

#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Expand.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

int Dimension::GlobalLessThan::heightOf(const Dimension& dim) const {
    return graph.colorOf(dim).getHeight();
}

bool Dimension::GlobalLessThan::height(const Dimension& lhs, const Dimension& rhs) const {
    return heightOf(lhs) < heightOf(rhs);
}
bool Dimension::GlobalLessThan::hash(const Dimension& lhs, const Dimension& rhs) const noexcept {
    return lhs.hash() < rhs.hash();
}

bool Dimension::GlobalLessThan::operator()(const Dimension& lhs, const Dimension& rhs) const {
    auto height = heightOf(lhs) <=> heightOf(rhs);
    if (height != 0) return height < 0;
    return lhs.hash() < rhs.hash();
}

Dimension::Origin Dimension::deduceOrigin(const Graph& graph) const {
    const auto& color = graph.colorOf(*this);
    if (is(DimensionTypeWithOrder::ShareR)) {
        KAS_ASSERT(!color.isDataDiscarding(), "A weight dimension must not be data discarding!");
        // As required by canonicalization, rhs of ShareOp is not allowed to be further transformed and must be weight.
        return Origin::Weight;
    } else if (color.isDataDiscarding()) {
        return Origin::UnfoldOrExpand;
    } else {
        switch (type()) {
        case DimensionType::Iterator:
            return Origin::InputOrWeight;
        case DimensionType::Reduce:
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
    constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
    constexpr int HexDigits = SizeTypeWidth / 4;
    return fmt::format("[{}]@{}{:0{}x}", size().toString(ctx), type(), hash(), HexDigits);
}

std::string Dimension::descendantsDescription(const BindingContext& ctx) const {
    struct visitor: public DimVisitor {
        const BindingContext& ctx;
        std::string result;
        void visit(const Iterator& dim) override {
            result = std::to_string(dim.getIndex());
        }
        void visit(const Reduce& dim) override {
            result = "";
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
    accept(v);
    return fmt::format("{}({})", description(ctx), std::move(v.result));
}

std::string Dimension::debugDescription() const {
    return BindingContext::ApplyDebugPublicCtx(&Dimension::description, *this);
}
std::string Dimension::debugDescendantsDescription() const {
    return BindingContext::ApplyDebugPublicCtx(&Dimension::descendantsDescription, *this);
}

std::vector<Dimension>::const_iterator Topmost::binarySearch(const Dimension& value) const {
    return WeakOrderedBinarySearch(interface, value, Dimension::HashLessThan{});
}

std::vector<Dimension>::iterator Topmost::binarySearch(const Dimension& value) {
    auto it = std::as_const(*this).binarySearch(value);
    // Amazing trick: https://stackoverflow.com/questions/765148/how-to-remove-constness-of-const-iterator
    return interface.erase(it, it);
}

std::size_t Topmost::binarySearchIndexOf(const Dimension& value) const {
    return std::distance(interface.begin(), binarySearch(value));
}

void Topmost::sortDimensions() {
    std::ranges::sort(interface, Dimension::HashLessThan{});
}

void Topmost::sortExpansions() {
    std::ranges::sort(expansions, Dimension::HashLessThan{}, Expand::PointerToDimension{});
}

void Topmost::sort() {
    sortDimensions();
    sortExpansions();
}
bool Topmost::isSorted() const {
    return
        std::ranges::is_sorted(interface, Dimension::HashLessThan{}) &&
        std::ranges::is_sorted(expansions, Dimension::HashLessThan{}, Expand::PointerToDimension{});
}

std::vector<Dimension> Topmost::getAllDimensions() const {
    std::vector<Dimension> result = interface;
    std::ranges::copy(expansions | std::views::transform(Expand::PointerToDimension{}), std::back_inserter(result));
    return result;
}

std::string Topmost::description(const BindingContext& ctx) const {
    if (expansions.empty()) {
        return fmt::format(
            "[{}]",
            fmt::join(
                interface | std::views::transform([&ctx](const Dimension& dim) { return dim.description(ctx); }),
                ", "
            )
        );
    } else {
        return fmt::format(
            "[{}]{{{}}}",
            fmt::join(
                interface | std::views::transform([&ctx](const Dimension& dim) { return dim.description(ctx); }),
                ", "
            ),
            fmt::join(
                expansions | std::views::transform([&ctx](const Expand *exp) { return exp->output.description(ctx); }),
                ", "
            )
        );
    }
}

std::string Topmost::debugDescription() const {
    return BindingContext::ApplyDebugPublicCtx(&Topmost::description, *this);
}

GraphHandle GraphHandle::FromInterfaces(const std::vector<Topmost>& interfaces) {
    std::vector<Dimension> interface;
    std::vector<const Expand *> expansions;
    for (const auto& topmost: interfaces) {
        interface.insert(interface.end(), topmost.getDimensions().begin(), topmost.getDimensions().end());
        expansions.insert(expansions.end(), topmost.getExpansions().begin(), topmost.getExpansions().end());
    }
    return GraphHandle(std::move(interface), std::move(expansions));
}

void GraphHandle::insert1(const Dimension& value) {
    auto insertionPoint = std::lower_bound(interface.begin(), interface.end(), value, Dimension::HashLessThan{});
    interface.insert(insertionPoint, value);
}

void GraphHandle::moveToExpansions(const Expand *value) {
    auto removed = binarySearch(value->output);
    KAS_ASSERT(removed != interface.end());
    interface.erase(removed);
    auto insertionPoint = std::lower_bound(expansions.begin(), expansions.end(), value, [](const Expand *lhs, const Expand *rhs) {
        return Dimension::HashLessThan{}(lhs->output, rhs->output);
    });
    expansions.insert(insertionPoint, value);
}

void GraphHandle::substitute1to1(const Dimension& fro, const Dimension& to) {
    bool res = WeakOrderedSubstituteVector1To1IfAny(interface, fro, to, Dimension::HashLessThan{});
    KAS_ASSERT(res);
}

void GraphHandle::substitute1to2(const Dimension& fro, const Dimension& to1, const Dimension& to2) {
    bool res = WeakOrderedSubstituteVector1To2IfAny(interface, fro, to1, to2, Dimension::HashLessThan{});
    KAS_ASSERT(res);
}

void GraphHandle::substitute2to1(const Dimension& fro1, const Dimension& fro2, const Dimension& to) {
    bool res = WeakOrderedSubstituteVector2To1IfAny(interface, fro1, fro2, to, Dimension::HashLessThan{});
    KAS_ASSERT(res);
}

Graph GraphHandle::buildGraph() const {
    KAS_ASSERT(isSorted());
    GraphBuilder builder;
    builder.addTopmost(*this);
    return builder.build();
}

std::size_t GraphHandle::hash() const noexcept {
    return std::hash<GraphHandle>{}(*this);
}

void Bottommost::extractReductions() {
    decltype(output) newOutput;
    decltype(reductions) newReductions;
    for (const Dimension& dim: output) {
        if (auto reduction = dim.tryAs<Reduce>(); reduction) {
            newReductions.push_back(reduction);
        } else {
            newOutput.push_back(dim);
        }
    }
    if (newReductions.empty()) return;
    output = std::move(newOutput);
    std::ranges::move(reductions, std::back_inserter(newReductions));
    reductions = std::move(newReductions);
}

Bottommost& Bottommost::operator+=(const Bottommost& other) {
    output.insert(output.end(), other.output.begin(), other.output.end());
    reductions.insert(reductions.end(), other.reductions.begin(), other.reductions.end());
    return *this;
}
Bottommost Bottommost::operator+(const Bottommost& other) const {
    Bottommost result = *this;
    result += other;
    return result;
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
