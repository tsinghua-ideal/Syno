#include "KAS/Core/Colors.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

bool Color::CheckConflict(const std::vector<Tag>& left, const std::vector<Tag>& right) {
    auto itL = left.begin();
    auto itR = right.begin();
    while (itL != left.end() && itR != right.end()) {
        if (*itL == *itR) {
            return true;
        } else if (*itL < *itR) {
            ++itL;
        } else {
            ++itR;
        }
    }
    return false;
}

void Color::merge(const Color& other) {
    auto mergeTags = [](std::vector<Tag>& tags, const std::vector<Tag>& otherTags) {
        // First merge them.
        auto it1 = tags.begin();
        auto it2 = otherTags.begin();
        std::vector<Tag> newTags;
        while (it1 != tags.end() && it2 != otherTags.end()) {
            if (*it1 < *it2) {
                newTags.emplace_back(*it1);
                ++it1;
            } else {
                newTags.emplace_back(*it2);
                ++it2;
            }
        }
        if (it1 != tags.end()) {
            newTags.insert(newTags.end(), it1, tags.end());
        } else if (it2 != otherTags.end()) {
            newTags.insert(newTags.end(), it2, otherTags.end());
        }
        // Then unique.
        auto uniqueResult = std::ranges::unique(newTags);
        newTags.erase(uniqueResult.begin(), uniqueResult.end());
        tags = std::move(newTags);
    };

    // Merge independently.
    mergeTags(tagsLeft, other.tagsLeft);
    mergeTags(tagsRight, other.tagsRight);

    // Now check for conflicts.
    if (CheckConflict(tagsLeft, tagsRight)) {
        KAS_CRITICAL("Merging conflicting colors!");
    }

    dataDiscardingFlag = dataDiscardingFlag || other.dataDiscardingFlag;
}

void Color::addTag(const Dimension& dim) {
    const auto& input = dim.as<MergeLikeOp::Input>();
    std::vector<Tag>& tags = input.getOrder() == Order::Left ? tagsLeft : tagsRight;
    auto op = input.getOp();
    // Insert tag into tags and keep it sorted.
    auto it = std::ranges::lower_bound(tags, op);
    if (it == tags.end() || *it != op) {
        tags.insert(it, op);
    }
}

bool Color::disjoint(const Color& other) const {
    return !CheckConflict(tagsLeft, other.tagsRight) && !CheckConflict(other.tagsLeft, tagsRight);
}

bool Color::CheckFinalization(const std::vector<Interface>& tensors) {
    struct Visitor: public DimVisitor {
        std::map<Dimension, CompactColor, Dimension::HashLessThan>& colorMap;
        Visitor(std::map<Dimension, CompactColor, Dimension::HashLessThan>& colorMap): colorMap(colorMap) {}
        bool fail = false;
        void visit(const RepeatLikeOp::Input& dim) override {
            auto [accept, newColor] = dim.getOp()->transformColor(colorMap.at(&dim));
            if (!accept) {
                fail = true;
            }
            colorMap.emplace(dim.getOp()->output, newColor);
            visit(dim.getOp()->output);
        }
        void visit(const SplitLikeOp::Input& dim) override {
            auto [accept, newColorL, newColorR] = dim.getOp()->transformColor(colorMap.at(&dim));
            if (!accept) {
                fail = true;
            }
            colorMap.emplace(dim.getOp()->outputLhs, newColorL);
            colorMap.emplace(dim.getOp()->outputRhs, newColorR);
            visit(dim.getOp()->outputLhs);
            visit(dim.getOp()->outputRhs);
        }
        void visit(const MergeLikeOp::Input& dim) override {
            auto otherDim = dim.getOther();
            if (auto it = colorMap.find(otherDim); it != colorMap.end()) {
                auto lhs = colorMap.at(&dim);
                auto rhs = it->second;
                if (dim.getOrder() == Order::Right) {
                    std::swap(lhs, rhs);
                }
                auto [accept, newColor] = dim.getOp()->transformColor(lhs, rhs);
                if (!accept) {
                    fail = true;
                }
                colorMap.emplace(dim.getOp()->output, newColor);
                visit(dim.getOp()->output);
            }
        }
        using DimVisitor::visit;
        bool pass() {
            return !fail;
        }
    };
    std::map<Dimension, CompactColor, Dimension::HashLessThan> colorMap;
    Visitor visitor { colorMap };
    for (int color = 0; auto&& tensor: tensors) {
        for (auto&& dim: tensor) {
            auto [_, inserted] = colorMap.emplace(dim, color);
            if (!inserted) {
                KAS_REPORT_DIMENSION_HASH_COLLISION(_->first, dim);
            }
            visitor.visit(dim);
        }
        ++color;
    }
    return visitor.pass();
}

auto ColoredDimension::deduceOrigin() const -> Origin {
    if (color.countRightTags() > 0) {
        KAS_ASSERT(!color.isDataDiscarding(), "A weight dimension must not be data discarding!");
        // As required by canonicalization, rhs of ShareOp is not allowed to be further transformed and must be weight.
        return Origin::Weight;
    } else if (color.isDataDiscarding()) {
        return Origin::Unfold;
    } else {
        auto t = dimension.type();
        switch (t) {
        case DimensionType::MapReduce:
        case DimensionType::Shift:
        case DimensionType::Stride:
        case DimensionType::Split:
        case DimensionType::Unfold:
        case DimensionType::Share: // In this case, always ShareL.
            return Origin::Input;
        case DimensionType::Iterator:
        case DimensionType::Merge: // At most one of the two MergeOp::Input from the same MergeOp is weight.
            return Origin::BothPossible;
        default:
            KAS_UNREACHABLE("Unknown DimensionType");
        }
    }
}

ColoredInterface ColoredInterface::substitute1to1(const Dimension& fro, const Dimension& to, bool addDataDiscardingFlag) const {
    auto newInterface = *this;
    auto coloredFro = (*this)[fro];
    auto coloredTo = ColoredDimension { to, coloredFro.color };
    if (addDataDiscardingFlag) {
        coloredTo.color.setDataDiscarding(true);
    }
    bool res = WeakOrderedSubstituteVector1To1IfAny(newInterface.items, fro, std::move(coloredTo), Dimension::HashLessThan{}, ColoredDimension::Projection{});
    KAS_ASSERT(res);
    return newInterface;
}

ColoredInterface ColoredInterface::substitute1to2(const Dimension& fro, const Dimension& to1, const Dimension& to2, bool addConstraint) const {
    auto newInterface = *this;
    auto coloredFro = (*this)[fro];
    auto coloredTo1 = ColoredDimension { to1, coloredFro.color };
    auto coloredTo2 = ColoredDimension { to2, coloredFro.color };
    if (addConstraint) {
        coloredTo1.color.addTag(to1);
        coloredTo2.color.addTag(to2);
    }
    bool res = WeakOrderedSubstituteVector1To2IfAny(newInterface.items, fro, std::move(coloredTo1), std::move(coloredTo2), Dimension::HashLessThan{}, ColoredDimension::Projection{});
    KAS_ASSERT(res);
    return newInterface;
}

ColoredInterface ColoredInterface::substitute2to1(const Dimension& fro1, const Dimension& fro2, const Dimension& to, bool absorbDataDiscardingFlagInFro2) const {
    auto newInterface = *this;
    auto coloredFro1 = (*this)[fro1];
    auto coloredFro2 = (*this)[fro2];
    auto coloredTo = ColoredDimension { to, coloredFro1.color };
    coloredTo.color.merge(coloredFro2.color);
    if (absorbDataDiscardingFlagInFro2) {
        coloredTo.color.setDataDiscarding(coloredFro1.color.isDataDiscarding());
    }
    bool res = WeakOrderedSubstituteVector2To1IfAny(newInterface.items, fro1, fro2, std::move(coloredTo), Dimension::HashLessThan{}, ColoredDimension::Projection{});
    KAS_ASSERT(res);
    return newInterface;
}

Graph ColoredInterface::buildGraph() const {
    Graph::Builder builder;
    builder.addTopmost(toDimensions());
    return builder.build();
}

} // namespace kas
