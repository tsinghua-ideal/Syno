#include "KAS/Core/Colors.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

void Color::merge(const Color& other) {
    const auto& otherTags = other.tags;

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

    dataDiscardingFlag = dataDiscardingFlag || other.dataDiscardingFlag;
}

void Color::addTag(Tag tag) {
    // Insert tag into tags and keep it sorted.
    auto it = std::ranges::lower_bound(tags, tag);
    if (it == tags.end() || *it != tag) {
        tags.insert(it, tag);
    }
}

bool Color::disjointWithWeightDim(const Dimension& dim) const {
    if (dim.is(DimensionTypeWithOrder::ShareR)) {
        auto color = dim.getColor();
        auto removed = color.removeTag(dim.as<MergeLikeOp::Input>().getOp());
        KAS_ASSERT(removed);
        const auto& left = tags;
        const auto& right = color.tags;
        auto itL = left.begin();
        auto itR = right.begin();
        while (itL != left.end() && itR != right.end()) {
            if (*itL == *itR) {
                return false;
            } else if (*itL < *itR) {
                ++itL;
            } else {
                ++itR;
            }
        }
        return true;
    } else {
        KAS_ASSERT(dim.is(DimensionType::Iterator));
        return true;
    }
}

void Color::mergeWeightDim(const Dimension& dim) {
    if (dim.is(DimensionTypeWithOrder::ShareR)) {
        auto color = dim.getColor();
        auto removed = color.removeTag(dim.as<MergeLikeOp::Input>().getOp());
        KAS_ASSERT(removed);
        merge(color);
    } else {
        KAS_ASSERT(dim.is(DimensionType::Iterator));
    }
}

bool Color::removeTag(Tag tag) {
    auto it = std::ranges::lower_bound(tags, tag);
    if (it != tags.end() && *it == tag) {
        tags.erase(it);
        return true;
    }
    return false;
}

const Color Color::None = Color();

bool Color::CheckFinalization(const std::vector<std::vector<Dimension>>& tensors) {
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

} // namespace kas
