#include "KAS/Core/Colors.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

bool Color::AnyCommonTags(const std::vector<Tag>& left, const std::vector<Tag>& right) {
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

std::vector<Color::Tag> Color::MergeTags(const std::vector<Tag>& left, const std::vector<Tag>& right) {
    // First merge them.
    auto it1 = left.begin();
    auto it2 = right.begin();
    std::vector<Tag> tags;
    while (it1 != left.end() && it2 != right.end()) {
        if (*it1 < *it2) {
            tags.emplace_back(*it1);
            ++it1;
        } else {
            tags.emplace_back(*it2);
            ++it2;
        }
    }
    if (it1 != left.end()) {
        tags.insert(tags.end(), it1, left.end());
    } else if (it2 != right.end()) {
        tags.insert(tags.end(), it2, right.end());
    }
    // Then unique.
    auto uniqueResult = std::ranges::unique(tags);
    tags.erase(uniqueResult.begin(), uniqueResult.end());
    return tags;
}

bool Color::RemoveTag(std::vector<Tag>& tags, Tag tag) {
    auto it = std::ranges::lower_bound(tags, tag);
    if (it != tags.end() && *it == tag) {
        tags.erase(it);
        return true;
    }
    return false;
}

std::size_t Color::RemoveTags(std::vector<Tag>& tags, const std::vector<Tag>& toRemove) {
    auto it1 = tags.begin();
    auto it2 = toRemove.begin();
    std::vector<Tag> newTags;
    std::size_t removed = 0;
    while (it1 != tags.end() && it2 != toRemove.end()) {
        if (*it1 < *it2) {
            newTags.emplace_back(*it1);
            ++it1;
        } else if (*it1 == *it2) {
            ++it1;
            ++it2;
            ++removed;
        } else {
            ++it2;
        }
    }
    if (it1 != tags.end()) {
        newTags.insert(newTags.end(), it1, tags.end());
    }
    tags = std::move(newTags);
    return removed;
}

void Color::merge(const Color& other) {
    tags = MergeTags(tags, other.tags);
    dataDiscardingFlag = dataDiscardingFlag || other.dataDiscardingFlag;
}

void Color::addTag(Tag tag) {
    // Insert tag into tags and keep it sorted.
    auto it = std::ranges::lower_bound(tags, tag);
    if (it == tags.end() || *it != tag) {
        tags.insert(it, tag);
    }
}

bool Color::removeTag(Tag tag) {
    return RemoveTag(tags, tag);
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

WeightColor::WeightColor(const Dimension& dim):
    leftTags(dim.getColor().tags), rightTags()
{
    if (dim.is(DimensionType::Share)) {
        const auto& input = dim.as<MergeLikeOp::Input>();
        KAS_ASSERT(input.getOrder() == Order::Right);
        auto *shareOp = input.getOp();
        bool removed = Color::RemoveTag(leftTags, shareOp);
        KAS_ASSERT(removed);
        rightTags.emplace_back(shareOp);
    } else {
        KAS_ASSERT(leftTags.empty(), "If the dimension is not a ShareR, then it should be an Iterator!");
    }
}

std::size_t WeightColor::countLeftTags() const {
    return leftTags.size();
}
std::size_t WeightColor::countRightTags() const {
    return rightTags.size();
}
std::size_t WeightColor::countTags() const {
    return leftTags.size() + rightTags.size();
}

void WeightColor::merge(const WeightColor& other) {
    leftTags = Color::MergeTags(leftTags, other.leftTags);
    rightTags = Color::MergeTags(rightTags, other.rightTags);
    KAS_ASSERT(!Color::AnyCommonTags(leftTags, rightTags), "WeightColor::merge() called with overlapping colors.");
}

void WeightColor::removeAllRightTagsIn(const WeightColor& color) {
    Color::RemoveTags(leftTags, color.rightTags);
}

bool WeightColor::disjointWith(const WeightColor& other) const {
    return !Color::AnyCommonTags(leftTags, other.rightTags) && !Color::AnyCommonTags(rightTags, other.leftTags);
}

} // namespace kas
