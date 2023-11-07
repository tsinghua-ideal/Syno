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

Color Color::Repeat(const Color& color) {
    return static_cast<Color>(color).setHeight(color.getHeight() + 1);
}

Color Color::Merge(const Color& lhs, const Color& rhs) {
    return {
        MergeTags(lhs.tags, rhs.tags),
        lhs.dataDiscardingFlag || rhs.dataDiscardingFlag,
        lhs.unorderedScope == rhs.unorderedScope ? lhs.unorderedScope : nullptr,
        std::max(lhs.height, rhs.height) + 1,
        lhs.endsUpReduceFlag && rhs.endsUpReduceFlag,
    };
}

Color& Color::addTag(Tag tag) & {
    // Insert tag into tags and keep it sorted.
    auto it = std::ranges::lower_bound(tags, tag);
    if (it == tags.end() || *it != tag) {
        tags.insert(it, tag);
    }
    return *this;
}

Dimension Color::getUnorderedScope() const {
    return unorderedScope;
}

bool Color::removeTag(Tag tag) {
    return RemoveTag(tags, tag);
}

Color& Color::setUnordered(const DimensionImpl *value) & {
    unorderedScope = value;
    return *this;
}

} // namespace kas
