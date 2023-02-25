#include <algorithm>
#include <array>
#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("{}@{}", size().toString(ctx), fmt::ptr(inner));
}

Interface Dimension::applyRepeatLike(const Dimension& dim, const Interface& interface) {
    const auto& replace = dim.as<RepeatLikePrimitiveOp>().output;
    bool toBeInserted = true;
    auto it = interface.begin();
    Interface result;
    while (it != interface.end()) {
        if (*it != replace) {
            if (toBeInserted && *it > dim) {
                result.emplace_back(dim);
                toBeInserted = false;
            }
            result.emplace_back(*it);
        }
        ++it;
    }
    if (toBeInserted) {
        result.emplace_back(dim);
    }
    return result;
}

Interface Dimension::applyMergeLike(const Dimension &dimLeft, const Dimension &dimRight, const Interface &interface) {
    const auto& replace = dimLeft.as<MergeLikePrimitiveOp>().output;
    std::array<Dimension, 2> substitutes = { std::min(dimLeft, dimRight), std::max(dimLeft, dimRight) };
    auto sub = substitutes.begin();
    auto it = interface.begin();
    Interface result;
    while (it != interface.end()) {
        if (*it != replace) {
            while (sub != substitutes.end() && *it > *sub) {
                result.emplace_back(*sub);
                ++sub;
            }
            result.emplace_back(*it);
        }
        ++it;
    }
    while (sub != substitutes.end()) {
        result.emplace_back(*sub);
        ++sub;
    }
    return result;
}

Interface Dimension::applySplitLike(const Dimension &dim, const Interface &interface) {
    const auto& op = dim.as<SplitLikePrimitiveOp>();
    const auto& [replaceLeft, replaceRight] = std::minmax(op.outputLhs, op.outputRhs);
    bool toBeInserted = true;
    auto it = interface.begin();
    Interface result;
    while (it != interface.end()) {
        if (*it != replaceLeft && *it != replaceRight) {
            if (toBeInserted && *it > dim) {
                result.emplace_back(dim);
                toBeInserted = false;
            }
            result.emplace_back(*it);
        }
        ++it;
    }
    if (toBeInserted) {
        result.emplace_back(dim);
    }
    return result;
}

} // namespace kas
