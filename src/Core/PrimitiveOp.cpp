#include <algorithm>
#include <array>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Interface NextRepeatLike::applyTo(const Interface& interface) const {
    const auto& replace = input.as<RepeatLikePrimitiveOp>().output;
    bool toBeInserted = true;
    auto it = interface.begin();
    Interface result;
    while (it != interface.end()) {
        if (*it != replace) {
            if (toBeInserted && *it > input) {
                result.emplace_back(input);
                toBeInserted = false;
            }
            result.emplace_back(*it);
        }
        ++it;
    }
    if (toBeInserted) {
        result.emplace_back(input);
    }
    return result;
}

Interface NextSplitLike::applyTo(const Interface &interface) const {
    const auto& op = input.as<SplitLikePrimitiveOp>();
    const auto& [replaceLeft, replaceRight] = std::minmax(op.outputLhs, op.outputRhs);
    bool toBeInserted = true;
    auto it = interface.begin();
    Interface result;
    while (it != interface.end()) {
        if (*it != replaceLeft && *it != replaceRight) {
            if (toBeInserted && *it > input) {
                result.emplace_back(input);
                toBeInserted = false;
            }
            result.emplace_back(*it);
        }
        ++it;
    }
    if (toBeInserted) {
        result.emplace_back(input);
    }
    return result;
}

Interface NextMergeLike::applyTo(const Interface &interface) const {
    const auto& replace = inputLhs.as<MergeLikePrimitiveOp>().output;
    std::array<Dimension, 2> substitutes = { std::min(inputLhs, inputRhs), std::max(inputLhs, inputRhs) };
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

} // namespace kas
