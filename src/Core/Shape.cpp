#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

Size::Size(std::size_t primaryCount, std::size_t coefficientCount):
    primaryCount { primaryCount },
    coefficientCount { coefficientCount },
    primary {},
    coefficient {}
{
    KAS_ASSERT(primaryCount <= MAX_VARIABLES && coefficientCount <= MAX_VARIABLES);
}

Size& Size::operator=(const Size& other) & {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    primary = other.primary;
    coefficient = other.coefficient;
    return *this;
}

std::span<Size::PowerType> Size::getPrimary() {
    return { primary.data(), primaryCount };
}
std::span<const Size::PowerType> Size::getPrimary() const {
    return { primary.data(), primaryCount };
}
std::span<Size::PowerType> Size::getCoefficient() {
    return { coefficient.data(), coefficientCount };
}
std::span<const Size::PowerType> Size::getCoefficient() const {
    return { coefficient.data(), coefficientCount };
}

Size Size::identity() const {
    return { primaryCount, coefficientCount };
}

Size::Trait Size::getTrait() const {
    bool hasPrimary = false;
    for (auto p: getPrimary()) {
        hasPrimary |= p > 0;
    }
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (auto c: getCoefficient()) {
        hasCoefficient |= c != 0;
        hasNegativeCoefficient |= c < 0;
    }
    if (hasPrimary) {
        return Trait::General;
    } else {
        if (hasCoefficient) {
            if (hasNegativeCoefficient) {
                return Trait::IllegalCoefficient;
            } else {
                return Trait::Coefficient;
            }
        } else {
            return Trait::One;
        }
    }
}

bool Size::is1() const {
    for (auto p: getPrimary()) {
        if (p != 0) return false;
    }
    for (auto c: getCoefficient()) {
        if (c != 0) return false;
    }
    return true;
}

bool Size::isLegalCoefficient() const {
    for (auto p: getPrimary()) {
        if (p != 0) return false;
    }
    for (auto c: getCoefficient()) {
        if (c < 0) return false;
    }
    return true;
}

int Size::getPrimaryPowersSum() const {
    auto primary = getPrimary();
    return std::accumulate(primary.begin(), primary.end(), 0);
}

std::shared_ptr<Size> Size::operator*(const Size& other) const {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newSize = std::make_shared<Size>(*this);
    auto& newPrimary = newSize->primary;
    auto& newCoefficient = newSize->coefficient;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] += other.primary[i];
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        newCoefficient[i] += other.coefficient[i];
    }
    return newSize;
}

std::shared_ptr<Size> Size::Product(const std::vector<std::shared_ptr<Size>>& operands) {
    auto newSize = std::make_shared<Size>(*operands.at(0));
    auto& newPrimary = newSize->primary;
    auto& newCoefficient = newSize->coefficient;
    const auto primaryCount = newSize->primaryCount;
    const auto coefficientCount = newSize->coefficientCount;
    for (std::size_t index = 1; index < operands.size(); ++index) {
        const auto& operand = *operands[index];
        KAS_ASSERT(primaryCount == operand.primaryCount && coefficientCount == operand.coefficientCount);
        for (std::size_t i = 0; i < primaryCount; ++i) {
            newPrimary[i] += operand.primary[i];
        }
        for (std::size_t i = 0; i < coefficientCount; ++i) {
            newCoefficient[i] += operand.coefficient[i];
        }
    }
    return newSize;
}

std::shared_ptr<Size> Size::operator/(const Size &other) const {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newSize = std::make_shared<Size>(*this);
    auto& newPrimary = newSize->primary;
    auto& newCoefficient = newSize->coefficient;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        KAS_ASSERT(newPrimary[i] >= 0);
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        newCoefficient[i] -= other.coefficient[i];
    }
    return newSize;
}

std::optional<Size::Trait> Size::testDividedBy(const Size& other) {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newPrimary = primary;
    bool hasPrimary = false;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        if(newPrimary[i] < 0) return std::nullopt;
        hasPrimary |= newPrimary[i] > 0;
    }
    primary = newPrimary;
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        coefficient[i] -= other.coefficient[i];
        hasCoefficient |= coefficient[i] != 0;
        hasNegativeCoefficient |= coefficient[i] < 0;
    }
    if (hasPrimary) {
        return Trait::General;
    } else {
        if (hasCoefficient) {
            if (hasNegativeCoefficient) {
                return Trait::IllegalCoefficient;
            } else {
                return Trait::Coefficient;
            }
        } else {
            return Trait::One;
        }
    }
}

std::optional<Size::Trait> Size::canBeDividedBy(const Size& other) const {
    Size temp { *this };
    return temp.testDividedBy(other);
}

std::size_t Size::estimate(const BindingContext& ctx) const {
    return eval<std::size_t>(
        [&ctx](std::size_t i) { return ctx.getPrimaryEstimate(i); },
        [&ctx](std::size_t i) { return ctx.getCoefficientEstimate(i); }
    );
}

std::string Size::toString(const BindingContext& ctx) const {
    KAS_ASSERT(primaryCount == ctx.getPrimaryCount() && coefficientCount == ctx.getCoefficientCount());
    std::stringstream result;
    bool hasAnything = false;
    auto print = [&result, &hasAnything](std::size_t count, auto&& f, const ExprType& powers) {
        for (std::size_t i = 0; i < count; ++i) {
            if (powers[i] != 0) {
                if (hasAnything) {
                    result << "*";
                }
                result << f(i);
                if (powers[i] != 1) {
                    result << "^" << static_cast<int>(powers[i]);
                }
                hasAnything = true;
            }
        }
    };
    print(coefficientCount, [&ctx](std::size_t i) { return ctx.getCoefficientAlias(i); }, coefficient);
    print(primaryCount, [&ctx](std::size_t i) { return ctx.getPrimaryAlias(i); }, primary);
    if (!hasAnything) {
        result << "1";
    }
    return result.str();
}

LabeledSize::LabeledSize(std::size_t primaryCount, std::size_t coefficientCount):
    Size { primaryCount, coefficientCount },
    trait { Trait::One }
{}

LabeledSize::LabeledSize(const Size& size):
    Size { size },
    trait { size.getTrait() }
{}

LabeledSize LabeledSize::identity() const {
    return { primaryCount, coefficientCount };
}

bool LabeledSize::is1() const {
    return trait == Trait::One;
}

bool LabeledSize::isLegalCoefficient() const {
    return trait == Trait::Coefficient;
}

bool LabeledSize::isIllegalCoefficient() const {
    return trait == Trait::IllegalCoefficient;
}

bool LabeledSize::isIndeterminedCoefficient() const {
    return trait == Trait::Coefficient || trait == Trait::IllegalCoefficient;
}

bool LabeledSize::isGeneral() const {
    return trait == Trait::General;
}

bool LabeledSize::testDividedBy(const Size& other) {
    auto res = Size::testDividedBy(other);
    if (res.has_value()) {
        trait = res.value();
        return true;
    }
    return false;
}

LabeledSize& LabeledSize::operator*=(const LabeledSize& other) {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    bool hasPrimary = false;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        primary[i] += other.primary[i];
        hasPrimary |= primary[i] > 0;
    }
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        coefficient[i] += other.coefficient[i];
        hasCoefficient |= coefficient[i] != 0;
        hasNegativeCoefficient |= coefficient[i] < 0;
    }
    if (hasPrimary) {
        trait = Trait::General;
    } else {
        if (hasCoefficient) {
            if (hasNegativeCoefficient) {
                trait = Trait::IllegalCoefficient;
            } else {
                trait = Trait::Coefficient;
            }
        } else {
            trait = Trait::One;
        }
    }
    return *this;
}

LabeledSize LabeledSize::operator*(const LabeledSize& other) const {
    auto newSize = *this;
    newSize *= other;
    return newSize;
}

std::optional<LabeledSize> LabeledSize::absorbCoefficientNumeratorToDenominator(const LabeledSize& other) const {
    KAS_ASSERT(isIndeterminedCoefficient() && other.isIndeterminedCoefficient());
    Size res(primaryCount, coefficientCount);
    bool contribution = false;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        auto c1 = coefficient[i];
        auto c2 = other.coefficient[i];
        contribution |= c1 < 0 && c2 > 0;
        res.coefficient[i] = c1 + c2;
    }
    if (contribution) {
        return { std::move(res) };
    } else {
        return std::nullopt;
    }
}

int LabeledSize::scoreOfGeneralDimension(const LabeledSize& other) const {
    KAS_ASSERT(isIllegalCoefficient() && other.isGeneral());
    int score = 0;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        score += std::abs(coefficient[i]) + std::abs(other.coefficient[i]) - std::abs(coefficient[i] + other.coefficient[i]);
    }
    return score;
}

Shape::Shape(const std::vector<std::shared_ptr<Size>>& sizes):
    sizes(sizes)
{}
Shape::Shape(std::vector<std::shared_ptr<Size>>&& sizes):
    sizes(std::move(sizes))
{}

const std::vector<std::shared_ptr<Size>>& Shape::getSizes() const {
    return sizes;
}

std::size_t Shape::size() const {
    return sizes.size();
}

const std::shared_ptr<Size>& Shape::operator[](std::size_t index) const {
    return sizes.at(index);
}

Size Shape::totalSize() const {
    return std::move(*Size::Product(sizes));
}

Shape Shape::replace(
    std::vector<std::size_t> drops,
    std::vector<std::pair<std::size_t, std::shared_ptr<Size>>> adds
) const {
    return Shape { ReplaceVector(sizes, drops, adds) };
}

std::vector<std::size_t> Shape::estimate(const BindingContext& ctx) const {
    std::vector<std::size_t> result;
    for (const auto& size: sizes) {
        result.emplace_back(size->estimate(ctx));
    }
    return result;
}

std::string Shape::toString(const BindingContext& ctx) const {
    return VectorToString(sizes, std::function([&ctx](const std::shared_ptr<Size>& size) -> std::string {
        return size->toString(ctx);
    }));
}

Allowance::Allowance(const Size& shape, const BindingContext& ctx):
    primary {},
    coefficientLower {},
    coefficientUpper {}
{
    auto primaryMeta = ctx.getPrimaryMetadata();
    auto coefficientMeta = ctx.getCoefficientMetadata();
    auto primary = shape.getPrimary();
    auto coefficient = shape.getCoefficient();
    const std::size_t primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
    for (std::size_t i = 0; i < primaryCount; ++i) {
        // Observe that in the sampling process, the primary variables are generated only by MapReduce. So we can limit it with maximumOccurrence.
        if (static_cast<std::size_t>(primary[i]) < primaryMeta[i].maximumOccurrence) {
            this->primary[i] = primaryMeta[i].maximumOccurrence - static_cast<std::size_t>(primary[i]);
        }
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        // Similar for coefficient.
        coefficientLower[i] = -coefficientMeta[i].maximumOccurrence - static_cast<std::size_t>(coefficient[i]);
        coefficientUpper[i] = coefficientMeta[i].maximumOccurrence - static_cast<std::size_t>(coefficient[i]);
    }
}

bool Allowance::withinAllowance(const Size& size) const {
    auto primary = size.getPrimary();
    auto coefficient = size.getCoefficient();
    const std::size_t primaryCount = primary.size(), coefficientCount = coefficient.size();
    for (std::size_t i = 0; i < primaryCount; ++i) {
        if (primary[i] > this->primary[i]) {
            return false;
        }
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        if (coefficient[i] < coefficientLower[i] || coefficient[i] > coefficientUpper[i]) {
            return false;
        }
    }
    return true;
}

} // namespace kas
