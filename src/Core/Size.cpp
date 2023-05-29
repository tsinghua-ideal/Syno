#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Utils/Common.hpp"


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

float Size::lowerBoundEst(const BindingContext& ctx) const {
    const auto& allConsts = ctx.getAllConsts();
    if (allConsts.empty()) {
        const auto& defaultConsts = ctx.getDefaultConsts();
        return eval<float>(defaultConsts);
    }
    return std::ranges::min(
        allConsts
        | std::views::transform([&](const ConcreteConsts& consts) {
            return eval<float>(consts);
        })
    );
}
float Size::upperBoundEst(const BindingContext& ctx) const {
    const auto& allConsts = ctx.getAllConsts();
    if (allConsts.empty()) {
        const auto& defaultConsts = ctx.getDefaultConsts();
        return eval<float>(defaultConsts);
    }
    return std::ranges::max(
        allConsts
        | std::views::transform([&](const ConcreteConsts& consts) {
            return eval<float>(consts);
        })
    );
}

Size Size::identity() const {
    return { primaryCount, coefficientCount };
}

Size Size::Identity(const BindingContext& ctx) {
    return { ctx.getPrimaryCount(), ctx.getCoefficientCount() };
}

Size::Trait Size::getTrait() const {
    bool hasPrimary = false;
    for (auto P = getPrimary(); auto p: P) {
        hasPrimary |= p > 0;
    }
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (auto C = getCoefficient(); auto c: C) {
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
    for (auto P = getPrimary(); auto p: P) {
        if (p != 0) return false;
    }
    for (auto C = getCoefficient(); auto c: C) {
        if (c != 0) return false;
    }
    return true;
}

bool Size::isLegalCoefficient() const {
    for (auto P = getPrimary(); auto p: P) {
        if (p != 0) return false;
    }
    for (auto C = getCoefficient(); auto c: C) {
        if (c < 0) return false;
    }
    return true;
}

bool Size::isGeneral() const {
    return getPrimaryPowersSum() > 0;
}

int Size::getPrimaryPowersSum() const {
    auto primary = getPrimary();
    return std::accumulate(primary.begin(), primary.end(), 0);
}

Size Size::operator*(const Size& other) const {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newSize = Size(*this);
    auto& newPrimary = newSize.primary;
    auto& newCoefficient = newSize.coefficient;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] += other.primary[i];
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        newCoefficient[i] += other.coefficient[i];
    }
    return newSize;
}

Size Size::operator/(const Size &other) const {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newSize = Size(*this);
    auto& newPrimary = newSize.primary;
    auto& newCoefficient = newSize.coefficient;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        // KAS_ASSERT(newPrimary[i] >= 0);
        // But we actually do not need this! We can simply evaluate and see if the result fits.
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

bool Size::quotientIsLegal(const Size& other) const {
    auto res = canBeDividedBy(other);
    return res.has_value() && res.value() != Trait::IllegalCoefficient && res.value() != Trait::One;
}

namespace {

// We frequently need to sample Sizes.
// We would like to enumerate all the possible sizes, in a fashion similar to the way we increment binary numbers.
// For example, there are 5 variables in total. We would like to enumerate only certain variables, then a possible combination is
// basesIndices = { 1, 2, 4 }
// lowerBound = { 0, 0, 0, 0, 0 } (or just nullptr to indicate all 0)
// upperBound = { 0, 1, 2, 0, 1 }
// We would like to enumerate
// 0, 0, 0, 0, 0
// 0, 1, 0, 0, 0
// 0, 0, 1, 0, 0
// 0, 1, 1, 0, 0
// 0, 0, 2, 0, 0
// 0, 1, 2, 0, 0
// 0, 0, 0, 0, 1
// 0, 1, 0, 0, 1
// 0, 0, 1, 0, 1
// 0, 1, 1, 0, 1
// 0, 0, 2, 0, 1
// 0, 1, 2, 0, 1
// which can be done by recursion.
// If this is successful, return true. Else return false.
bool NextSize(Size::ExprType& powers, const std::vector<std::size_t>& basesIndices, const Size::ExprType& lowerBound, const Size::ExprType& upperBound, std::size_t indexOfIndicesOfVarToIncrease = 0) {
    if (indexOfIndicesOfVarToIncrease >= basesIndices.size()) {
        return false;
    }
    std::size_t varToIncrease = basesIndices[indexOfIndicesOfVarToIncrease];
    Size::PowerType diff = upperBound[varToIncrease] - powers[varToIncrease];
    if (diff > 0) {
        ++powers[varToIncrease];
        return true;
    } else {
        powers[varToIncrease] = lowerBound[varToIncrease];
        if (indexOfIndicesOfVarToIncrease == basesIndices.size() - 1) {
            return false;
        } else {
            return NextSize(powers, basesIndices, lowerBound, upperBound, indexOfIndicesOfVarToIncrease + 1);
        }
    }
}

}

Generator<Size> Size::sampleDivisors(const BindingContext& ctx) const {
    auto trait = getTrait();
    switch (trait) {
    case Trait::One:
        co_return;
    case Trait::IllegalCoefficient:
        KAS_CRITICAL("Trying to sample divisors of illegal coefficient!");
    case Trait::Coefficient: {
        // If the size is completely composed of coefficients, we only need to enumerate the powers, and exclude 1 and this.
        std::vector<std::size_t> nonzeroPowers;
        for (std::size_t i = 0; i < coefficientCount; ++i) {
            if (coefficient[i] > 0) {
                nonzeroPowers.push_back(i);
            }
        }
        auto divisor = identity();
        while (true) {
            NextSize(divisor.coefficient, nonzeroPowers, {}, coefficient);
            if (divisor == *this) {
                co_return;
            }
            co_yield divisor;
        }
        break;
    }
    case Trait::General: {
        auto divisor = identity();
        std::vector<std::size_t> primaryNonzeroPowers;
        for (std::size_t i = 0; i < primaryCount; ++i) {
            if (primary[i] > 0) {
                primaryNonzeroPowers.push_back(i);
            }
        }
        std::vector<std::size_t> coefficientNonzeroPowers(coefficientCount);
        std::iota(coefficientNonzeroPowers.begin(), coefficientNonzeroPowers.end(), 0);
        ExprType coefficientLower {}, coefficientUpper {};
        for (std::size_t i = 0; i < coefficientCount; ++i) {
            coefficientLower[i] = coefficient[i] / 2 - 1;
            coefficientUpper[i] = coefficient[i] / 2 + 1;
        }
        // These are just too many! We cannot do it this way! TODO!!!
        while (true) {
            divisor.coefficient = coefficientLower;
            while (true) {
                auto dTrait = divisor.getTrait();
                auto quotient = *this;
                auto qTrait = quotient.testDividedBy(divisor);
                if (
                    dTrait != Trait::IllegalCoefficient && dTrait != Trait::One
                    && qTrait && *qTrait != Trait::IllegalCoefficient && *qTrait != Trait::One
                    && divisor.lowerBoundEst(ctx) > 1.0f && quotient.lowerBoundEst(ctx) > 1.0f
                ) {
                    co_yield divisor;
                }
                if(!NextSize(divisor.coefficient, coefficientNonzeroPowers, coefficientLower, coefficientUpper)) {
                    break;
                }
            }
            if(!NextSize(divisor.primary, primaryNonzeroPowers, {}, primary)) {
                co_return;
            }
        }
        break;
    }
    }
}

Generator<Size> Size::EnumerateSizes(const BindingContext& ctx, Size lowerBound, Size upperBound) {
    const std::size_t primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
    std::vector<std::size_t> primaryNonzeroPowers;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        if (lowerBound.primary[i] < upperBound.primary[i]) {
            primaryNonzeroPowers.push_back(i);
        }
    }
    std::vector<std::size_t> coefficientNonzeroPowers;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        if (lowerBound.coefficient[i] < upperBound.coefficient[i]) {
            coefficientNonzeroPowers.push_back(i);
        }
    }

    auto size = lowerBound;
    while (true) {
        size.coefficient = lowerBound.coefficient;
        while (true) {
            auto trait = size.getTrait();
            if (
                trait != Trait::IllegalCoefficient && trait != Trait::One
                && size.lowerBoundEst(ctx) > 1.0f
            ) {
                co_yield size;
            }
            if(!NextSize(size.coefficient, coefficientNonzeroPowers, lowerBound.coefficient, upperBound.coefficient)) {
                break;
            }
        }
        if(!NextSize(size.primary, primaryNonzeroPowers, lowerBound.primary, upperBound.primary)) {
            co_return;
        }
    }
}

bool Size::operator==(const Size& other) const {
    return std::ranges::equal(primary, other.primary) && std::ranges::equal(coefficient, other.coefficient);
}

bool Size::LexicographicalLEQ(const Size& lhs, const Size& rhs) {
    for (std::size_t i = 0; i < lhs.primaryCount; ++i) {
        auto res = lhs.primary[i] <=> rhs.primary[i];
        if (res != 0) {
            return res < 0;
        }
    }
    for (std::size_t i = 0; i < lhs.coefficientCount; ++i) {
        auto res = lhs.coefficient[i] <=> rhs.coefficient[i];
        if (res != 0) {
            return res < 0;
        }
    }
    return true;
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

std::pair<int, int> PaddingSolver::evalFractionalCoefficient(const Size& size) const {
    return size.evalFraction<int>(size.coefficientCount, consts.coefficientWrapper(), size.coefficient);
}

PaddingSolver::Power PaddingSolver::lhsLowerBound(Prime prime, const PrimeFactorInequality::LHS& lhs) {
    Power result = 0;
    for (auto [index, power]: lhs) {
        result += power * determinedPaddings[index][prime];
    }
    return result;
}

int PaddingSolver::estimateDeterminedPadding(std::size_t primaryIndex) {
    int result = 1;
    for (auto [prime, power]: determinedPaddings[primaryIndex]) {
        result *= std::pow(prime, power);
    }
    return result;
}

void PaddingSolver::addSingleTermInequality(Prime prime, std::size_t indexPrimary, Power powerPrimary, Power powerPrime) {
    Power requiredPower = (powerPrime + powerPrimary - 1) / powerPrimary;
    Power& result = determinedPaddings[indexPrimary][prime];
    result = std::max(result, requiredPower);
}

void PaddingSolver::addMultiTermInequality(Prime prime, PrimeFactorInequality::LHS&& lhs, Power rhs) {
    Power lowewrBound = lhsLowerBound(prime, lhs);
    if (lowewrBound >= rhs) return;
    inequalities[prime].emplace_back(std::move(lhs), rhs);
}

PaddingSolver::PaddingSolver(const BindingContext& ctx, const ConcreteConsts& consts):
    ctx { ctx },
    consts { consts },
    determinedPaddings { ctx.getPrimaryCount() },
    inequalities {}
{}

void PaddingSolver::addConstraint(const Size& size) {
    // First we need to determine the factors of primary variables.
    auto [numerator, denominator] = evalFractionalCoefficient(size);
    int gcd = std::gcd(numerator, denominator);
    int remaining = denominator / gcd;
    // Factor the remaining denominator using prime table.
    Divisors factorization;
    for (Prime prime: Primes) {
        if (prime > remaining) break;
        if (remaining % prime == 0) {
            int count = 0;
            while (remaining % prime == 0) {
                remaining /= prime;
                ++count;
            }
            factorization.emplace(prime, count);
        }
    }
    if (remaining != 1) {
        KAS_CRITICAL("Don't you think you have passed in too big a coefficient variable? denominator = {}", denominator / gcd);
    }

    // Now we have the factorization of the remaining denominator. Derive inequalities.
    for (auto [prime, powerPrime]: factorization) {
        PrimeFactorInequality::LHS lhs;
        for (std::size_t i = 0; i < size.primaryCount; ++i) {
            Power powerPrimary = size.primary[i];
            if (powerPrimary != 0) {
                lhs.emplace_back(i, powerPrimary);
            }
        }
        if (lhs.size() <= 1) {
            // We can eagerly add this to determinedPaddings.
            auto [indexPrimary, powerPrimary] = lhs.at(0);
            addSingleTermInequality(prime, indexPrimary, powerPrimary, powerPrime);
        } else {
            // Add this to inequalities. But we can check whether this is satisfied or not to avoid redundant constraints.
            addMultiTermInequality(prime, std::move(lhs), powerPrime);
        }
    }
}

ConcreteConsts PaddingSolver::solve(const Size& inputSize, const Size& outputSize) {
    auto computePenalty = [&](std::size_t primaryIndex) -> float {
        int existingPadding = estimateDeterminedPadding(primaryIndex);
        int n = std::max(inputSize.primary[primaryIndex], outputSize.primary[primaryIndex]);
        int current = consts.primary[primaryIndex];
        return static_cast<float>(n) / current * existingPadding;
    };
    for (auto&& [prime, ineqs]: inequalities) {
        while (true) {
            // First scan through the inequalities to find which ones we have figured out.
            std::vector<PrimeFactorInequality> validIneqs;
            std::set<std::size_t> relevantVars;
            float leastPenalty = std::numeric_limits<float>::max();
            std::size_t argLeastPenalty = std::numeric_limits<std::size_t>::max();
            for (auto& ineq: ineqs) {
                if (lhsLowerBound(prime, ineq.lhs) < ineq.rhs) {
                    // This ineq is still valid. We need to first collect the relevant variables.
                    for (auto [index, _]: ineq.lhs) {
                        auto [it, inserted] = relevantVars.insert(index);
                        if (inserted) {
                            // Update penalty.
                            float penalty = computePenalty(index);
                            if (penalty < leastPenalty) {
                                leastPenalty = penalty;
                                argLeastPenalty = index;
                            }
                        }
                    }
                    validIneqs.emplace_back(std::move(ineq));
                }
            }
            ineqs = std::move(validIneqs);

            // If there are no more inequalities to satisfy, we are done.
            if (ineqs.empty()) {
                break;
            }

            // Use argLeastPenalty to satisfy as many inequalities as possible.
            for (auto& ineq: ineqs) {
                auto it = std::ranges::lower_bound(ineq.lhs, argLeastPenalty, {}, &std::pair<std::size_t, Power>::first);
                // If this inequality is not controlled by argLeastPenalty, skip it.
                if (it == ineq.lhs.end() || it->first != argLeastPenalty) {
                    continue;
                }
                Power existing = lhsLowerBound(prime, ineq.lhs);
                Power required = ineq.rhs;
                Power delta = required - existing;
                // If this ineq is satisfied, OK.
                if (delta <= 0) {
                    continue;
                }
                // Otherwise we need to add padding.
                delta = (delta + it->second - 1) / it->second;
                determinedPaddings[argLeastPenalty][prime] += delta;
            }
        }
    }

    // Now all the inequalities are satisfied. We can compute the final consts.
    ConcreteConsts result { consts };
    for (std::size_t i = 0; i < ctx.getPrimaryCount(); ++i) {
        int factor = 1;
        for (auto&& [prime, power]: determinedPaddings[i]) {
            factor *= std::pow(prime, power);
        }
        int padded = (consts.primary[i] + factor - 1) / factor * factor;
        result.primary[i] = padded;
    }
    return result;
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
        // Observe that in the sampling process, the primary variables are generated only by Share and MapReduce. So we can limit it with maximumOccurrence.
        Size::PowerType maximumOccurrence = primaryMeta[i].maximumOccurrence;
        this->primary[i] = maximumOccurrence - std::min(maximumOccurrence, primary[i]);
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        // Similar for coefficient.
        Size::PowerType maximumOccurrence = coefficientMeta[i].maximumOccurrence;
        coefficientLower[i] = -maximumOccurrence - coefficient[i];
        coefficientUpper[i] = maximumOccurrence - coefficient[i];
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

Generator<Size> Allowance::enumerateSizes(const BindingContext& ctx) const {
    constexpr Size::PowerType maxEnumerationsPerVar = 4;
    auto primary = this->primary;
    for (std::size_t i = 0; i < ctx.getPrimaryCount(); ++i) {
        primary[i] = std::clamp(primary[i], static_cast<Size::PowerType>(0), maxEnumerationsPerVar);
    }
    auto coefficientLower = this->coefficientLower;
    auto coefficientUpper = this->coefficientUpper;
    for (std::size_t i = 0; i < ctx.getCoefficientCount(); ++i) {
        Size::PowerType &lower = coefficientLower[i], &upper = coefficientUpper[i];
        Size::PowerType enumerations = upper - lower + 1;
        if (enumerations > maxEnumerationsPerVar) {
            if (lower > 0) {
                upper = lower + maxEnumerationsPerVar - 1;
            } else if (upper < 0) {
                lower = upper - maxEnumerationsPerVar + 1;
            } else {
                auto delta = (enumerations - maxEnumerationsPerVar) / 2;
                lower += delta;
                upper -= delta;
            }
        }
    }
    auto lower = Size { ctx.getPrimaryCount(), ctx.getCoefficientCount(), Size::ExprType {}, coefficientLower };
    auto upper = Size { ctx.getPrimaryCount(), ctx.getCoefficientCount(), primary, coefficientUpper };
    return Size::EnumerateSizes(ctx, lower, upper);
}

} // namespace kas
