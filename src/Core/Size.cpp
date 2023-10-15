#include <algorithm>
#include <bitset>
#include <numeric>
#include <set>
#include <sstream>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Size& Size::operator=(const Size& other) & {
    KAS_ASSERT(varCount == other.varCount);
    primary = other.primary;
    coefficient = other.coefficient;
    return *this;
}

std::span<Size::PowerType> Size::getPrimary() {
    return { primary.data(), getPrimaryCount() };
}
std::span<const Size::PowerType> Size::getPrimary() const {
    return { primary.data(), getPrimaryCount() };
}
std::span<Size::PowerType> Size::getCoefficient() {
    return { coefficient.data(), getCoefficientCount() };
}
std::span<const Size::PowerType> Size::getCoefficient() const {
    return { coefficient.data(), getCoefficientCount() };
}

boost::rational<std::size_t> Size::lowerBoundEst(const BindingContext& ctx) const {
    const auto& allConsts = ctx.getAllConsts();
    return std::ranges::min(
        allConsts
        | std::views::transform([&](const ConcreteConsts& consts) {
            return eval<std::size_t>(consts);
        })
    );
}
boost::rational<std::size_t> Size::upperBoundEst(const BindingContext& ctx) const {
    const auto& allConsts = ctx.getAllConsts();
    return std::ranges::max(
        allConsts
        | std::views::transform([&](const ConcreteConsts& consts) {
            return eval<std::size_t>(consts);
        })
    );
}

Size Size::identity() const {
    return { getPrimaryCount(), getCoefficientCount() };
}

Size Size::Identity(const BindingContext& ctx) {
    return { ctx.getPrimaryCount(), ctx.getCoefficientCount() };
}

std::optional<Size::Trait> Size::getTrait() const {
    bool hasPrimary = false;
    for (auto P = getPrimary(); auto p: P) {
        if (p < 0) return std::nullopt;
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
    return GetLimitsUsage(getPrimary()).varsInSize > 0;
}

SizeLimitsUsage Size::GetLimitsUsage(std::span<const PowerType> powers) {
    std::size_t vars = 0, varsPowers = 0;
    for (auto power: powers) {
        vars += power != 0;
        varsPowers += std::abs(power);
    }
    return { vars, varsPowers };
}

SizeLimitsUsage Size::getLimitsUsage() const {
    return GetLimitsUsage(getPrimary()) + GetLimitsUsage(getCoefficient());
}

Size& Size::operator*=(const Size& other) {
    KAS_ASSERT(varCount == other.varCount);
    for (std::size_t i = 0; i < getPrimaryCount(); ++i) {
        primary[i] += other.primary[i];
    }
    for (std::size_t i = 0; i < getCoefficientCount(); ++i) {
        coefficient[i] += other.coefficient[i];
    }
    return *this;
}

Size Size::operator*(const Size& other) const {
    auto result = *this;
    result *= other;
    return result;
}

Size Size::operator^(PowerType power) const {
    auto newSize = Size(*this);
    auto& newPrimary = newSize.primary;
    auto newCoefficient = newSize.coefficient;
    for (std::size_t i = 0; i < getPrimaryCount(); ++i) {
        newPrimary[i] *= power;
    }
    for (std::size_t i = 0; i < getCoefficientCount(); ++i) {
        newCoefficient[i] *= power;
    }
    return newSize;
}

Size& Size::operator/=(const Size& other) {
    KAS_ASSERT(varCount == other.varCount);
    for (std::size_t i = 0; i < getPrimaryCount(); ++i) {
        primary[i] -= other.primary[i];
    }
    for (std::size_t i = 0; i < getCoefficientCount(); ++i) {
        coefficient[i] -= other.coefficient[i];
    }
    return *this;
}

Size Size::operator/(const Size &other) const {
    auto result = *this;
    result /= other;
    return result;
}

std::optional<Size::Trait> Size::testDividedBy(const Size& other) {
    KAS_ASSERT(varCount == other.varCount);
    auto newPrimary = primary;
    bool hasPrimary = false;
    for (std::size_t i = 0; i < getPrimaryCount(); ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        if (newPrimary[i] < 0) return std::nullopt;
        hasPrimary |= newPrimary[i] > 0;
    }
    primary = newPrimary;
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (std::size_t i = 0; i < getCoefficientCount(); ++i) {
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

Size Size::sqrt() const {
    auto newSize = *this;
    for (auto& p: newSize.getPrimary()) {
        p /= 2;
    }
    for (auto& c: newSize.getCoefficient()) {
        c /= 2;
    }
    return newSize;
}

Size Size::primaryPart() const {
    auto result = *this;
    result.coefficient = {};
    return result;
}
Size Size::coefficientPart() const {
    auto result = *this;
    result.primary = {};
    return result;
}

Size Size::getAllowanceUsage() const {
    auto result = *this;
    for (auto& p: result.getPrimary()) {
        p = std::abs(p);
    }
    for (auto& c: result.getCoefficient()) {
        c = std::abs(c);
    }
    return result;
}

Size::EnumerationOptions::EnumerationOptions(const ExprType& lowerBound, const ExprType& upperBound, const SizeLimitsUsage& limits):
    lowerBound(lowerBound),
    upperBound(upperBound),
    limits { limits }
{
    std::bitset<MAX_VARIABLES> seen;
    for (std::size_t i = 0; i < MAX_VARIABLES; ++i) {
        int diff = upperBound[i] - lowerBound[i];
        KAS_ASSERT(diff >= 0);
        if (diff > 0) {
            seen[i] = true;
        }
    }
    for (std::size_t i = 0; i < MAX_VARIABLES; ++i) {
        if (seen[i]) {
            basesIndices.push_back(i);
        }
    }
}

Size::ExprType Size::EnumerationOptions::begin() const {
    return lowerBound;
}

bool Size::EnumerationOptions::isValid(const ExprType& powers) const {
    const auto usage = GetLimitsUsage(powers);
    return usage <= limits;
}

bool Size::EnumerateNext(ExprType& powers, const EnumerationOptions& options) {
    const auto& [basesIndices, lowerBound, upperBound, _] = options;
    for (const std::size_t varToIncrease: basesIndices) {
        const PowerType diff = upperBound[varToIncrease] - powers[varToIncrease];
        KAS_ASSERT(diff >= 0);
        if (diff > 0) {
            ++powers[varToIncrease];
            if (options.isValid(powers)) {
                return true;
            } else {
                // This is not valid. Jump to the next.
                return EnumerateNext(powers, options);
            }
        } else {
            powers[varToIncrease] = lowerBound[varToIncrease];
        }
    }
    return false;
}

Generator<Size> Size::EnumerateSizes(const BindingContext& ctx, Size lowerBound, Size upperBound) {
    const auto primaryOptions = EnumerationOptions { lowerBound.primary, upperBound.primary, ctx.getUsageLimits() };
    auto primary = primaryOptions.begin();
    while (true) {
        const auto& remainingLimits = ctx.getUsageLimits() - Size::GetLimitsUsage(primary);
        const auto coefficientOptions = EnumerationOptions { lowerBound.coefficient, upperBound.coefficient, remainingLimits };
        auto coefficient = coefficientOptions.begin();
        while (true) {
            auto result = Size { ctx.getPrimaryCount(), ctx.getCoefficientCount(), primary, coefficient };
            if (ctx.isSizeLegalToSample(result) && ctx.isSizeValid(result)) {
                co_yield result;
            }
            if (!EnumerateNext(coefficient, coefficientOptions)) break;
        }
        if (!EnumerateNext(primary, primaryOptions)) break;
    }
}

bool Size::operator==(const Size& other) const {
    return std::ranges::equal(primary, other.primary) && std::ranges::equal(coefficient, other.coefficient);
}

std::strong_ordering Size::LexicographicalCompare(const Size& lhs, const Size& rhs) {
    for (std::size_t i = 0; i < lhs.getPrimaryCount(); ++i) {
        auto res = lhs.primary[i] <=> rhs.primary[i];
        if (res != std::strong_ordering::equal) {
            return res;
        }
    }
    for (std::size_t i = 0; i < lhs.getCoefficientCount(); ++i) {
        auto res = lhs.coefficient[i] <=> rhs.coefficient[i];
        if (res != std::strong_ordering::equal) {
            return res;
        }
    }
    return std::strong_ordering::equal;
}

std::string Size::toString(const BindingContext& ctx) const {
    KAS_ASSERT(getPrimaryCount() == ctx.getPrimaryCount() && getCoefficientCount() == ctx.getCoefficientCount());
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
    print(getCoefficientCount(), [&ctx](std::size_t i) { return ctx.getCoefficientAlias(i); }, coefficient);
    print(getPrimaryCount(), [&ctx](std::size_t i) { return ctx.getPrimaryAlias(i); }, primary);
    if (!hasAnything) {
        result << "1";
    }
    return result.str();
}

std::string Size::debugToString() const {
    return BindingContext::ApplyDebugPublicCtx(&Size::toString, *this);
}

boost::rational<int> PaddingSolver::evalFractionalCoefficient(const Size& size) const {
    return Size::EvalFraction<int>(size.getCoefficientCount(), consts.coefficientWrapper(), size.coefficient);
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
    auto fractionalCoefficient = evalFractionalCoefficient(size);
    int remaining = fractionalCoefficient.denominator();
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
        KAS_CRITICAL("Don't you think you have passed in too big a coefficient variable? denominator = {}", fractionalCoefficient.denominator());
    }

    // Now we have the factorization of the remaining denominator. Derive inequalities.
    for (auto [prime, powerPrime]: factorization) {
        PrimeFactorInequality::LHS lhs;
        for (std::size_t i = 0; Power powerPrimary: size.getPrimary()) {
            if (powerPrimary != 0) {
                lhs.emplace_back(i, powerPrimary);
            }
            ++i;
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
    auto inputSizePrimary = inputSize.getPrimary(), outputSizePrimary = outputSize.getPrimary();
    auto computePenalty = [&](std::size_t primaryIndex) -> float {
        int existingPadding = estimateDeterminedPadding(primaryIndex);
        int n = std::max(inputSizePrimary[primaryIndex], outputSizePrimary[primaryIndex]);
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

Allowance::Allowance(const BindingContext& ctx, const Size& currentUsage, bool countSharedCoefficientsAsAllowanceUsage):
    ctx { ctx },
    countSharedCoefficientsAsAllowanceUsage { countSharedCoefficientsAsAllowanceUsage },
    primaryAllowance { },
    coefficientAllowance { }
{
    auto primaryMeta = ctx.getPrimaryMetadata();
    auto coefficientMeta = ctx.getCoefficientMetadata();
    auto primaryUsage = currentUsage.getPrimary();
    auto coefficientUsage = currentUsage.getCoefficient();
    const std::size_t primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
    for (std::size_t i = 0; i < primaryCount; ++i) {
        // Observe that in the sampling process, the primary variables are generated only by Share and Reduce. So we can limit it with maximumOccurrence.
        const Size::PowerType maxOccurrences = primaryMeta[i].maximumOccurrence; // unsigned -> signed
        primaryAllowance[i] = maxOccurrences - primaryUsage[i];
        KAS_ASSERT(primaryUsage[i] >= 0 && primaryAllowance[i] >= 0);
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        // Similar for coefficient.
        const Size::PowerType maxOccurrences = coefficientMeta[i].maximumOccurrence;
        coefficientAllowance[i] = maxOccurrences - coefficientUsage[i];
        KAS_ASSERT(coefficientUsage[i] >= 0 && coefficientAllowance[i] >= 0);
    }
}

bool Allowance::shareWithinAllowance(const Size& size) const {
    auto primary = size.getPrimary();
    auto coefficient = size.getCoefficient();
    const std::size_t primaryCount = primary.size(), coefficientCount = coefficient.size();
    for (std::size_t i = 0; i < primaryCount; ++i) {
        if (std::abs(primary[i]) > this->primaryAllowance[i]) {
            return false;
        }
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        if (std::abs(coefficient[i]) > this->coefficientAllowance[i]) {
            return false;
        }
    }
    return true;
}

Generator<Size> Allowance::enumerateSizes() const {
    const int maxEnum = ctx.getMaxEnumerationsPerVar();
    Size::ExprType primaryLower{}, primaryUpper{}, coefficientLower{}, coefficientUpper{};
    for (std::size_t i = 0; i < ctx.getPrimaryCount(); ++i) {
        const int allowed = primaryAllowance[i];
        primaryLower[i] = 0;
        primaryUpper[i] = std::min(allowed, maxEnum - 1);
    }
    for (std::size_t i = 0; i < ctx.getCoefficientCount(); ++i) {
        const int allowed = coefficientAllowance[i];
        coefficientLower[i] = std::max(maxEnum / 2 - maxEnum + 1, -allowed);
        coefficientUpper[i] = std::min(maxEnum / 2, allowed);
    }
    auto lower = Size { ctx.getPrimaryCount(), ctx.getCoefficientCount(), std::move(primaryLower), std::move(coefficientLower) };
    auto upper = Size { ctx.getPrimaryCount(), ctx.getCoefficientCount(), std::move(primaryUpper), std::move(coefficientUpper) };
    return Size::EnumerateSizes(ctx, std::move(lower), std::move(upper));
}

Generator<Size> Allowance::enumerateDivisors(Size size) const {
    const int maxEnum = ctx.getMaxEnumerationsPerVar();
    auto primary = size.getPrimary(), coefficient = size.getCoefficient();
    Size::ExprType primaryLower{}, primaryUpper{}, coefficientLower{}, coefficientUpper{};
    for (std::size_t i = 0; i < ctx.getPrimaryCount(); ++i) {
        const int current = primary[i];
        const int baseline = current / 2;
        // Note that we will never exceed the allowance, because we crop the range later.
        primaryLower[i] = baseline + maxEnum / 2 - maxEnum + 1;
        primaryUpper[i] = baseline + maxEnum / 2;
        // Then crop the range.
        if (primaryLower[i] < 0) {
            primaryUpper[i] -= primaryLower[i];
            primaryLower[i] = 0;
        }
        if (primaryUpper[i] > current) {
            primaryUpper[i] = current;
        }
    }
    for (std::size_t i = 0; i < ctx.getCoefficientCount(); ++i) {
        const int current = coefficient[i];
        const int allowed = coefficientAllowance[i];
        const int baseline = current / 2;
        // Consider max occurrence.
        // If the sampled power is farther from 0 than the current value, it is counted as an occurrence.
        int lowerBoundOccur, upperBoundOccur;
        if (current >= 0) {
            lowerBoundOccur = -allowed;
            upperBoundOccur = current + allowed;
        } else {
            lowerBoundOccur = current - allowed;
            upperBoundOccur = allowed;
        }
        // Take the intersection.
        coefficientLower[i] = std::max(baseline + maxEnum / 2 - maxEnum + 1, lowerBoundOccur);
        coefficientUpper[i] = std::min(baseline + maxEnum / 2, upperBoundOccur);
    }
    auto lower = Size { ctx.getPrimaryCount(), ctx.getCoefficientCount(), std::move(primaryLower), std::move(coefficientLower) };
    auto upper = Size { ctx.getPrimaryCount(), ctx.getCoefficientCount(), std::move(primaryUpper), std::move(coefficientUpper) };
    for (Size divisor: Size::EnumerateSizes(ctx, std::move(lower), std::move(upper))) {
        auto quotient = size / divisor;
        if (ctx.isSizeLegalToSample(quotient) && ctx.isSizeValid(quotient)) {
            co_yield divisor;
        }
    }
}

} // namespace kas
