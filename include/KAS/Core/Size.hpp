#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <optional>

#include <boost/rational.hpp>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Coroutine.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

struct Size;
template<typename R>
concept SizeRange =
    std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, Size>;

struct Size {
    friend class PaddingSolver;
public:
    constexpr static std::size_t MAX_VARIABLES = 8;
    using PowerType = std::int8_t;
    using ExprType = std::array<PowerType, MAX_VARIABLES>;
    enum class Trait {
        One, // powers of primary == 0, powers of coefficient == 0.
        Coefficient, // powers of primary == 0, powers of coefficient >= 0 but not all 0.
        IllegalCoefficient, // powers of primary == 0, powers of coefficient < 0.
        General, // powers of primary >= 0 but not all 0.
    };

private:
    // Use the first half to store primaryCount, and the second half to store coefficientCount.
    using VariableCountType = std::uint8_t;
    static_assert(std::is_unsigned_v<VariableCountType>);
    static constexpr int HalfVariableCountType = std::numeric_limits<VariableCountType>::digits / 2;
    static constexpr VariableCountType LowerVariableCountTypeMask = (1 << HalfVariableCountType) - 1;
    const VariableCountType varCount;
    static VariableCountType ToVariableCountType(std::size_t primaryCount, std::size_t coefficientCount) {
        KAS_ASSERT(primaryCount <= MAX_VARIABLES && coefficientCount <= MAX_VARIABLES);
        return static_cast<VariableCountType>((primaryCount << HalfVariableCountType) | coefficientCount);
    }
    // Powers of large variables. Must be non-negative.
    ExprType primary;
    // Powers of small variables. Can be negative (when in denominator).
    ExprType coefficient;

    std::size_t getPrimaryCount() const { return varCount >> HalfVariableCountType; }
    std::size_t getCoefficientCount() const { return varCount & LowerVariableCountTypeMask; }

    template<std::integral ValueType, typename Consts>
    static boost::rational<ValueType> EvalFraction(std::size_t cnt, Consts&& consts, const Size::ExprType& powers) {
        ValueType nominator = 1;
        ValueType denominator = 1;
        for (std::size_t i = 0; i < cnt; ++i) {
            if (powers[i] > 0) {
                const ValueType v = consts(i);
                auto power = static_cast<std::size_t>(powers[i]);
                for (std::size_t j = 0; j < power; ++j) {
                    nominator *= v;
                }
            } else if (powers[i] < 0) {
                const ValueType v = consts(i);
                auto power = static_cast<std::size_t>(-powers[i]);
                for (std::size_t j = 0; j < power; ++j) {
                    denominator *= v;
                }
            }
        }
        return boost::rational<ValueType>(nominator, denominator);
    };

public:
    Size(std::size_t primaryCount, std::size_t coefficientCount):
        varCount { ToVariableCountType(primaryCount, coefficientCount) },
        primary {},
        coefficient {}
    {
        KAS_ASSERT(primaryCount <= MAX_VARIABLES && coefficientCount <= MAX_VARIABLES);
    }
    template<typename Tp, typename Tc>
    Size(std::size_t primaryCount, std::size_t coefficientCount, Tp&& primary, Tc&& coefficient):
        varCount { ToVariableCountType(primaryCount, coefficientCount) },
        primary { std::forward<Tp>(primary) },
        coefficient { std::forward<Tc>(coefficient) }
    {
        KAS_ASSERT(primaryCount <= MAX_VARIABLES && coefficientCount <= MAX_VARIABLES);
    }
    Size& operator=(const Size& other) &;
    std::span<PowerType> getPrimary();
    std::span<const PowerType> getPrimary() const;
    std::span<PowerType> getCoefficient();
    std::span<const PowerType> getCoefficient() const;

    template<std::integral ValueType, typename Tp, typename Tc>
    boost::rational<ValueType> evalFraction(Tp&& p, Tc&& c) const {
        auto fractionP = EvalFraction<ValueType>(getPrimaryCount(), std::forward<Tp>(p), primary);
        auto fractionC = EvalFraction<ValueType>(getCoefficientCount(), std::forward<Tc>(c), coefficient);
        return fractionP * fractionC;
    };
    template<std::integral ValueType>
    boost::rational<ValueType> evalFraction(const ConcreteConsts& consts) const {
        return evalFraction<ValueType>(consts.primaryWrapper(), consts.coefficientWrapper());
    }
    bool isInteger(const ConcreteConsts& consts) const {
        return evalFraction<std::size_t>(consts).denominator() == 1;
    }
    template<std::integral ValueType>
    ValueType eval(const ConcreteConsts& consts) const {
        return boost::rational_cast<ValueType>(evalFraction<ValueType>(consts.primaryWrapper(), consts.coefficientWrapper()));
    }
    template<std::floating_point ValueType>
    ValueType eval(const ConcreteConsts& consts) const {
        return boost::rational_cast<ValueType>(evalFraction<std::size_t>(consts.primaryWrapper(), consts.coefficientWrapper()));
    }

    // Evaluates with all the consts and take the minimum of all results.
    boost::rational<std::size_t> lowerBoundEst(const BindingContext& ctx) const;
    // Evaluates with all the consts and take the maximum of all results.
    boost::rational<std::size_t> upperBoundEst(const BindingContext& ctx) const;

    Size identity() const;
    static Size Identity(const BindingContext& ctx);

    std::optional<Trait> getTrait() const;
    bool is1() const;
    // Returns whether there are no primary variables.
    bool isLegalCoefficient() const;
    bool isGeneral() const;

    static SizeLimitsUsage GetLimitsUsage(std::span<const PowerType> powers);
    SizeLimitsUsage getLimitsUsage() const;

    Size& operator*=(const Size& other);
    // The product of two Size's
    Size operator*(const Size& other) const;
    // The product of multiple Size's
    template<SizeRange R>
    Size static Product(R&& operands) {
        auto oi = std::ranges::begin(operands);
        KAS_ASSERT(oi != std::ranges::end(operands));
        auto newSize = Size(*oi);
        auto& newPrimary = newSize.primary;
        auto& newCoefficient = newSize.coefficient;
        const std::size_t primaryCount = newSize.getPrimaryCount();
        const std::size_t coefficientCount = newSize.getCoefficientCount();
        ++oi;
        while (oi != std::ranges::end(operands)) {
            const auto& operand = *oi;
            KAS_ASSERT(primaryCount == operand.getPrimaryCount() && coefficientCount == operand.getCoefficientCount());
            for (std::size_t i = 0; i < primaryCount; ++i) {
                newPrimary[i] += operand.primary[i];
            }
            for (std::size_t i = 0; i < coefficientCount; ++i) {
                newCoefficient[i] += operand.coefficient[i];
            }
            ++oi;
        }
        return newSize;
    }

    Size operator^(PowerType power) const;

    // The quotient of two Size's
    Size operator/(const Size& other) const;
    std::optional<Trait> testDividedBy(const Size& other);
    std::optional<Trait> canBeDividedBy(const Size& other) const;
    bool quotientIsLegal(const Size& other) const;

    Size sqrt() const;

    Size primaryPart() const;
    Size coefficientPart() const;
    Size getAllowanceUsage() const;

    struct EnumerationOptions {
        std::vector<std::size_t> basesIndices;
        ExprType lowerBound;
        ExprType upperBound;
        SizeLimitsUsage limits;
        // Compute baseIndices.
        EnumerationOptions(const ExprType& lowerBound, const ExprType& upperBound, const SizeLimitsUsage& limits);
        // Get the starting point, i.e., the lower bound.
        ExprType begin() const;
        // Checks if the given powers are within maxVarsInSize and maxVarsPowersInSize.
        bool isValid(const ExprType& powers) const;
    };
    // We frequently need to sample Sizes.
    // We would like to enumerate all the possible sizes, in a fashion similar to the way we increment binary numbers.
    // For example, there are 5 variables in total. We would like to enumerate only certain variables, then a possible combination is
    // basesIndices = { 1, 2, 4 }
    // lowerBound = { 0, 0, 0, 0, 0 }
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
    // Moreover, the results are contrained by maxVarsInSize and maxVarsPowersInSize.
    // If and only if this is successful, i.e., we have not reached upperBound, return true.
    static bool EnumerateNext(ExprType& powers, const EnumerationOptions& options);
    // Excluding 1, including lowerBound and upperBound.
    static Generator<Size> EnumerateSizes(const BindingContext& ctx, Size lowerBound, Size upperBound);

    bool operator==(const Size& other) const;
    static std::strong_ordering LexicographicalCompare(const Size& lhs, const Size& rhs);
    static bool LexicographicalLEQ(const Size& lhs, const Size& rhs) {
        return LexicographicalCompare(lhs, rhs) != std::strong_ordering::greater;
    }

    std::string toString(const BindingContext& ctx) const;

    // FOR DEBUG USAGE ONLY!
    std::string debugToString() const;

    template<typename C = decltype([](const std::string&){})>
    static std::vector<std::string> parseNames(std::string_view shape) {
        auto parsedShape = Parser(shape).parseShape();
        std::vector<std::string> result;
        for (auto& size: parsedShape) {
            KAS_ASSERT(size.size() == 1 && size[0].second == 1);
            std::string name = std::move(size[0].first);
            result.emplace_back(std::move(name));
        }
        return result;
    }
};

class PaddingSolver {
    using Power = int;
    using Prime = int;
    // Support coefficient up to 30.
    static constexpr std::array<int, 10> Primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };

    const BindingContext& ctx;
    const ConcreteConsts& consts;

    // For each primary variable, there are some divisors.
    using Divisors = std::map<Prime, Power>;
    // A variable must pad to a multiple of all its divisors.
    std::vector<Divisors> determinedPaddings;

    // For example, if for prime factor p, lhs = [(0, 1), (3, 2)], and rhs = 5, then p^5 divides x_0^1 * x_3^2.
    struct PrimeFactorInequality {
        // The (primary_id, power) list.
        using LHS = std::vector<std::pair<std::size_t, Power>>;
        LHS lhs;
        // The required power.
        Power rhs;
    };
    // For each prime, there are some inequalities.
    std::map<Prime, std::vector<PrimeFactorInequality>> inequalities;

    // Evaluates only the coefficients, and return a fraction.
    boost::rational<int> evalFractionalCoefficient(const Size& size) const;

    // Look up the determinedPaddings and find the value of a LHS.
    Power lhsLowerBound(Prime prime, const PrimeFactorInequality::LHS& lhs);
    int estimateDeterminedPadding(std::size_t primaryIndex);
    void addSingleTermInequality(Prime prime, std::size_t indexPrimary, Power powerPrimary, Power powerPrime);
    void addMultiTermInequality(Prime prime, PrimeFactorInequality::LHS&& lhs, Power rhs);

public:
    PaddingSolver(const BindingContext& ctx, const ConcreteConsts& consts);

    // Add inequalities without which the size is not an integer.
    void addConstraint(const Size& size);

    // Solve all the inequalities and return the result.
    ConcreteConsts solve(const Size& inputSize, const Size& outputSize);
};

struct Allowance {
    const BindingContext& ctx;
    bool countSharedCoefficientsAsAllowanceUsage;
    Size::ExprType primaryAllowance;
    Size::ExprType coefficientAllowance;
    Allowance(const BindingContext& ctx, const Size& currentUsage, bool countSharedCoefficientsAsAllowanceUsage);
    // Counts primary vars, and optionally coefficient vars.
    bool shareWithinAllowance(const Size& size) const;
    // Excludes 1.
    Generator<Size> enumerateSizes() const;
    // Return divisors of this Size. Guarantee that for all consts, the divisor and the quotient are realizable, and not equal to 1 or this.
    Generator<Size> enumerateDivisors(Size size) const;
};

} // namespace kas

template<>
struct std::hash<kas::Size> {
    std::size_t operator()(const kas::Size& size) const noexcept {
        constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
        using namespace std::string_view_literals;
        static const auto sizeHash = std::hash<std::string_view>{}("Size"sv);
        static const auto primaryHash = std::hash<std::string_view>{}("PrimaryVariable"sv);
        static const auto coefficientHash = std::hash<std::string_view>{}("CoefficientVariable"sv);
        auto h = sizeHash;
        std::span<const kas::Size::PowerType> ps = size.getPrimary(), cs = size.getCoefficient();
        const std::size_t pc = ps.size(), cc = cs.size();
        kas::HashCombine(h, pc);
        for (std::size_t i = 0; i < pc; ++i) {
            if (ps[i] != 0) {
                kas::HashCombine(h, std::rotl(primaryHash, SizeTypeWidth / 2 + static_cast<int>(i)));
                kas::HashCombine(h, std::rotl<std::size_t>(1, SizeTypeWidth / 2 + ps[i]));
            }
        }
        kas::HashCombine(h, cc);
        for (std::size_t i = 0; i < cc; ++i) {
            if (cs[i] != 0) {
                kas::HashCombine(h, std::rotl(coefficientHash, static_cast<int>(i)));
                kas::HashCombine(h, std::rotl<std::size_t>(1, SizeTypeWidth / 2 + cs[i]));
            }
        }
        return h;
    }
};
