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
    friend class BindingContext;
    friend struct LabeledSize;
    friend class PaddingSolver;
    friend class HalideGen;

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
    const std::size_t primaryCount;
    const std::size_t coefficientCount;
    // Powers of large variables. Must be non-negative.
    ExprType primary;
    // Powers of small variables. Can be negative (when in denominator).
    ExprType coefficient;

    template<typename ValueType, typename Consts>
    static std::pair<ValueType, ValueType> EvalFraction(std::size_t cnt, Consts&& consts, const Size::ExprType& powers) {
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
        return { nominator, denominator };
    };

public:
    Size(std::size_t primaryCount, std::size_t coefficientCount);
    template<typename Tp, typename Tc>
    Size(std::size_t primaryCount, std::size_t coefficientCount, Tp&& primary, Tc&& coefficient):
        primaryCount { primaryCount },
        coefficientCount { coefficientCount },
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

    template<typename ValueType, typename Tp, typename Tc>
    std::pair<ValueType, ValueType> evalFraction(Tp&& p, Tc&& c) const {
        auto [nP, dP] = EvalFraction<ValueType>(primaryCount, std::forward<Tp>(p), primary);
        auto [nC, dC] = EvalFraction<ValueType>(coefficientCount, std::forward<Tc>(c), coefficient);
        return { nP * nC, dP * dC };
    };
    template<typename ValueType, typename Tp, typename Tc>
    ValueType eval(Tp&& p, Tc&& c) const {
        auto [n, d] = evalFraction<ValueType>(std::forward<Tp>(p), std::forward<Tc>(c));
        return n / d;
    };
    template<typename ValueType>
    std::pair<ValueType, ValueType> evalFraction(const ConcreteConsts& consts) const {
        return evalFraction<ValueType>(consts.primaryWrapper(), consts.coefficientWrapper());
    }
    bool isInteger(const ConcreteConsts& consts) const {
        auto [nom, den] = evalFraction<std::size_t>(consts);
        return nom % den == 0;
    }
    template<typename ValueType>
    ValueType eval(const ConcreteConsts& consts) const {
        return eval<ValueType>(consts.primaryWrapper(), consts.coefficientWrapper());
    }

    // Evaluates with all the consts and take the minimum of all results.
    float lowerBoundEst(const BindingContext& ctx) const;
    // Evaluates with all the consts and take the maximum of all results.
    float upperBoundEst(const BindingContext& ctx) const;

    Size identity() const;
    static Size Identity(const BindingContext& ctx);

    std::optional<Trait> getTrait() const;
    bool is1() const;
    // Returns whether there are no primary variables.
    bool isLegalCoefficient() const;
    bool isGeneral() const;

    int getPrimaryPowersSum() const;

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
        const auto primaryCount = newSize.primaryCount;
        const auto coefficientCount = newSize.coefficientCount;
        ++oi;
        while (oi != std::ranges::end(operands)) {
            const auto& operand = *oi;
            KAS_ASSERT(primaryCount == operand.primaryCount && coefficientCount == operand.coefficientCount);
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

    // Return divisors of this Size. Guarantee that for all consts, the divisor is realizable, and not equal to 1 or this.
    Generator<Size> sampleDivisors(const BindingContext& ctx) const;

    // Excludes 1, including lowerBound and upperBound.
    static Generator<Size> EnumerateSizes(const BindingContext& ctx, Size lowerBound, Size upperBound);

    bool operator==(const Size& other) const;
    static bool LexicographicalLEQ(const Size& lhs, const Size& rhs);

    std::string toString(const BindingContext& ctx) const;

    // FOR DEBUG USAGE ONLY!
    std::string debugToString() const;

    template<typename C = decltype([](const std::string&){})>
    static std::vector<std::string> parseNames(std::string_view shape, C&& onNewName = C()) {
        auto parsedShape = Parser(shape).parseShape();
        std::vector<std::string> result;
        for (auto& size: parsedShape) {
            KAS_ASSERT(size.size() == 1 && size[0].second == 1);
            std::string name = std::move(size[0].first);
            onNewName(name);
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
    std::pair<int, int> evalFractionalCoefficient(const Size& size) const;

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

struct LabeledSize: public Size {
    Trait trait;

public:
    LabeledSize(std::size_t primaryCount, std::size_t coefficientCount);
    LabeledSize(const Size& size);

    LabeledSize identity() const;

    bool is1() const;
    bool isLegalCoefficient() const;
    bool isIllegalCoefficient() const;
    bool isIndeterminedCoefficient() const;
    bool isGeneral() const;

    bool testDividedBy(const Size& other);
    LabeledSize& operator*=(const LabeledSize& other);
    LabeledSize operator*(const LabeledSize& other) const;
};

struct Allowance {
    Size::ExprType primary;
    Size::ExprType coefficientLower;
    Size::ExprType coefficientUpper;
    Allowance(const Size& shape, const BindingContext& ctx);
    bool withinAllowance(const Size& size) const;
    Generator<Size> enumerateSizes(const BindingContext& ctx) const;
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
