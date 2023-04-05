#pragma once

#include <array>
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
#include "KAS/Utils/Hash.hpp"


namespace kas {

struct Size {
    friend class BindingContext;
    friend struct LabeledSize;
    friend class HalideGen;

public:
    constexpr static std::size_t MAX_VARIABLES = 16;
    using PowerType = std::int8_t;
    using ExprType = std::array<PowerType, MAX_VARIABLES>;
    enum class Trait {
        One, // powers of primary == 0, powers of coefficient == 0.
        Coefficient, // powers of primary == 0, powers of coefficient >= 0 but not all 0.
        IllegalCoefficient, // powers of primary == 0, powers of coefficient < 0.
        General, // powers of primary >= 0 but not all 0.
    };

protected:
    const std::size_t primaryCount;
    const std::size_t coefficientCount;
    // Powers of large variables. Must be non-negative.
    ExprType primary;
    // Powers of small variables. Can be negative (when in denominator).
    ExprType coefficient;

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
    ValueType eval(Tp&& p, Tc&& c) const {
        auto factor = [](std::size_t cnt, auto&& f, const Size::ExprType& powers) -> std::pair<ValueType, ValueType> {
            ValueType nominator = 1;
            ValueType denominator = 1;
            for (std::size_t i = 0; i < cnt; ++i) {
                if (powers[i] > 0) {
                    const ValueType v = f(i);
                    auto power = static_cast<std::size_t>(powers[i]);
                    for (std::size_t j = 0; j < power; ++j) {
                        nominator *= v;
                    }
                } else if (powers[i] < 0) {
                    const ValueType v = f(i);
                    auto power = static_cast<std::size_t>(-powers[i]);
                    for (std::size_t j = 0; j < power; ++j) {
                        denominator *= v;
                    }
                }
            }
            return { nominator, denominator };
        };
        auto [nP, dP] = factor(primaryCount, std::forward<Tp>(p), primary);
        auto [nC, dC] = factor(coefficientCount, std::forward<Tc>(c), coefficient);
        return nP * nC / dP / dC;
    };

    Size identity() const;

    Trait getTrait() const;
    bool is1() const;
    // Returns whether there are no primary variables.
    bool isLegalCoefficient() const;
    bool isGeneral() const;

    int getPrimaryPowersSum() const;

    // The product of two Size's
    Size operator*(const Size& other) const;
    // The product of multiple Size's
    template<typename Storage, auto Mapping>
    Size static Product(AbstractShape<Storage, Mapping> operands) {
        KAS_ASSERT(operands.size() > 0);
        auto newSize = Size(operands[0]);
        auto& newPrimary = newSize.primary;
        auto& newCoefficient = newSize.coefficient;
        const auto primaryCount = newSize.primaryCount;
        const auto coefficientCount = newSize.coefficientCount;
        for (std::size_t index = 1; index < operands.size(); ++index) {
            const auto& operand = operands[index];
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

    // The quotient of two Size's
    Size operator/(const Size& other) const;
    std::optional<Trait> testDividedBy(const Size& other);
    std::optional<Trait> canBeDividedBy(const Size& other) const;

    bool operator==(const Size& other) const;

    std::string toString(const BindingContext& ctx) const;

    // FOR DEBUG USAGE ONLY!
    inline std::string toString() const {
        if (BindingContext::PublicCtx) {
            return toString(*BindingContext::PublicCtx);
        } else {
            return "NO_PUBLIC_CONTEXT";
        }
    }

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

struct LabeledSize: public Size {
    friend class FinalizeShapeOp;

protected:
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

    // Assumes that both are coefficients. In effect performs multiplication, and checks if the power of any variable in the denominator decreases. If so, returns the result, otherwise returns nullopt.
    std::optional<LabeledSize> absorbCoefficientNumeratorToDenominator(const LabeledSize& other) const;

    // Assumes this is IllegalCoefficient and other is General. Computes how much the variables can counteract.
    int scoreOfGeneralDimension(const LabeledSize& other) const;
};

struct Allowance {
    Size::ExprType primary;
    Size::ExprType coefficientLower;
    Size::ExprType coefficientUpper;
    Allowance(const Size& shape, const BindingContext& ctx);
    bool withinAllowance(const Size& size) const;
};

} // namespace kas

template<typename T>
struct std::hash<std::span<T>> {
    std::size_t operator()(const std::span<T>& span) const noexcept {
        std::size_t seed = span.size();
        for (const auto& item: span) {
            kas::HashCombine(seed, item);
        }
        return seed;
    }
};

template<>
struct std::hash<kas::Size> {
    std::size_t operator()(const kas::Size& size) const noexcept {
        auto h = std::hash<std::string>{}("Size");
        kas::HashCombine(h, std::hash<std::span<const kas::Size::PowerType>>{}(size.getPrimary()));
        kas::HashCombine(h, std::hash<std::span<const kas::Size::PowerType>>{}(size.getCoefficient()));
        return h;
    }
};
