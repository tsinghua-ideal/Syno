#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <memory>
#include <optional>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

struct Size {
    friend class BindingContext;
    friend class LabeledSize;
    friend class HalideGen;

public:
    constexpr static std::size_t MAX_VARIABLES = 16;
    using ExprType = std::array<std::int8_t, MAX_VARIABLES>;
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
    {}
    Size& operator=(const Size& other) &;

    template<typename ValueType, typename Tp, typename Tc>
    ValueType eval(Tp&& p, Tc&& c) const {
        auto factor = [](std::size_t cnt, auto&& f, const Size::ExprType& powers) -> std::pair<ValueType, ValueType> {
            ValueType nominator = 1;
            ValueType denominator = 1;
            for (std::size_t i = 0; i < cnt; ++i) {
                if (powers[i] > 0) {
                    const ValueType v = f(i);
                    for (std::size_t j = 0; j < powers[i]; ++j) {
                        nominator *= v;
                    }
                } else if (powers[i] < 0) {
                    const ValueType v = f(i);
                    for (std::size_t j = 0; j < -powers[i]; ++j) {
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

    std::vector<std::shared_ptr<Size>> sampleFactors(const BindingContext& ctx) const;

    // The product of two Size's
    std::shared_ptr<Size> operator*(const Size& other) const;
    // The product of multiple Size's
    std::shared_ptr<Size> static Product(const std::vector<std::shared_ptr<Size>>& operands);

    // The quotient of two Size's
    std::shared_ptr<Size> operator/(const Size& other) const;
    std::optional<Trait> testDividedBy(const Size& other);

    bool operator==(const Size& other) const = default;

    std::size_t estimate(const BindingContext& ctx) const;

    std::string toString(const BindingContext& ctx) const;
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

struct Shape final {
protected:
    std::vector<std::shared_ptr<Size>> sizes;

public:
    Shape() = default;
    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) = default;
    Shape& operator=(const Shape& shape) & = default;
    Shape& operator=(Shape&& shape) & = default;
    explicit Shape(const std::vector<std::shared_ptr<Size>>& sizes);
    explicit Shape(std::vector<std::shared_ptr<Size>>&& sizes);

    const std::vector<std::shared_ptr<Size>>& getSizes() const;
    bool operator==(const Shape& other) const = default;
    std::size_t size() const;
    const std::shared_ptr<Size>& operator[](std::size_t index) const;

    template<std::size_t N>
    std::array<Shape, N> cut(std::array<std::size_t, N> dimensions) const {
        std::array<Shape, N> res;
        std::size_t begin = 0;
        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            std::size_t dim = dimensions[i];
            const std::size_t end = begin + dim;
            std::vector<std::shared_ptr<Size>> newSizes;
            std::copy(sizes.begin() + begin, sizes.begin() + end, std::back_inserter(newSizes));
            res[i] = Shape(std::move(newSizes));
            begin = end;
        }
        KAS_ASSERT(begin == size());
        return res;
    }

    Shape replace(
        std::vector<std::size_t> drops,
        std::vector<std::pair<std::size_t, std::shared_ptr<Size>>> adds
    ) const;

    std::vector<std::size_t> estimate(const BindingContext& ctx) const;

    std::string toString(const BindingContext& ctx) const;

    static std::vector<std::string> parseNames(std::string_view shape);
};

} // namespace kas
