#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <optional>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

struct Size final {
public:
    constexpr static std::size_t MAX_VARIABLES = 16;
    using ExprType = std::array<std::int8_t, MAX_VARIABLES>;
protected:
    // Powers of large variables. Must be non-negative.
    ExprType primary;
    // Powers of small variables. Can be negative (when in denominator).
    ExprType coefficient;
    // Returns whether the coefficients cannot be possibly realized. This excludes some sizes in forms like 1/K.
    static bool isCoefficientRealizable(const ExprType& toBeRealized, const BindingContext& ctx);
    friend class BindingContext;

public:
    Size();
    template<typename Tp, typename Tc>
    Size(Tp&& primary, Tc&& coefficient):
        primary { std::forward<Tp>(primary) },
        coefficient { std::forward<Tc>(coefficient) }
    {}

    bool is1() const;
    // Returns whether there are no primary variables.
    bool isCoefficient() const;
    // Returns whether the powers of all primary variables are greater than equal to that of the input.
    bool isMultipleOf(const Size& factor, const BindingContext& ctx) const;

    std::vector<std::shared_ptr<Size>> sampleFactors(const BindingContext& ctx) const;

    // The product of two Size's
    std::shared_ptr<Size> operator*(const Size& other) const;
    // The product of multiple Size's
    std::shared_ptr<Size> static Product(const std::vector<std::shared_ptr<Size>>& operands);

    // The quotient of two Size's
    std::shared_ptr<Size> operator/(const Size& other) const;

    bool operator==(const Size& other) const;

    std::string toString(const BindingContext& ctx) const;
};

struct Shape final {
protected:
    std::vector<std::shared_ptr<Size>> sizes;

public:
    Shape() = default;
    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) = default;
    Shape& operator=(const Shape& shape) = default;
    Shape& operator=(Shape&& shape) = default;
    explicit Shape(const std::vector<std::shared_ptr<Size>>& sizes);
    explicit Shape(std::vector<std::shared_ptr<Size>>&& sizes);

    const std::vector<std::shared_ptr<Size>>& getSizes() const;
    std::size_t size() const;
    const std::shared_ptr<Size>& operator[](std::size_t index) const;

    // drops and adds must be sorted by index
    Shape replace(
        std::vector<int> drops,
        std::vector<std::pair<int, std::shared_ptr<Size>>> adds
    ) const;

    std::vector<int> findSize(const Size& size) const;
    // Find the indices of sizes that are divisible by given factor.
    std::vector<int> findMultipleOfSize(const Size& factor, const BindingContext& ctx) const;

    std::string toString(const BindingContext& ctx) const;

    static std::vector<std::string> parseNames(std::string_view shape);
};

} // namespace kas
