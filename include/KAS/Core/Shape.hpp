#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>


namespace kas {

class BindingContext;
class Shape;

class Size final {
protected:
    // Powers of large variables. Must be non-negative.
    const std::vector<int> primary;
    // Powers of small variables. Can be negative (when in denominator).
    const std::vector<int> coefficient;
    // Returns whether the coefficients cannot be possibly realized. This excludes some sizes in forms like 1/K.
    static bool isCoefficientRealizable(const std::vector<int>& toBeRealized, const BindingContext& ctx);

public:
    Size() = delete;
    template<typename Tp, typename Tc>
    Size(Tp&& primary, Tc&& coefficient):
        primary { std::forward<Tp>(primary) },
        coefficient { std::forward<Tc>(coefficient) }
    {}

    // Returns whether there are no primary variables.
    bool isCoefficient() const;
    // Returns whether the powers of all primary variables are greater than equal to that of the input.
    bool isMultipleOf(const Size& factor, const BindingContext& ctx) const;

    std::vector<std::shared_ptr<Size>> sampleFactors(const BindingContext& ctx) const;

    // The product of two Size's
    std::shared_ptr<Size> operator*(const Size& other) const;

    // The quotient of two Size's
    std::shared_ptr<Size> operator/(const Size& other) const;

    bool operator==(const Size& other) const;

    std::string toString(const BindingContext& ctx) const;
};

class BindingContext final {
public:
    // Metadata includes aliases, whether preferred by specific ops (TODO), which context a variable is in (when there are multiple contexts, required by Blending) (TODO), etc...
    struct Metadata {
        std::string alias;
        Metadata() = default;
        Metadata(const std::string& alias);
    };

protected:
    int namedPrimaryCount;
    // The varaibles are the indices. Metadata can be accessed by index.
    std::vector<Metadata> primaryMetadata;
    std::vector<Metadata> coefficientMetadata;

public:
    BindingContext(int countPrimary, int countCoefficient);
    template<typename Tp, typename Tc>
    BindingContext(Tp&& primaryMetadata, Tc&& coefficientMetadata):
        primaryMetadata { std::forward<Tp>(primaryMetadata) },
        coefficientMetadata { std::forward<Tc>(coefficientMetadata) }
    {
        namedPrimaryCount = primaryMetadata.size();
    }

    size_t getPrimaryCount() const;
    size_t getCoefficientCount() const;
    std::string_view getPrimaryAlias(size_t index) const;
    std::string_view getCoefficientAlias(size_t index) const;

    std::shared_ptr<Size> getSinglePrimaryVariableSize(int index) const;
    std::shared_ptr<Size> getSingleCoefficientVariableSize(int index) const;

    std::vector<std::shared_ptr<Size>> getPositiveCoefficients() const;

    Shape getShapeFromNames(const std::vector<std::string>& names);
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

    size_t size() const;
    const std::shared_ptr<Size>& operator[](size_t index) const;

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
