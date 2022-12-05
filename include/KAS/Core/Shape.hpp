#pragma once

#include <string>
#include <vector>
#include <memory>


namespace kas {

class BindingContext;

struct Size {
public:
    // Powers of large variables. Must be non-negative.
    const std::vector<int> primary;
    // Powers of small variables. Can be negative (when in denominator).
    const std::vector<int> coefficient;

    Size() = delete;
    template<typename Tp, typename Tc>
    Size(Tp&& primary, Tc&& coefficient):
        primary { std::forward<Tp>(primary) },
        coefficient { std::forward<Tc>(coefficient) }
    {}

    // Returns whether there are no primary variables.
    bool isCoefficient() const;

    // The product of two Size's
    std::shared_ptr<Size> operator*(const Size& other) const;

    // The quotient of two Size's
    std::shared_ptr<Size> operator/(const Size& other) const;

    bool operator==(const Size& other) const;

    std::string toString(const BindingContext& ctx) const;
};

class BindingContext {
public:
    // Metadata includes aliases, whether preferred by specific ops (TODO), which context a variable is in (when there are multiple contexts, required by Blending) (TODO), etc...
    struct Metadata {
        std::string alias;
        Metadata() = default;
        Metadata(const std::string& alias);
    };
    // The varaibles are the indices. Metadata can be accessed by index.
    std::vector<Metadata> primaryMetadata;
    std::vector<Metadata> coefficientMetadata;

    BindingContext(int countPrimary, int countCoefficient);
    template<typename Tp, typename Tc>
    BindingContext(Tp&& primaryMetadata, Tc&& coefficientMetadata):
        primaryMetadata { std::forward<Tp>(primaryMetadata) },
        coefficientMetadata { std::forward<Tc>(coefficientMetadata) }
    {}

    std::shared_ptr<Size> getSinglePrimaryVariableSize(int index) const;
    std::shared_ptr<Size> getSingleCoefficientVariableSize(int index) const;
};

struct Shape {
public:
    const std::vector<std::shared_ptr<Size>> sizes;

    Shape() = delete;
    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) = default;
    explicit Shape(const std::vector<std::shared_ptr<Size>>& sizes);
    explicit Shape(std::vector<std::shared_ptr<Size>>&& sizes);

    size_t size() const;
    const std::shared_ptr<Size>& operator[](size_t index) const;

    // drops and adds must be sorted by index
    Shape replace(
        const std::vector<int>& drops,
        const std::vector<std::pair<int, std::shared_ptr<Size>>>& adds
    ) const;

    std::string toString(const BindingContext& ctx) const;
};

} // namespace kas
