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

    // The product of two Size's
    Size operator*(const Size& other) const;

    bool operator==(const Size& other) const;

    std::string toString(const BindingContext& ctx) const;
};

class BindingContext {
public:
    // Metadata includes aliases, whether preferred by specific ops (TODO), etc..
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
    std::vector<std::shared_ptr<Size>> sizes;

    Shape() = delete;
    template<typename T>
    Shape(T&& sizes):
        sizes { std::forward<T>(sizes) }
    {}
};

} // namespace kas
