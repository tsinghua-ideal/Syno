#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <memory>


namespace kas {

struct Shape;
struct Size;

class BindingContext final {
public:
    // Metadata includes aliases, whether preferred by specific ops (TODO), which context a variable is in (when there are multiple contexts, required by Blending) (TODO), etc...
    struct Metadata {
        std::string alias;
        Metadata() = default;
        Metadata(std::string_view alias);
    };

protected:
    std::size_t namedPrimaryCount;
    // The varaibles are the indices. Metadata can be accessed by index.
    std::vector<Metadata> primaryMetadata;
    std::vector<Metadata> coefficientMetadata;

public:
    BindingContext(std::size_t countPrimary, std::size_t countCoefficient);
    template<typename Tp, typename Tc>
    BindingContext(Tp&& primaryMetadata, Tc&& coefficientMetadata):
        primaryMetadata { std::forward<Tp>(primaryMetadata) },
        coefficientMetadata { std::forward<Tc>(coefficientMetadata) }
    {
        namedPrimaryCount = this->primaryMetadata.size();
    }

    std::size_t getPrimaryCount() const;
    std::size_t getCoefficientCount() const;
    std::string_view getPrimaryAlias(std::size_t index) const;
    std::string_view getCoefficientAlias(std::size_t index) const;

    std::shared_ptr<Size> getSinglePrimaryVariableSize(std::size_t index) const;
    std::shared_ptr<Size> getSingleCoefficientVariableSize(std::size_t index) const;

    std::vector<std::shared_ptr<Size>> getPositiveCoefficients() const;

    Shape getShapeFromNames(const std::vector<std::string>& names);
};

} // namespace kas
