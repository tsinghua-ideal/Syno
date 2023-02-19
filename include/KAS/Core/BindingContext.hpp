#pragma once

#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <vector>
#include <memory>

#include "KAS/Core/Parser.hpp"


namespace kas {

class Shape;
struct Size;

class BindingContext final {
    friend class HalideGen;

public:
    // Metadata includes aliases, whether preferred by specific ops (TODO), which context a variable is in (when there are multiple contexts, required by Blending) (TODO), etc...
    struct Metadata {
        std::string alias = "D";
        bool isOdd = false;
        std::size_t maximumOccurrence = 3;
        std::size_t estimate = 128;
    };

protected:
    // The varaibles are the indices. Metadata can be accessed by index.
    std::vector<Metadata> primaryMetadata;
    std::vector<Metadata> coefficientMetadata;

    std::map<std::string, std::size_t> getPrimaryLookupTable() const;
    std::map<std::string, std::size_t> getCoefficientLookupTable() const;

public:
    BindingContext(std::size_t countPrimary, std::size_t countCoefficient);
    template<typename Tp, typename Tc>
    BindingContext(Tp&& primaryMetadata, Tc&& coefficientMetadata):
        primaryMetadata { std::forward<Tp>(primaryMetadata) },
        coefficientMetadata { std::forward<Tc>(coefficientMetadata) }
    {}

    std::size_t getPrimaryCount() const;
    std::size_t getCoefficientCount() const;
    std::span<const Metadata> getPrimaryMetadata() const;
    std::span<const Metadata> getCoefficientMetadata() const;
    std::string getPrimaryAlias(std::size_t index) const;
    std::string getCoefficientAlias(std::size_t index) const;
    std::size_t getPrimaryEstimate(std::size_t index) const;
    std::size_t getCoefficientEstimate(std::size_t index) const;

    Size getSinglePrimaryVariableSize(std::size_t index) const;
    Size getSingleCoefficientVariableSize(std::size_t index) const;

    Shape getShapeFromNames(const std::vector<std::string>& names);
    // This overwrites the current metadata.
    void applySpecs(std::vector<std::pair<std::string, Parser::PureSpec>>& primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>>& coefficientSpecs);
    // Change the estimation of the values.
    void applyEstimates(const std::map<std::string, std::size_t>& estimates);
    // Get the arguments for calling.
    std::vector<std::size_t> getKernelArguments(const std::map<std::string, std::size_t>& mappings) const;
    std::vector<std::size_t> evaluateShape(const Shape& shape, const std::map<std::string, std::size_t>& mappings) const;
};

} // namespace kas
