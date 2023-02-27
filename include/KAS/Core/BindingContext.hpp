#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <vector>
#include <memory>

#include "KAS/Core/Parser.hpp"


namespace kas {

struct Size;
template<typename Storage, auto Mapping> class AbstractShape;
inline const Size& ShapeMapping(const Size& size) { return size; };
using Shape = AbstractShape<std::vector<Size>, ShapeMapping>;

struct ConcreteConsts {
    std::vector<int> primary;
    std::vector<int> coefficient;
    inline auto primaryWrapper() const {
        return [this](std::size_t i) { return primary[i]; };
    }
    inline auto coefficientWrapper() const {
        return [this](std::size_t i) { return coefficient[i]; };
    }
};

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

    using LookUpTable = std::map<std::string, std::size_t>;
    LookUpTable getPrimaryLookupTable() const;
    LookUpTable getCoefficientLookupTable() const;
    Size lookUp(const std::string& name, const LookUpTable& primaryTable, const LookUpTable& coefficientTable) const;

public:
    BindingContext(std::size_t countPrimary, std::size_t countCoefficient);
    template<typename Tp, typename Tc>
    requires(std::is_same_v<std::remove_cvref_t<Tp>, std::vector<Metadata>> && std::is_same_v<std::remove_cvref_t<Tc>, std::vector<Metadata>>)
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
    Size get(const std::string& name) const;
    template<typename... Args>
    auto getSizes(Args&&... args) const -> std::array<Size, sizeof...(Args)> {
        std::map<std::string, std::size_t> pNameToIndex = getPrimaryLookupTable();
        std::map<std::string, std::size_t> cNameToIndex = getCoefficientLookupTable();
        return std::array { lookUp(std::forward<Args>(args), pNameToIndex, cNameToIndex)... };
    }

    Shape getShapeFromNames(const std::vector<std::string>& names) const;
    // This overwrites the current metadata.
    void applySpecs(std::vector<std::pair<std::string, Parser::PureSpec>>& primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>>& coefficientSpecs);

    ConcreteConsts realizeConsts(const std::map<std::string, std::size_t>& mappings) const;
};

} // namespace kas
