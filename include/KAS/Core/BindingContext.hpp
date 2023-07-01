#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <vector>
#include <memory>

#include "KAS/Core/Parser.hpp"


namespace kas {

struct Size;
template<typename Storage, auto Mapping> class AbstractShape;
using Shape = AbstractShape<std::vector<Size>, std::identity{}>;
class BindingContext;

struct ConcreteConsts {
    std::vector<int> primary;
    std::vector<int> coefficient;
    auto primaryWrapper() const {
        return [this](std::size_t i) { return primary[i]; };
    }
    auto coefficientWrapper() const {
        return [this](std::size_t i) { return coefficient[i]; };
    }
    bool operator==(const ConcreteConsts& rhs) const = default;
};

struct PaddedConsts {
    ConcreteConsts unpadded;
    ConcreteConsts padded;
    std::string toString(const BindingContext& ctx) const;
};

class BindingContext final {
public:
    struct Metadata {
        std::string alias = "D";
        bool isOdd = false;
        std::size_t maximumOccurrence = 3;
        std::optional<std::size_t> estimate = std::nullopt;
    };

protected:
    // The varaibles are the indices. Metadata can be accessed by index.
    std::vector<Metadata> primaryMetadata;
    std::vector<Metadata> coefficientMetadata;

    std::size_t maximumVariablesInSize = std::numeric_limits<std::size_t>::max();
    std::size_t maximumVariablesPowersInSize = std::numeric_limits<std::size_t>::max();

    ConcreteConsts defaultConsts;
    std::vector<ConcreteConsts> allConsts;

    using LookUpTable = std::map<std::string, std::size_t>;
    LookUpTable primaryLookUpTable, coefficientLookUpTable;
    void updateLookUpTables();
    Size getSizeFromFactors(const std::vector<Parser::Factor>& factors) const;

public:
    BindingContext() = default;
    BindingContext(std::size_t countPrimary, std::size_t countCoefficient);
    BindingContext(std::vector<Metadata> primaryMetadata, std::vector<Metadata> coefficientMetadata);

    std::size_t getPrimaryCount() const;
    std::size_t getCoefficientCount() const;
    std::span<const Metadata> getPrimaryMetadata() const;
    std::span<const Metadata> getCoefficientMetadata() const;
    std::string getPrimaryAlias(std::size_t index) const;
    std::string getCoefficientAlias(std::size_t index) const;

    Size getSinglePrimaryVariableSize(std::size_t index) const;
    Size getSingleCoefficientVariableSize(std::size_t index) const;
    Size getSize(const std::string& name) const;
    template<typename... Args>
    requires std::conjunction_v<std::is_convertible<Args, std::string>...>
    auto getSizes(Args&&... args) const -> std::array<Size, sizeof...(Args)> {
        return std::array { getSize(std::forward<Args>(args))... };
    }

    void setMaxVariablesInSize(std::size_t maximumVariablesInSize);
    std::size_t getMaxVariablesInSize() const;
    void setMaxVariablesPowersInSize(std::size_t maximumVariablesPowersInSize);
    std::size_t getMaxVariablesPowersInSize() const;
    bool isSizeValid(const Size& size) const;

    std::vector<Size> getSizes(const std::vector<std::string>& names) const;
    Shape getShape(const std::vector<std::string>& names) const;
    // This overwrites the current metadata.
    void applySpecs(std::vector<std::pair<std::string, Parser::PureSpec>>& primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>>& coefficientSpecs);

    ConcreteConsts realizeConsts(const std::map<std::string, std::size_t>& mappings, bool defaultFallback = false) const;
    // This overwrites the current allConsts.
    void applyMappings(const std::vector<std::map<std::string, std::size_t>>& allMappings);
    const std::vector<ConcreteConsts>& getAllConsts() const { return allConsts; }
    const ConcreteConsts& getDefaultConsts() const { return defaultConsts; }

    // FOR DEBUG USAGE ONLY!
    static inline const BindingContext *DebugPublicCtx = nullptr;
    template<typename F>
    static std::string ApplyDebugPublicCtx(F&& f, auto&& self) {
        if (BindingContext::DebugPublicCtx) {
            return std::invoke(std::forward<F>(f), std::forward<decltype(self)>(self), *DebugPublicCtx);
        } else {
            return "NO_PUBLIC_CONTEXT";
        }
    }
};

} // namespace kas
