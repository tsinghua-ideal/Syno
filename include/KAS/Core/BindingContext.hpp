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

#include <nlohmann/json.hpp>

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
    std::strong_ordering operator<=>(const ConcreteConsts& rhs) const = default;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ConcreteConsts, primary, coefficient)

struct PaddedConsts {
    ConcreteConsts unpadded;
    ConcreteConsts padded;
    std::string toString(const BindingContext& ctx) const;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PaddedConsts, unpadded, padded)

struct SizeLimitsUsage;

class BindingContext {
public:
    struct Options {
        std::size_t maximumEnumerationsPerVar = 5;
        std::size_t maximumVariablesInSize = std::numeric_limits<std::size_t>::max();
        std::size_t maximumVariablesPowersInSize = std::numeric_limits<std::size_t>::max();
        bool requiresExactDivision = false;
        void check() const;
    };

private:
    struct Metadata {
        std::string alias = "D";
        std::size_t maximumOccurrence = 3;
        std::optional<std::size_t> estimate = std::nullopt;
    };

    // The varaibles are the indices. Metadata can be accessed by index.
    std::vector<Metadata> primaryMetadata;
    std::vector<Metadata> coefficientMetadata;

    Options options;

    std::vector<ConcreteConsts> allConsts;

    using LookUpTable = std::map<std::string, std::size_t>;
    LookUpTable primaryLookUpTable, coefficientLookUpTable;
    void updateLookUpTables();
    Size getSizeFromFactors(const std::vector<Parser::Factor>& factors) const;

    // This overwrites the current metadata.
    void applySpecs(const ShapeSpecParser::NamedSpecs& primarySpecs, const ShapeSpecParser::NamedSpecs& coefficientSpecs);
    // This overwrites the current allConsts.
    void applyMappings(const std::vector<std::map<std::string, std::size_t>>& allMappings, bool defaultFallback = false);

public:
    // This is not a legal state. You must overwrite it.
    BindingContext() = default;
    // Specify all you want.
    BindingContext(const ShapeSpecParser::NamedSpecs& primarySpecs, const ShapeSpecParser::NamedSpecs& coefficientSpecs, const std::vector<std::map<std::string, std::size_t>>& allMappings, const Options& options);
    // Convenient constructor. Specify the sizes in textual form.
    BindingContext(const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs);
    // Convenient constructor. Name all the variables by default. Provide one consts.
    BindingContext(const std::vector<std::size_t>& primaryEstimates, const std::vector<std::size_t>& coefficientEstimates);
    // Even more convenient constructor, that sets all primary estimates to 128, and all coefficient estimates to 3.
    BindingContext(std::size_t primaryCount, std::size_t coefficientCount);

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

    std::size_t getMaxEnumerationsPerVar() const;
    std::size_t getMaxVariablesInSize() const;
    std::size_t getMaxVariablesPowersInSize() const;
    bool requiresExactDivision() const;
    SizeLimitsUsage getUsageLimits() const;
    bool isUsageWinthinLimits(const SizeLimitsUsage& usage) const;
    bool isUsageWinthinLimits(const Size& size) const;
    bool isSizeLegalToSample(const Size& size) const;
    bool isSizeValid(const Size& size) const;

    std::vector<Size> getSizes(const std::vector<std::string>& names) const;
    Shape getShape(const std::vector<std::string>& names) const;
    std::pair<Shape, std::vector<Parser::Attributes>> getShapeAndAttributes(std::string_view shape) const;

    ConcreteConsts realizeConsts(const std::map<std::string, std::size_t>& mappings, bool defaultFallback = false) const;
    const std::vector<ConcreteConsts>& getAllConsts() const { return allConsts; }

    // FOR DEBUG USAGE ONLY!
    static inline const BindingContext *DebugPublicCtx = nullptr;
    void debug() const { DebugPublicCtx = this; }
    template<typename F>
    static std::string ApplyDebugPublicCtx(F&& f, auto&& self) {
        if (BindingContext::DebugPublicCtx) {
            return std::invoke(std::forward<F>(f), std::forward<decltype(self)>(self), *DebugPublicCtx);
        } else {
            return "NO_PUBLIC_CONTEXT";
        }
    }
};

struct SizeLimitsUsage {
    std::size_t varsInSize = 0;
    std::size_t varsPowersInSize = 0;
    SizeLimitsUsage& operator+=(const SizeLimitsUsage& rhs);
    SizeLimitsUsage operator+(const SizeLimitsUsage& rhs) const;
    SizeLimitsUsage& operator-=(const SizeLimitsUsage& rhs);
    SizeLimitsUsage operator-(const SizeLimitsUsage& rhs) const;
    // Whether within limits.
    bool operator<=(const SizeLimitsUsage& rhs) const;
};

} // namespace kas
