#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Algorithm.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

std::string PaddedConsts::toString(const BindingContext& ctx) const {
    auto formatPart = [](const std::vector<int>& fro, const std::vector<int>& to, auto&& alias) {
        return fmt::format("{}", fmt::join(
            std::views::iota(std::size_t{0}, fro.size())
            | std::views::transform([&](std::size_t i) {
                if (fro[i] == to[i]) {
                    return fmt::format("{}: {}", alias(i), fro[i]);
                } else {
                    return fmt::format("{}: {} -> {}", alias(i), fro[i], to[i]);
                }
            }), ", "
        ));
    };
    return fmt::format("Primary: {{ {} }}, Coefficient {{ {} }}",
        formatPart(unpadded.primary, padded.primary, [&ctx](auto i) { return ctx.getPrimaryAlias(i); }),
        formatPart(unpadded.coefficient, padded.coefficient, [&ctx](auto i) { return ctx.getCoefficientAlias(i); })
    );
}

void BindingContext::updateLookUpTables() {
    auto getLookUpTable = [](const std::vector<Metadata>& metadata) {
        std::map<std::string, std::size_t> lookupTable;
        for (std::size_t i = 0; i < metadata.size(); ++i) {
            lookupTable[metadata[i].alias] = i;
        }
        return lookupTable;
    };
    primaryLookUpTable = getLookUpTable(primaryMetadata);
    coefficientLookUpTable = getLookUpTable(coefficientMetadata);
}

Size BindingContext::getSizeFromFactors(const std::vector<Parser::Factor>& factors) const {
    Size result(getPrimaryCount(), getCoefficientCount());
    for (const auto& [name, power]: factors) {
        if (auto it = primaryLookUpTable.find(name); it != primaryLookUpTable.end())
            result.primary[it->second] += power;
        else if (auto it = coefficientLookUpTable.find(name); it != coefficientLookUpTable.end())
            result.coefficient[it->second] += power;
        else
            KAS_CRITICAL("Unknown variable name: {}", name);
    }
    return result;
}

void BindingContext::applySpecs(const ShapeSpecParser::NamedSpecs& primarySpecs, const ShapeSpecParser::NamedSpecs& coefficientSpecs) {
    primaryMetadata.resize(primarySpecs.size());
    for (std::size_t i = 0; i < primarySpecs.size(); ++i) {
        const auto& [name, spec] = primarySpecs[i];
        primaryMetadata[i] = Metadata {
            .alias = name,
            .maximumOccurrence = spec.maxOccurrences.value_or(3),
            .estimate = spec.size,
        };
    }
    coefficientMetadata.resize(coefficientSpecs.size());
    for (std::size_t i = 0; i < coefficientSpecs.size(); ++i) {
        const auto& [name, spec] = coefficientSpecs[i];
        coefficientMetadata[i] = Metadata {
            .alias = name,
            .maximumOccurrence = spec.maxOccurrences.value_or(3),
            .estimate = spec.size,
        };
    }
    updateLookUpTables();
}

void BindingContext::applyMappings(const std::vector<std::map<std::string, std::size_t>>& allMappings, bool defaultFallback) {
    decltype(allConsts) result;
    for (const auto& mappings: allMappings) {
        result.emplace_back(realizeConsts(mappings, defaultFallback));
    }
    allConsts = std::move(result);
}

BindingContext::BindingContext(const ShapeSpecParser::NamedSpecs& primarySpecs, const ShapeSpecParser::NamedSpecs& coefficientSpecs, const std::vector<std::map<std::string, std::size_t>>& allMappings, const Options& options):
    options(options)
{
    applySpecs(primarySpecs, coefficientSpecs);
    KAS_ASSERT(!allMappings.empty());
    applyMappings(allMappings);
}

BindingContext::BindingContext(const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs) {
    auto parser = ShapeSpecParser(primarySpecs, coefficientSpecs);
    auto [contractedPrimarySpecs, contractedCoefficientSpecs] = parser.build();
    applySpecs(contractedPrimarySpecs, contractedCoefficientSpecs);
    applyMappings({{}}, true);
}


BindingContext::BindingContext(const std::vector<std::size_t>& primaryEstimates, const std::vector<std::size_t>& coefficientEstimates) {
    ShapeSpecParser::NamedSpecs primarySpecs, coefficientSpecs;
    for (std::size_t i = 0; i < primaryEstimates.size(); ++i) {
        primarySpecs.emplace_back("x_" + std::to_string(i), Parser::PureSpec { .size = primaryEstimates[i] });
    }
    for (std::size_t i = 0; i < coefficientEstimates.size(); ++i) {
        coefficientSpecs.emplace_back("c_" + std::to_string(i), Parser::PureSpec { .size = coefficientEstimates[i] });
    }
    applySpecs(primarySpecs, coefficientSpecs);
    applyMappings({{}}, true);
}

BindingContext::BindingContext(std::size_t primaryCount, std::size_t coefficientCount):
    BindingContext(
        std::vector<std::size_t>(primaryCount, 128),
        std::vector<std::size_t>(coefficientCount, 3)
    )
{}

std::size_t BindingContext::getPrimaryCount() const {
    return primaryMetadata.size();
}
std::size_t BindingContext::getCoefficientCount() const {
    return coefficientMetadata.size();
}
std::span<const BindingContext::Metadata> BindingContext::getPrimaryMetadata() const {
    return { primaryMetadata.data(), primaryMetadata.size() };
}
std::span<const BindingContext::Metadata> BindingContext::getCoefficientMetadata() const {
    return { coefficientMetadata.data(), coefficientMetadata.size() };
}
std::string BindingContext::getPrimaryAlias(std::size_t index) const {
    return primaryMetadata.at(index).alias;
}
std::string BindingContext::getCoefficientAlias(std::size_t index) const {
    return coefficientMetadata.at(index).alias;
}

Size BindingContext::getSinglePrimaryVariableSize(std::size_t index) const {
    KAS_ASSERT(index >= 0 && index < getPrimaryCount());
    auto res = Size(getPrimaryCount(), getCoefficientCount());
    res.primary[index] = 1;
    return res;
}

Size BindingContext::getSingleCoefficientVariableSize(std::size_t index) const {
    KAS_ASSERT(index >= 0 && index < getCoefficientCount());
    auto res = Size(getPrimaryCount(), getCoefficientCount());
    res.coefficient[index] = 1;
    return res;
}

Size BindingContext::getSize(const std::string& name) const {
    auto factors = Parser(name).parseSize();
    return getSizeFromFactors(factors);
}

std::size_t BindingContext::getMaxEnumerationsPerVar() const {
    return options.maximumEnumerationsPerVar;
}
std::size_t BindingContext::getMaxVariablesInSize() const {
    return options.maximumVariablesInSize;
}
std::size_t BindingContext::getMaxVariablesPowersInSize() const {
    return options.maximumVariablesPowersInSize;
}

bool BindingContext::isSizeValid(const Size& size) const {
    int usedPrimaryVars = 0, primaryVarsPowers = 0;
    int usedCoefficientVars = 0, coefficientVarsPowers = 0;
    for (auto p: size.getPrimary()) {
        usedPrimaryVars += p != 0;
        primaryVarsPowers += std::abs(p);
    }
    for (auto c: size.getCoefficient()) {
        usedCoefficientVars += c != 0;
        coefficientVarsPowers += std::abs(c);
    }
    const bool divides =
        !options.requiresExactDivision ||
        std::ranges::all_of(
            getAllConsts(),
            [&](const ConcreteConsts& consts) { return size.isInteger(consts); }
        );
    return
        divides &&
        usedPrimaryVars + usedCoefficientVars <= options.maximumVariablesInSize &&
        primaryVarsPowers + coefficientVarsPowers <= options.maximumVariablesPowersInSize;
}

std::vector<Size> BindingContext::getSizes(const std::vector<std::string>& names) const {
    std::vector<Size> result;
    for (const auto& name: names) {
        result.emplace_back(getSize(name));
    }
    return result;
}

Shape BindingContext::getShape(const std::vector<std::string>& names) const {
    return Shape { getSizes(names) };
}

Shape BindingContext::getShape(std::string_view shape) const {
    return ranges::to<std::vector<Size>>(Parser(shape).parseShape() | std::views::transform([&](const auto& factors) {
        return getSizeFromFactors(factors);
    }));
}

ConcreteConsts BindingContext::realizeConsts(const std::map<std::string, std::size_t>& mappings, bool defaultFallback) const {
    ConcreteConsts consts;
    for (const auto& metadata: primaryMetadata) {
        std::size_t concrete;
        if (auto it = mappings.find(metadata.alias); it != mappings.end()) {
            concrete = it->second;
        } else {
            if (metadata.estimate.has_value()) {
                concrete = *metadata.estimate;
            } else {
                if (defaultFallback) {
                    concrete = 128;
                } else {
                    KAS_CRITICAL("No estimate for primary variable {}", metadata.alias);
                }
            }
        }
        consts.primary.emplace_back(static_cast<int>(concrete));
    }
    for (const auto& metadata: coefficientMetadata) {
        std::size_t concrete;
        if (auto it = mappings.find(metadata.alias); it != mappings.end()) {
            concrete = it->second;
        } else {
            if (metadata.estimate.has_value()) {
                concrete = *metadata.estimate;
            } else {
                if (defaultFallback) {
                    concrete = 3;
                } else {
                    KAS_CRITICAL("No estimate for coefficient variable {}", metadata.alias);
                }
            }
        }
        consts.coefficient.emplace_back(static_cast<int>(concrete));
    }
    return consts;
}

} // namespace kas
