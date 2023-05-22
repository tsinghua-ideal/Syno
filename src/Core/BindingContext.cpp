#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"


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

namespace {
    std::map<std::string, std::size_t> GetLookupTable(const std::vector<BindingContext::Metadata>& metadata) {
        std::map<std::string, std::size_t> lookupTable;
        for (std::size_t i = 0; i < metadata.size(); ++i) {
            lookupTable[metadata[i].alias] = i;
        }
        return lookupTable;
    }
}

BindingContext::LookUpTable BindingContext::getPrimaryLookupTable() const {
    return GetLookupTable(primaryMetadata);
}

BindingContext::LookUpTable BindingContext::getCoefficientLookupTable() const {
    return GetLookupTable(coefficientMetadata);
}

Size BindingContext::lookUp(const std::string& name, const LookUpTable& primaryTable, const LookUpTable& coefficientTable) const {
    if (auto it = primaryTable.find(name); it != primaryTable.end())
        return getSinglePrimaryVariableSize(it->second);
    else if (auto it = coefficientTable.find(name); it != coefficientTable.end())
        return getSingleCoefficientVariableSize(it->second);
    else
        throw std::runtime_error("Unknown variable name: " + name);
}

BindingContext::BindingContext(std::size_t countPrimary, std::size_t countCoefficient):
    primaryMetadata(countPrimary),
    coefficientMetadata(countCoefficient) {
    for (std::size_t i = 0; i < countPrimary; ++i) {
        primaryMetadata[i] = Metadata {
            .alias = "x_" + std::to_string(i),
        };
    }
    for (std::size_t i = 0; i < countCoefficient; ++i) {
        coefficientMetadata[i] = Metadata {
            .alias = "c_" + std::to_string(i),
        };
    }
    defaultConsts = realizeConsts({});
}

BindingContext::BindingContext(std::vector<Metadata> primaryMetadata, std::vector<Metadata> coefficientMetadata):
    primaryMetadata { std::move(primaryMetadata) },
    coefficientMetadata { std::move(coefficientMetadata) }
{
    defaultConsts = realizeConsts({});
}

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
std::size_t BindingContext::getPrimaryEstimate(std::size_t index) const {
    return primaryMetadata.at(index).estimate;
}
std::size_t BindingContext::getCoefficientEstimate(std::size_t index) const {
    return coefficientMetadata.at(index).estimate;
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
    auto pNameToIndex = getPrimaryLookupTable();
    auto cNameToIndex = getCoefficientLookupTable();
    return lookUp(name, pNameToIndex, cNameToIndex);
}

std::vector<Size> BindingContext::getSizes(const std::vector<std::string>& names) const {
    std::map<std::string, std::size_t> pNameToIndex = getPrimaryLookupTable();
    std::map<std::string, std::size_t> cNameToIndex = getCoefficientLookupTable();
    std::vector<Size> result;
    for (const auto& name: names) {
        result.emplace_back(lookUp(name, pNameToIndex, cNameToIndex));
    }
    return result;
}

Shape BindingContext::getShape(const std::vector<std::string>& names) const {
    return Shape { getSizes(names) };
}

void BindingContext::applySpecs(std::vector<std::pair<std::string, Parser::PureSpec>>& primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>>& coefficientSpecs) {
    for (std::size_t i = 0; i < primarySpecs.size(); ++i) {
        auto& [name, spec] = primarySpecs[i];
        primaryMetadata[i] = Metadata {
            .alias = std::move(name),
            .maximumOccurrence = spec.maxOccurrences.value_or(3),
            .estimate = spec.size.value_or(128),
        };
    }
    for (std::size_t i = 0; i < coefficientSpecs.size(); ++i) {
        auto& [name, spec] = coefficientSpecs[i];
        coefficientMetadata[i] = Metadata {
            .alias = std::move(name),
            .isOdd = spec.size.value_or(0) % 2 == 1,
            .maximumOccurrence = spec.maxOccurrences.value_or(3),
            .estimate = spec.size.value_or(3),
        };
    }
    defaultConsts = realizeConsts({});
}

ConcreteConsts BindingContext::realizeConsts(const std::map<std::string, std::size_t>& mappings) const {
    ConcreteConsts consts;
    for (std::size_t i = 0; const auto& metadata: primaryMetadata) {
        int concrete;
        if (auto it = mappings.find(metadata.alias); it != mappings.end()) {
            concrete = static_cast<int>(it->second);
        } else {
            concrete = static_cast<int>(getPrimaryEstimate(i));
        }
        consts.primary.emplace_back(concrete);
        ++i;
    }
    for (std::size_t i = 0; const auto& metadata: coefficientMetadata) {
        int concrete;
        if (auto it = mappings.find(metadata.alias); it != mappings.end()) {
            concrete = static_cast<int>(it->second);
        } else {
            concrete = static_cast<int>(getCoefficientEstimate(i));
        }
        consts.coefficient.emplace_back(concrete);
        ++i;
    }
    return consts;
}

void BindingContext::applyMappings(const std::vector<std::map<std::string, std::size_t>>& allMappings) {
    decltype(allConsts) result;
    for (const auto& mappings: allMappings) {
        result.emplace_back(realizeConsts(mappings));
    }
    allConsts = std::move(result);
}

} // namespace kas
