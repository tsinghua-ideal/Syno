#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

namespace {
    std::map<std::string, std::size_t> GetLookupTable(const std::vector<BindingContext::Metadata>& metadata) {
        std::map<std::string, std::size_t> lookupTable;
        for (std::size_t i = 0; i < metadata.size(); ++i) {
            lookupTable[metadata[i].alias] = i;
        }
        return lookupTable;
    }
}

std::map<std::string, std::size_t> BindingContext::getPrimaryLookupTable() const {
    return GetLookupTable(primaryMetadata);
}

std::map<std::string, std::size_t> BindingContext::getCoefficientLookupTable() const {
    return GetLookupTable(coefficientMetadata);
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

std::shared_ptr<Size> BindingContext::getSinglePrimaryVariableSize(std::size_t index) const {
    KAS_ASSERT(index >= 0 && index < getPrimaryCount());
    auto res = std::make_shared<Size>(getPrimaryCount(), getCoefficientCount());
    res->primary[index] = 1;
    return res;
}

std::shared_ptr<Size> BindingContext::getSingleCoefficientVariableSize(std::size_t index) const {
    KAS_ASSERT(index >= 0 && index < getCoefficientCount());
    auto res = std::make_shared<Size>(getPrimaryCount(), getCoefficientCount());
    res->coefficient[index] = 1;
    return res;
}

Shape BindingContext::getShapeFromNames(const std::vector<std::string>& names) {
    std::map<std::string, std::size_t> pNameToIndex = getPrimaryLookupTable();
    std::map<std::string, std::size_t> cNameToIndex = getCoefficientLookupTable();
    std::vector<std::shared_ptr<Size>> result;
    for (const auto& name: names) {
        if (auto it = pNameToIndex.find(name); it != pNameToIndex.end())
            result.emplace_back(getSinglePrimaryVariableSize(it->second));
        else if (auto it = cNameToIndex.find(name); it != cNameToIndex.end())
            result.emplace_back(getSingleCoefficientVariableSize(it->second));
        else
            throw std::runtime_error("Unknown variable name: " + name);
    }
    return Shape { std::move(result) };
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
}

void BindingContext::applyEstimates(const std::map<std::string, std::size_t>& estimates) {
    std::map<std::string, std::size_t> pNameToIndex = getPrimaryLookupTable();
    std::map<std::string, std::size_t> cNameToIndex = getCoefficientLookupTable();
    for (const auto& [name, estimate]: estimates) {
        if (auto it = pNameToIndex.find(name); it != pNameToIndex.end())
            primaryMetadata[it->second].estimate = estimate;
        else if (auto it = cNameToIndex.find(name); it != cNameToIndex.end())
            coefficientMetadata[it->second].estimate = estimate;
        else
            throw std::runtime_error("Unknown variable name: " + name);
    }
}

namespace {
    void shadowNames(const std::map<std::string, std::size_t>& mappings, std::vector<std::size_t>& shadowed, const std::vector<BindingContext::Metadata>& metadata) {
        for (const auto& m: metadata) {
            if (auto it = mappings.find(m.alias); it != mappings.end())
                shadowed.emplace_back(it->second);
            else
                shadowed.emplace_back(m.estimate);
        }
    }
}

std::vector<std::size_t> BindingContext::getKernelArguments(const std::map<std::string, std::size_t>& mappings) const {
    std::vector<std::size_t> result;
    shadowNames(mappings, result, primaryMetadata);
    shadowNames(mappings, result, coefficientMetadata);
    return result;
}

std::vector<std::size_t> BindingContext::evaluateShape(const Shape& shape, const std::map<std::string, std::size_t>& mappings) const {
    std::vector<std::size_t> shadowedPrimary, shadowedCoefficient;
    shadowNames(mappings, shadowedPrimary, primaryMetadata);
    shadowNames(mappings, shadowedCoefficient, coefficientMetadata);
    return shape.eval<std::size_t>(
        [&](std::size_t i) { return shadowedPrimary[i]; },
        [&](std::size_t i) { return shadowedCoefficient[i]; }
    );
}

} // namespace kas
