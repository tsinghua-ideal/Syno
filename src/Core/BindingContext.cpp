#include <map>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

BindingContext::BindingContext(std::size_t countPrimary, std::size_t countCoefficient):
    namedPrimaryCount { 0 },
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
std::string_view BindingContext::getPrimaryAlias(std::size_t index) const {
    return primaryMetadata.at(index).alias;
}
std::string_view BindingContext::getCoefficientAlias(std::size_t index) const {
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

std::vector<std::shared_ptr<Size>> BindingContext::getPositiveCoefficients() const {
    std::vector<std::shared_ptr<Size>> result;
    result.reserve(getCoefficientCount());
    for (std::size_t i = 0; i < getCoefficientCount(); ++i) {
        result.emplace_back(getSingleCoefficientVariableSize(i));
    }
    return result;
}

Shape BindingContext::getShapeFromNames(const std::vector<std::string>& names) {
    using Metadata = BindingContext::Metadata;
    std::map<std::string, std::size_t> nameToIndex;
    for (std::size_t i = 0; i < namedPrimaryCount; ++i) {
        const auto& alias = primaryMetadata[i].alias;
        nameToIndex[alias] = i;
    }
    std::vector<std::shared_ptr<Size>> result;
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto& name = names[i];
        auto it = nameToIndex.find(name);
        if (it == nameToIndex.end()) {
            KAS_ASSERT(namedPrimaryCount < getPrimaryCount());
            nameToIndex[name] = namedPrimaryCount;
            primaryMetadata[namedPrimaryCount] = Metadata {
                .alias = name,
            };
            result.emplace_back(getSinglePrimaryVariableSize(namedPrimaryCount));
            ++namedPrimaryCount;
        } else {
            result.emplace_back(getSinglePrimaryVariableSize(it->second));
        }
    }
    return Shape { std::move(result) };
}

} // namespace kas
