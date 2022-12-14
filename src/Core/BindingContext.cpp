#include <map>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

BindingContext::Metadata::Metadata(std::string_view alias):
    alias { alias }
{}

BindingContext::TensorMetadata::TensorMetadata(std::string_view name):
    name { name }
{}

BindingContext::IteratorVariableMetadata::IteratorVariableMetadata(std::string_view name):
    name { name }
{}

BindingContext::BindingContext(int countPrimary, int countCoefficient):
    namedPrimaryCount { 0 },
    primaryMetadata(countPrimary),
    coefficientMetadata(countCoefficient) {
    for (int i = 0; i < countPrimary; ++i) {
        primaryMetadata[i] = Metadata { "x_" + std::to_string(i) };
    }
    for (int i = 0; i < countCoefficient; ++i) {
        coefficientMetadata[i] = Metadata { "c_" + std::to_string(i) };
    }
}

std::size_t BindingContext::getPrimaryCount() const {
    return primaryMetadata.size();
}
std::size_t BindingContext::getCoefficientCount() const {
    return coefficientMetadata.size();
}
std::string_view BindingContext::getPrimaryAlias(std::size_t index) const {
    return primaryMetadata.at(index).alias;
}
std::string_view BindingContext::getCoefficientAlias(std::size_t index) const {
    return coefficientMetadata.at(index).alias;
}

std::shared_ptr<Size> BindingContext::getSinglePrimaryVariableSize(int index) const {
    KAS_ASSERT(index >= 0 && index < getPrimaryCount());
    auto res = std::make_shared<Size>(getPrimaryCount(), getCoefficientCount());
    res->primary[index] = 1;
    return res;
}

std::shared_ptr<Size> BindingContext::getSingleCoefficientVariableSize(int index) const {
    KAS_ASSERT(index >= 0 && index < getCoefficientCount());
    auto res = std::make_shared<Size>(getPrimaryCount(), getCoefficientCount());
    res->coefficient[index] = 1;
    return res;
}

std::vector<std::shared_ptr<Size>> BindingContext::getPositiveCoefficients() const {
    std::vector<std::shared_ptr<Size>> result;
    result.reserve(getCoefficientCount());
    for (int i = 0; i < getCoefficientCount(); ++i) {
        result.push_back(getSingleCoefficientVariableSize(i));
    }
    return result;
}

Shape BindingContext::getShapeFromNames(const std::vector<std::string>& names) {
    using Metadata = BindingContext::Metadata;
    std::map<std::string, int> nameToIndex;
    for (int i = 0; i < namedPrimaryCount; ++i) {
        const auto& alias = primaryMetadata[i].alias;
        nameToIndex[alias] = i;
    }
    std::vector<std::shared_ptr<Size>> result;
    for (int i = 0; i < names.size(); ++i) {
        const auto& name = names[i];
        auto it = nameToIndex.find(name);
        if (it == nameToIndex.end()) {
            KAS_ASSERT(namedPrimaryCount < getPrimaryCount());
            nameToIndex[name] = namedPrimaryCount;
            primaryMetadata[namedPrimaryCount] = Metadata { name };
            result.push_back(getSinglePrimaryVariableSize(namedPrimaryCount));
            ++namedPrimaryCount;
        } else {
            result.push_back(getSinglePrimaryVariableSize(it->second));
        }
    }
    return Shape { std::move(result) };
}

std::string_view BindingContext::getTensorName(std::size_t index) const {
    return tensorMetadata.at(index).name;
}

std::size_t BindingContext::addTensor(std::string_view name) {
    tensorMetadata.push_back(TensorMetadata { name });
    return tensorMetadata.size() - 1;
}

std::string_view BindingContext::getIteratorVariableName(std::size_t index) const {
    return iteratorVariableMetadata.at(index).name;
}

std::size_t BindingContext::addIteratorVariable(std::string_view name) {
    iteratorVariableMetadata.push_back(IteratorVariableMetadata { name });
    return iteratorVariableMetadata.size() - 1;
}

} // namespace kas
