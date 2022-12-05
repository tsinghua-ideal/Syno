#include <functional>
#include <memory>
#include <sstream>

#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

bool Size::isCoefficient() const {
    for (int dim: primary) {
        if (dim < 0) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<Size> Size::operator*(const Size& other) const {
    KAS_ASSERT(primary.size() == other.primary.size() && coefficient.size() == other.coefficient.size());
    std::vector<int> newPrimary { primary };
    std::vector<int> newCoefficient { coefficient };
    for (int i = 0; i < primary.size(); ++i) {
        newPrimary[i] += other.primary[i];
    }
    for (int i = 0; i < coefficient.size(); ++i) {
        newCoefficient[i] += other.coefficient[i];
    }
    return std::make_shared<Size>(std::move(newPrimary), std::move(newCoefficient));
}

std::shared_ptr<Size> Size::operator/(const Size &other) const {
    KAS_ASSERT(primary.size() == other.primary.size() && coefficient.size() == other.coefficient.size());
    std::vector<int> newPrimary { primary };
    std::vector<int> newCoefficient { coefficient };
    for (int i = 0; i < primary.size(); ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        KAS_ASSERT(newPrimary[i] >= 0);
    }
    for (int i = 0; i < coefficient.size(); ++i) {
        newCoefficient[i] -= other.coefficient[i];
    }
    return std::make_shared<Size>(std::move(newPrimary), std::move(newCoefficient));
}

bool Size::operator==(const Size& other) const {
    return primary == other.primary && coefficient == other.coefficient;
}

std::string Size::toString(const BindingContext& ctx) const {
    KAS_ASSERT(primary.size() == ctx.primaryMetadata.size() && coefficient.size() == ctx.coefficientMetadata.size());
    std::stringstream result;
    bool hasCoefficient = false;
    result << "(";
    bool hasNominator = false;
    bool hasDenominator = false;
    std::stringstream denominator;
    for (int i = 0; i < coefficient.size(); ++i) {
        if (coefficient[i] < 0) {
            hasCoefficient = true;
            hasDenominator = true;
            denominator << ctx.coefficientMetadata[i].alias;
            if (coefficient[i] != -1) {
                denominator << "^" << -coefficient[i];
            }
        } else if (coefficient[i] > 0) {
            hasCoefficient = true;
            hasNominator = true;
            result << ctx.coefficientMetadata[i].alias;
            if (coefficient[i] != 1) {
                result << "^" << coefficient[i];
            }
        }
    }
    if (!hasNominator) {
        result << "1";
    }
    if (hasDenominator) {
        result << "/" << denominator.str() << ")";
    } else {
        result << ")";
    }
    if (!hasCoefficient) {
        result.str("");
    }
    bool hasPrimary = false;
    for (int i = 0; i < primary.size(); ++i) {
        if (primary[i] > 0) {
            hasPrimary = true;
            result << ctx.primaryMetadata[i].alias;
            if (primary[i] != 1) {
                result << "^" << primary[i];
            }
        }
    }
    if (!hasPrimary) {
        result << "1";
    }
    return result.str();
}

BindingContext::Metadata::Metadata(const std::string& alias):
    alias { alias }
{}

BindingContext::BindingContext(int countPrimary, int countCoefficient):
    primaryMetadata(countPrimary),
    coefficientMetadata(countCoefficient) {
    for (int i = 0; i < countPrimary; ++i) {
        primaryMetadata[i] = Metadata { "x_" + std::to_string(i) };
    }
    for (int i = 0; i < countCoefficient; ++i) {
        coefficientMetadata[i] = Metadata { "c_" + std::to_string(i) };
    }
}

std::shared_ptr<Size> BindingContext::getSinglePrimaryVariableSize(int index) const {
    KAS_ASSERT(index >= 0 && index < primaryMetadata.size());
    std::vector<int> primary(primaryMetadata.size(), 0);
    primary[index] = 1;
    return std::make_shared<Size>(std::move(primary), std::vector<int>(coefficientMetadata.size(), 0));
}

std::shared_ptr<Size> BindingContext::getSingleCoefficientVariableSize(int index) const {
    KAS_ASSERT(index >= 0 && index < coefficientMetadata.size());
    std::vector<int> coefficient(coefficientMetadata.size(), 0);
    coefficient[index] = 1;
    return std::make_shared<Size>(std::vector<int>(primaryMetadata.size(), 0), std::move(coefficient));
}

Shape::Shape(const std::vector<std::shared_ptr<Size>>& sizes):
    sizes(sizes)
{}
Shape::Shape(std::vector<std::shared_ptr<Size>>&& sizes):
    sizes(std::move(sizes))
{}

size_t Shape::size() const {
    return sizes.size();
}

const std::shared_ptr<Size>& Shape::operator[](size_t index) const {
    KAS_ASSERT(index < sizes.size());
    return sizes[index];
}

Shape Shape::replace(
    const std::vector<int>& drops,
    const std::vector<std::pair<int, std::shared_ptr<Size>>>& adds
) const {
    return Shape { ReplaceVector(sizes, drops, adds) };
}

std::string Shape::toString(const BindingContext& ctx) const {
    return VectorToString(sizes, std::function([&ctx](const std::shared_ptr<Size>& size) -> std::string {
        return size->toString(ctx);
    }));
}

} // namespace kas
