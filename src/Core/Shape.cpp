#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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

bool Size::isMultipleOf(const Size& factor) const {
    for (int i = 0; i < primary.size(); ++i) {
        if (primary[i] < factor.primary[i]) {
            return false;
        }
    }
    return true;
}

std::vector<std::shared_ptr<Size>> Size::sampleFactors() const {
    std::vector<std::shared_ptr<Size>> factors;
    for (int primaryIndex = 0; primaryIndex < primary.size(); ++primaryIndex) {
        int primaryDim = primary[primaryIndex];
        if (primaryDim >= 1) {
            std::vector<int> newPrimary(primary.size(), 0);
            newPrimary[primaryIndex] = 1;
            factors.push_back(std::make_shared<Size>(newPrimary, std::vector<int>(coefficient.size(), 0)));
            for (int coefficientIndex = 0; coefficientIndex < coefficient.size(); ++coefficientIndex) {
                int coefficientDim = coefficient[coefficientIndex];
                if (coefficientDim >= 1) {
                    std::vector<int> newCoefficient(coefficient.size(), 0);
                    newCoefficient[coefficientIndex] = 1;
                    factors.push_back(std::make_shared<Size>(newPrimary, newCoefficient));
                }
            }
        }
    }
    return factors;
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

std::vector<std::shared_ptr<Size>> BindingContext::getPositiveCoefficients() const {
    const int count = coefficientMetadata.size();
    std::vector<std::shared_ptr<Size>> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i) {
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
            KAS_ASSERT(namedPrimaryCount < primaryMetadata.size());
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
    std::vector<int> drops,
    std::vector<std::pair<int, std::shared_ptr<Size>>> adds
) const {
    return Shape { ReplaceVector(sizes, drops, adds) };
}

std::vector<int> Shape::findSize(const Size& size) const {
    std::vector<int> result;
    for (int i = 0; i < sizes.size(); ++i) {
        if (*sizes[i] == size) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<int> Shape::findMultipleOfSize(const Size& factor) const {
    std::vector<int> result;
    for (int i = 0; i < sizes.size(); ++i) {
        if (sizes[i]->isMultipleOf(factor)) {
            result.push_back(i);
        }
    }
    return result;
}

std::string Shape::toString(const BindingContext& ctx) const {
    return VectorToString(sizes, std::function([&ctx](const std::shared_ptr<Size>& size) -> std::string {
        return size->toString(ctx);
    }));
}

namespace {
    enum class Token {
        OpenBracket, CloseBracket, Identifier, Comma, End
    };
    enum class Expect {
        OpenBracket, ListHead, Identifier, ListEnd, End
    };
    std::vector<std::string> parseShape(std::string_view shape) {
        std::vector<std::string> result;
        int pointer = 0;
        auto scan = [&]() -> std::pair<Token, std::string> {
            if (pointer >= shape.size()) {
                return { Token::End, "" };
            }
            std::stringstream token;
            bool inIdentifier = false;
            do {
                char curr = shape[pointer];
                switch (curr) {
                case ' ':
                case '\t':
                case '\r':
                case '\n':
                    if (inIdentifier) {
                        ++pointer;
                        return { Token::Identifier, token.str() };
                    }
                    break;
                case '[':
                    if (inIdentifier) {
                        return { Token::Identifier, token.str() };
                    }
                    ++pointer;
                    token << curr;
                    return { Token::OpenBracket, token.str() };
                case ']':
                    if (inIdentifier) {
                        return { Token::Identifier, token.str() };
                    }
                    ++pointer;
                    token << curr;
                    return { Token::CloseBracket, token.str() };
                case ',':
                    if (inIdentifier) {
                        return { Token::Identifier, token.str() };
                    }
                    ++pointer;
                    token << curr;
                    return { Token::Comma, token.str() };
                default:
                    inIdentifier = true;
                    token << curr;
                    break;
                }
                ++pointer;
            } while (pointer < shape.size());
            auto finalToken = token.str();
            if (!finalToken.empty())
                return { Token::Identifier, finalToken };
            return { Token::End, "" };
        };
        Expect expect = Expect::OpenBracket;
        while (true) {
            auto [token, value] = scan();
            switch (expect) {
            case Expect::OpenBracket:
                KAS_ASSERT(token == Token::OpenBracket);
                expect = Expect::ListHead;
                break;
            case Expect::ListHead:
                if (token == Token::CloseBracket) {
                    expect = Expect::End;
                } else {
                    KAS_ASSERT(token == Token::Identifier);
                    result.push_back(std::move(value));
                    expect = Expect::ListEnd;
                }
                break;
            case Expect::Identifier:
                KAS_ASSERT(token == Token::Identifier);
                result.push_back(std::move(value));
                expect = Expect::ListEnd;
                break;
            case Expect::ListEnd:
                if (token == Token::Comma) {
                    expect = Expect::Identifier;
                } else {
                    KAS_ASSERT(token == Token::CloseBracket);
                    expect = Expect::End;
                }
                break;
            case Expect::End:
                KAS_ASSERT(token == Token::End);
                return result;
            }
        }
        KAS_UNREACHABLE();
    }
}

std::vector<std::string> Shape::parseNames(std::string_view shape) {
    return parseShape(shape);
}

} // namespace kas
