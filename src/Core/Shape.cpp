#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

bool Size::isCoefficientRealizable(const ExprType& toBeRealized, const BindingContext& ctx) {
    bool allZero = true;
    bool somePositive = false;
    for (int power: toBeRealized) {
        if (power != 0) allZero = false;
        if (power > 0) somePositive = true;
    }
    return somePositive || allZero;
}

Size::Size(std::size_t primaryCount, std::size_t coefficientCount):
    primaryCount { primaryCount },
    coefficientCount { coefficientCount },
    primary {},
    coefficient {}
{}

bool Size::is1() const {
    for (int dim: primary) {
        if (dim != 0) return false;
    }
    for (int dim: coefficient) {
        if (dim != 0) return false;
    }
    return true;
}

bool Size::isCoefficient() const {
    for (int dim: primary) {
        if (dim < 0) {
            return false;
        }
    }
    return true;
}

bool Size::isMultipleOf(const Size& factor, const BindingContext& ctx) const {
    for (int i = 0; i < primaryCount; ++i) {
        if (primary[i] < factor.primary[i]) {
            return false;
        }
    }
    ExprType newCoefficient { };
    for (int i = 0; i < coefficientCount; ++i) {
        newCoefficient[i] = coefficient[i] - factor.coefficient[i];
    }
    return isCoefficientRealizable(newCoefficient, ctx);
}

std::vector<std::shared_ptr<Size>> Size::sampleFactors(const BindingContext& ctx) const {
    std::vector<std::shared_ptr<Size>> factors;
    for (int primaryIndex = 0; primaryIndex < primaryCount; ++primaryIndex) {
        int primaryDim = primary[primaryIndex];
        if (primaryDim >= 1) {
            auto primaryRes = std::make_shared<Size>(primaryCount, coefficientCount);
            primaryRes->primary[primaryIndex] = 1;
            factors.push_back(primaryRes);
            for (int coefficientIndex = 0; coefficientIndex < coefficientCount; ++coefficientIndex) {
                int coefficientDim = coefficient[coefficientIndex];
                if (coefficientDim >= 1) {
                    auto res = std::make_shared<Size>(*primaryRes);
                    res->coefficient[coefficientIndex] = 1;
                    factors.push_back(res);
                }
            }
        }
    }
    return factors;
}

std::shared_ptr<Size> Size::operator*(const Size& other) const {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newSize = std::make_shared<Size>(*this);
    auto& newPrimary = newSize->primary;
    auto& newCoefficient = newSize->coefficient;
    for (int i = 0; i < primaryCount; ++i) {
        newPrimary[i] += other.primary[i];
    }
    for (int i = 0; i < coefficientCount; ++i) {
        newCoefficient[i] += other.coefficient[i];
    }
    return newSize;
}

std::shared_ptr<Size> Size::Product(const std::vector<std::shared_ptr<Size>>& operands) {
    auto newSize = std::make_shared<Size>(*operands.at(0));
    auto& newPrimary = newSize->primary;
    auto& newCoefficient = newSize->coefficient;
    const auto primaryCount = newSize->primaryCount;
    const auto coefficientCount = newSize->coefficientCount;
    for (std::size_t index = 1; index < operands.size(); ++index) {
        const auto& operand = *operands[index];
        KAS_ASSERT(primaryCount == operand.primaryCount && coefficientCount == operand.coefficientCount);
        for (int i = 0; i < primaryCount; ++i) {
            newPrimary[i] += operand.primary[i];
        }
        for (int i = 0; i < coefficientCount; ++i) {
            newCoefficient[i] += operand.coefficient[i];
        }
    }
    return newSize;
}

std::shared_ptr<Size> Size::operator/(const Size &other) const {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newSize = std::make_shared<Size>(*this);
    auto& newPrimary = newSize->primary;
    auto& newCoefficient = newSize->coefficient;
    for (int i = 0; i < primaryCount; ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        KAS_ASSERT(newPrimary[i] >= 0);
    }
    for (int i = 0; i < coefficientCount; ++i) {
        newCoefficient[i] -= other.coefficient[i];
    }
    return newSize;
}

bool Size::operator==(const Size& other) const {
    return primary == other.primary && coefficient == other.coefficient;
}

std::string Size::toString(const BindingContext& ctx) const {
    KAS_ASSERT(primaryCount == ctx.getPrimaryCount() && coefficientCount == ctx.getCoefficientCount());
    std::stringstream result;
    bool hasCoefficient = false;
    result << "(";
    bool hasNominator = false;
    bool hasDenominator = false;
    std::stringstream denominator;
    for (int i = 0; i < coefficientCount; ++i) {
        if (coefficient[i] < 0) {
            hasCoefficient = true;
            hasDenominator = true;
            denominator << ctx.getCoefficientAlias(i);
            if (coefficient[i] != -1) {
                denominator << "^" << static_cast<int>(-coefficient[i]);
            }
        } else if (coefficient[i] > 0) {
            hasCoefficient = true;
            hasNominator = true;
            result << ctx.getCoefficientAlias(i);
            if (coefficient[i] != 1) {
                result << "^" << static_cast<int>(coefficient[i]);
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
    for (int i = 0; i < primaryCount; ++i) {
        if (primary[i] > 0) {
            hasPrimary = true;
            result << ctx.getPrimaryAlias(i);
            if (primary[i] != 1) {
                result << "^" << static_cast<int>(primary[i]);
            }
        }
    }
    if (!hasPrimary) {
        result << "1";
    }
    return result.str();
}

Shape::Shape(const std::vector<std::shared_ptr<Size>>& sizes):
    sizes(sizes)
{}
Shape::Shape(std::vector<std::shared_ptr<Size>>&& sizes):
    sizes(std::move(sizes))
{}

const std::vector<std::shared_ptr<Size>>& Shape::getSizes() const {
    return sizes;
}

std::size_t Shape::size() const {
    return sizes.size();
}

const std::shared_ptr<Size>& Shape::operator[](std::size_t index) const {
    return sizes.at(index);
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

std::vector<int> Shape::findMultipleOfSize(const Size& factor, const BindingContext& ctx) const {
    std::vector<int> result;
    for (int i = 0; i < sizes.size(); ++i) {
        if (sizes[i]->isMultipleOf(factor, ctx)) {
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
