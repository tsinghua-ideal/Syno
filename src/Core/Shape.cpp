#include <algorithm>
#include <cmath>
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

Size::Size(std::size_t primaryCount, std::size_t coefficientCount):
    primaryCount { primaryCount },
    coefficientCount { coefficientCount },
    primary {},
    coefficient {}
{}

Size& Size::operator=(const Size& other) & {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    primary = other.primary;
    coefficient = other.coefficient;
    return *this;
}

Size Size::identity() const {
    return { primaryCount, coefficientCount };
}

Size::Trait Size::getTrait() const {
    bool hasPrimary = false;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        hasPrimary |= primary[i] > 0;
    }
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        hasCoefficient |= coefficient[i] != 0;
        hasNegativeCoefficient |= coefficient[i] < 0;
    }
    if (hasPrimary) {
        return Trait::General;
    } else {
        if (hasCoefficient) {
            if (hasNegativeCoefficient) {
                return Trait::IllegalCoefficient;
            } else {
                return Trait::Coefficient;
            }
        } else {
            return Trait::One;
        }
    }
}

bool Size::is1() const {
    for (std::size_t i = 0; i < primaryCount; ++i) {
        if (primary[i] != 0) return false;
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        if (coefficient[i] != 0) return false;
    }
    return true;
}

bool Size::isLegalCoefficient() const {
    for (std::size_t i = 0; i < primaryCount; ++i) {
        if (primary[i] != 0) return false;
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        if (coefficient[i] < 0) return false;
    }
    return true;
}

std::vector<std::shared_ptr<Size>> Size::sampleFactors(const BindingContext& ctx) const {
    std::vector<std::shared_ptr<Size>> factors;
    for (std::size_t primaryIndex = 0; primaryIndex < primaryCount; ++primaryIndex) {
        int primaryDim = primary[primaryIndex];
        if (primaryDim >= 1) {
            auto primaryRes = std::make_shared<Size>(primaryCount, coefficientCount);
            primaryRes->primary[primaryIndex] = 1;
            factors.emplace_back(primaryRes);
            for (int coefficientIndex = 0; coefficientIndex < coefficientCount; ++coefficientIndex) {
                std::size_t coefficientDim = coefficient[coefficientIndex];
                if (coefficientDim >= 1) {
                    auto res = std::make_shared<Size>(*primaryRes);
                    res->coefficient[coefficientIndex] = 1;
                    factors.emplace_back(res);
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
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] += other.primary[i];
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
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
        for (std::size_t i = 0; i < primaryCount; ++i) {
            newPrimary[i] += operand.primary[i];
        }
        for (std::size_t i = 0; i < coefficientCount; ++i) {
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
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        KAS_ASSERT(newPrimary[i] >= 0);
    }
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        newCoefficient[i] -= other.coefficient[i];
    }
    return newSize;
}

std::optional<Size::Trait> Size::testDividedBy(const Size& other) {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    auto newPrimary = primary;
    bool hasPrimary = false;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        newPrimary[i] -= other.primary[i];
        // Ensure that no primary variable is in denominator
        if(newPrimary[i] < 0) return std::nullopt;
        hasPrimary |= newPrimary[i] > 0;
    }
    primary = newPrimary;
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        coefficient[i] -= other.coefficient[i];
        hasCoefficient |= coefficient[i] != 0;
        hasNegativeCoefficient |= coefficient[i] < 0;
    }
    if (hasPrimary) {
        return Trait::General;
    } else {
        if (hasCoefficient) {
            if (hasNegativeCoefficient) {
                return Trait::IllegalCoefficient;
            } else {
                return Trait::Coefficient;
            }
        } else {
            return Trait::One;
        }
    }
}

std::size_t Size::estimate(const BindingContext& ctx) const {
    auto factor = [](std::size_t cnt, auto f, const Size::ExprType& powers) -> std::pair<std::size_t, std::size_t> {
        std::size_t nominator = 1;
        std::size_t denominator = 1;
        for (std::size_t i = 0; i < cnt; ++i) {
            if (powers[i] > 0) {
                const std::size_t est = f(i);
                for (std::size_t j = 0; j < powers[i]; ++j) {
                    nominator *= est;
                }
            } else if (powers[i] < 0) {
                const std::size_t est = f(i);
                for (std::size_t j = 0; j < -powers[i]; ++j) {
                    denominator *= est;
                }
            }
        }
        return { nominator, denominator };
    };
    auto [nP, dP] = factor(primaryCount, [&ctx](std::size_t i) { return ctx.getPrimaryEstimate(i); }, primary);
    auto [nC, dC] = factor(coefficientCount, [&ctx](std::size_t i) { return ctx.getCoefficientEstimate(i); }, coefficient);
    return nP * nC / dP / dC;
}

std::string Size::toString(const BindingContext& ctx) const {
    KAS_ASSERT(primaryCount == ctx.getPrimaryCount() && coefficientCount == ctx.getCoefficientCount());
    std::stringstream result;
    bool hasCoefficient = false;
    result << "(";
    bool hasNominator = false;
    bool hasDenominator = false;
    std::stringstream denominator;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
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
    for (std::size_t i = 0; i < primaryCount; ++i) {
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

LabeledSize::LabeledSize(std::size_t primaryCount, std::size_t coefficientCount):
    trait { Trait::One },
    Size { primaryCount, coefficientCount }
{}

LabeledSize::LabeledSize(const Size& size):
    trait { size.getTrait() },
    Size { size }
{}

LabeledSize LabeledSize::identity() const {
    return { primaryCount, coefficientCount };
}

bool LabeledSize::is1() const {
    return trait == Trait::One;
}

bool LabeledSize::isLegalCoefficient() const {
    return trait == Trait::Coefficient;
}

bool LabeledSize::isIllegalCoefficient() const {
    return trait == Trait::IllegalCoefficient;
}

bool LabeledSize::isIndeterminedCoefficient() const {
    return trait == Trait::Coefficient || trait == Trait::IllegalCoefficient;
}

bool LabeledSize::isGeneral() const {
    return trait == Trait::General;
}

bool LabeledSize::testDividedBy(const Size& other) {
    auto res = Size::testDividedBy(other);
    if (res.has_value()) {
        trait = res.value();
        return true;
    }
    return false;
}

LabeledSize& LabeledSize::operator*=(const LabeledSize& other) {
    KAS_ASSERT(primaryCount == other.primaryCount && coefficientCount == other.coefficientCount);
    bool hasPrimary = false;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        primary[i] += other.primary[i];
        hasPrimary |= primary[i] > 0;
    }
    bool hasCoefficient = false;
    bool hasNegativeCoefficient = false;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        coefficient[i] += other.coefficient[i];
        hasCoefficient |= coefficient[i] != 0;
        hasNegativeCoefficient |= coefficient[i] < 0;
    }
    if (hasPrimary) {
        trait = Trait::General;
    } else {
        if (hasCoefficient) {
            if (hasNegativeCoefficient) {
                trait = Trait::IllegalCoefficient;
            } else {
                trait = Trait::Coefficient;
            }
        } else {
            trait = Trait::One;
        }
    }
    return *this;
}

LabeledSize LabeledSize::operator*(const LabeledSize& other) const {
    auto newSize = *this;
    newSize *= other;
    return newSize;
}

std::optional<LabeledSize> LabeledSize::absorbCoefficientNumeratorToDenominator(const LabeledSize& other) const {
    KAS_ASSERT(isIndeterminedCoefficient() && other.isIndeterminedCoefficient());
    Size res(primaryCount, coefficientCount);
    bool contribution = false;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        auto c1 = coefficient[i];
        auto c2 = other.coefficient[i];
        contribution |= c1 < 0 && c2 > 0;
        res.coefficient[i] = c1 + c2;
    }
    if (contribution) {
        return { std::move(res) };
    } else {
        return std::nullopt;
    }
}

int LabeledSize::scoreOfGeneralDimension(const LabeledSize& other) const {
    KAS_ASSERT(isIllegalCoefficient() && other.isGeneral());
    int score = 0;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        score += std::abs(coefficient[i]) + std::abs(other.coefficient[i]) - std::abs(coefficient[i] + other.coefficient[i]);
    }
    return score;
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
    std::vector<std::size_t> drops,
    std::vector<std::pair<std::size_t, std::shared_ptr<Size>>> adds
) const {
    return Shape { ReplaceVector(sizes, drops, adds) };
}

std::vector<std::size_t> Shape::estimate(const BindingContext& ctx) const {
    std::vector<std::size_t> result;
    for (const auto& size: sizes) {
        result.emplace_back(size->estimate(ctx));
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
        std::size_t pointer = 0;
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
                    result.emplace_back(std::move(value));
                    expect = Expect::ListEnd;
                }
                break;
            case Expect::Identifier:
                KAS_ASSERT(token == Token::Identifier);
                result.emplace_back(std::move(value));
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
