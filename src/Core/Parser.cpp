#include <cstddef>
#include <optional>
#include <sstream>
#include <vector>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string_view Parser::what(Token tok) {
    switch (tok) {
    case Token::Identifier:
        return "identifier";
    case Token::Integer:
        return "integer";
    case Token::Plus:
        return "'+'";
    case Token::Times:
        return "'*'";
    case Token::Power:
        return "'^'";
    case Token::Comma:
        return "','";
    case Token::Colon:
        return "':'";
    case Token::Equal:
        return "'='";
    case Token::OpenParen:
        return "'('";
    case Token::CloseParen:
        return "')'";
    case Token::OpenBracket:
        return "'['";
    case Token::CloseBracket:
        return "']'";
    case Token::End:
        return "EOF";
    }
    KAS_UNREACHABLE();
}

Parser::Token Parser::current() const {
    return tokens.at(pointer).first;
}

Parser::Token Parser::peek() const {
    return tokens.at(pointer + 1).first;
}

std::string Parser::consume(Token token) {
    if (current() != token) {
        std::ostringstream oss;
        oss << "Expected " << what(token) << " but got " << what(current());
        throw std::runtime_error(oss.str());
    }
    return std::move(tokens.at(pointer++).second);
}

Parser::Parser(std::string_view buffer) {
    std::stringstream token;
    enum class State {
        Normal, Identifier, Integer,
    };
    State state = State::Normal;
    auto finish = [&](Token tok) {
        tokens.emplace_back(tok, token.str());
        token.str("");
        state = State::Normal;
    };
    auto switchToNormal = [&]() {
        switch (state) {
        case State::Normal:
            break;
        case State::Identifier:
            finish(Token::Identifier);
            break;
        case State::Integer:
            finish(Token::Integer);
            break;
        }
    };
    auto isIdentifierHead = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    auto isIntegerPart = [](char c) {
        return c >= '0' && c <= '9';
    };
    auto isIntegerHead = [](char c) {
        return (c >= '0' && c <= '9') || c == '-';
    };
    for (char c: buffer) {
        switch (c) {
        case ' ':
        case '\t':
        case '\r':
        case '\n':
            switchToNormal();
            break;
        case '+':
            switchToNormal();
            tokens.emplace_back(Token::Plus, "");
            break;
        case '*':
            switchToNormal();
            tokens.emplace_back(Token::Times, "");
            break;
        case '^':
            switchToNormal();
            tokens.emplace_back(Token::Power, "");
            break;
        case ',':
            switchToNormal();
            tokens.emplace_back(Token::Comma, "");
            break;
        case ':':
            switchToNormal();
            tokens.emplace_back(Token::Colon, "");
            break;
        case '=':
            switchToNormal();
            tokens.emplace_back(Token::Equal, "");
            break;
        case '(':
            switchToNormal();
            tokens.emplace_back(Token::OpenParen, "");
            break;
        case ')':
            switchToNormal();
            tokens.emplace_back(Token::CloseParen, "");
            break;
        case '[':
            switchToNormal();
            tokens.emplace_back(Token::OpenBracket, "");
            break;
        case ']':
            switchToNormal();
            tokens.emplace_back(Token::CloseBracket, "");
            break;
        default:
            switch (state) {
            case State::Normal:
                if (isIdentifierHead(c)) {
                    state = State::Identifier;
                    token << c;
                } else if (isIntegerHead(c)) {
                    state = State::Integer;
                    token << c;
                } else {
                    throw std::runtime_error("Invalid character in shape string.");
                }
                break;
            case State::Identifier:
                token << c;
                break;
            case State::Integer:
                if (isIntegerPart(c)) {
                    token << c;
                } else {
                    throw std::runtime_error("Invalid character in shape string.");
                }
                break;
            }
            break;
        }
    }
    switchToNormal();
    tokens.emplace_back(Token::End, "");
}

std::string Parser::parseIdentifier() {
    return consume(Token::Identifier);
}
int Parser::parseInteger() {
    return std::stoi(consume(Token::Integer));
}
Parser::Factor Parser::parseBaseAndPower() {
    std::string id = parseIdentifier();
    if (current() == Token::Power) {
        consume(Token::Power);
        int power = parseInteger();
        return {std::move(id), power};
    } else {
        return {std::move(id), 1};
    }
}
std::vector<Parser::Factor> Parser::parseSize() {
    if (current() == Token::Identifier) {
        std::vector<Factor> factors = { parseBaseAndPower() };
        while (current() == Token::Times) {
            consume(Token::Times);
            factors.emplace_back(parseBaseAndPower());
        }
        return factors;
    } else if (current() == Token::Integer) {
        int value = parseInteger();
        if (value != 1) {
            throw std::runtime_error("Numeric constants other than 1 are not allowed.");
        }
        return {};
    }
    throw std::runtime_error("Invalid size expression.");
}
Parser::AttributedSize Parser::parseSizeAndAttributes() {
    auto size = parseSize();
    if (current() == Token::Colon) {
        consume(Token::Colon);
        std::set<std::string> attributes;
        attributes.emplace(parseIdentifier());
        while (current() == Token::Plus) {
            consume(Token::Plus);
            attributes.emplace(parseIdentifier());
        }
        return { std::move(size), std::move(attributes) };
    }
    return { std::move(size), {} };
}
std::vector<Parser::AttributedSize> Parser::parseCommaSeparatedSizesAndAttributes() {
    std::vector<AttributedSize> sizes = { parseSizeAndAttributes() };
    while (current() == Token::Comma) {
        consume(Token::Comma);
        sizes.emplace_back(parseSizeAndAttributes());
    }
    return sizes;
}

std::vector<Parser::AttributedSize> Parser::parseShapeAndAttributes() {
    consume(Token::OpenBracket);
    if (current() != Token::CloseBracket) {
        return parseCommaSeparatedSizesAndAttributes();
    }
    consume(Token::CloseBracket);
    return {};
}

TensorExpression Parser::parseFactorExpression() {
    if (current() == Token::OpenParen) {
        return parseTensorExpression();
    }
    auto id = parseIdentifier();
    KAS_ASSERT(id.starts_with("in_"));
    int index = -1;
    std::stringstream ss;
    ss << id.substr(3);
    ss >> index;
    return TensorTensorExpression::Create(index);
}

TensorExpression Parser::parseTermExpression() {
    TensorExpression expr = parseFactorExpression();
    while (current() == Token::Times) {
        consume(Token::Times);
        expr *= parseFactorExpression();
    }
    return expr;
}

TensorExpression Parser::parseTensorExpression() {
    if (current() == Token::OpenParen) {
        consume(Token::OpenParen);
        TensorExpression expr = parseTensorExpression();
        consume(Token::CloseParen);
        return expr;
    }
    TensorExpression expr = parseTermExpression();
    while (current() == Token::Plus) {
        consume(Token::Plus);
        expr += parseTermExpression();
    }
    return expr;
}

std::optional<std::string> Parser::SizeSpec::name() const {
    if (std::holds_alternative<std::string>(quantity)) {
        return std::get<std::string>(quantity);
    } else if (std::holds_alternative<std::pair<std::string, std::size_t>>(quantity)) {
        return std::get<std::pair<std::string, std::size_t>>(quantity).first;
    }
    return std::nullopt;
}

Parser::PureSpec Parser::SizeSpec::toPureSpec() const {
    if (std::holds_alternative<std::size_t>(quantity)) {
        return { std::get<std::size_t>(quantity), maxOccurrences };
    } else if (std::holds_alternative<std::pair<std::string, std::size_t>>(quantity)) {
        return { std::get<std::pair<std::string, std::size_t>>(quantity).second, maxOccurrences };
    }
    return { std::nullopt, maxOccurrences };
}

Parser::SizeSpec Parser::parseSizeSpec() {
    SizeSpec::Quantity size;
    if (current() == Token::Identifier) {
        std::string symbol = parseIdentifier();
        if (current() == Token::Equal) {
            consume(Token::Equal);
            size = std::make_pair(std::move(symbol), parseInteger());
        } else {
            size = std::move(symbol);
        }
    } else {
        size = static_cast<std::size_t>(parseInteger());
    }
    if (current() == Token::Colon) {
        consume(Token::Colon);
        return { std::move(size), parseInteger() };
    }
    return { std::move(size), std::nullopt };
}

ShapeSpecParser::SpecsDict ShapeSpecParser::ParseSpecs(const std::vector<std::string>& specs, const std::string& prefix) {
    SpecsDict names;
    std::size_t unnamed = 0;
    for (const auto& spec: specs) {
        auto result = Parser(spec).parseSizeSpec();
        auto name = result.name();
        auto [_, inserted] = names.try_emplace(name.value_or(prefix + std::to_string(unnamed++)), std::move(result));
        if (!inserted) {
            throw std::runtime_error("Duplicate size name.");
        }
    }
    return names;
}

ShapeSpecParser::NamedSpecs ShapeSpecParser::ContractSpecs(const SpecsDict& specs) {
    NamedSpecs result;
    for (auto&& [name, spec]: specs) {
        result.emplace_back(name, spec.toPureSpec());
    }
    return result;
}

ShapeSpecParser::ShapeSpecParser(const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs):
    primarySpecs(ParseSpecs(primarySpecs, "x_")),
    coefficientSpecs(ParseSpecs(coefficientSpecs, "c_"))
{}

ShapeSpecParser& ShapeSpecParser::addShape(std::string_view shape) {
    auto parsedShape = Parser(shape).parseShapeAndAttributes();
    for (const auto& sizeAndAttributes: parsedShape) {
        const auto& [size, _] = sizeAndAttributes;
        for (const auto& [var, _]: size) {
            if (!coefficientSpecs.contains(var) && !primarySpecs.contains(var)) {
                // We have to add a default spec for the name.
                primarySpecs.try_emplace(var, Parser::SizeSpec { .quantity = var, .maxOccurrences = std::nullopt });
            }
        }
    }
    return *this;
}

std::pair<ShapeSpecParser::NamedSpecs, ShapeSpecParser::NamedSpecs> ShapeSpecParser::build() const {
    return { ContractSpecs(primarySpecs), ContractSpecs(coefficientSpecs) };
}

} // namespace kas
