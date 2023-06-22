#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>


namespace kas {

class TensorExpression;

class Parser {
protected:
    enum class Token {
        Identifier, Integer,
        Plus, Times, Power, Comma, Colon, Equal,
        OpenParen, CloseParen,
        OpenBracket, CloseBracket,
        End,
    };
    static std::string_view what(Token tok);

    std::vector<std::pair<Token, std::string>> tokens;

    std::size_t pointer = 0;
    Token current() const;
    Token peek() const;
    std::string consume(Token token);

    std::string parseIdentifier();
    int parseInteger();
    using Factor = std::pair<std::string, int>;
    Factor parseBaseAndPower();
    std::vector<Factor> parseSize();
    std::vector<std::vector<Factor>> parseCommaSeparatedSizes();

    TensorExpression parseFactorExpression();
    TensorExpression parseTermExpression();

public:
    Parser(std::string_view buffer);
    std::vector<std::vector<Factor>> parseShape();
    std::vector<std::vector<std::vector<Factor>>> parseShapes();
    TensorExpression parseTensorExpression();

    struct PureSpec;
    // For size specifications.
    // `SizeSpec` ::= `Size` (`:` `int`)?.
    // The integer is the maximum occurrences of a size.
    // `Size` ::= `int` | `id` | `id` `=` `int`.
    // A size can be anonymous (in which case it is a numeric constant), or named with optional specified value.
    struct SizeSpec {
        using Quantity = std::variant<std::string, std::size_t, std::pair<std::string, std::size_t>>;
        Quantity quantity;
        std::optional<std::size_t> maxOccurrences;
        std::optional<std::string> name() const;
        PureSpec toPureSpec() const &;
        PureSpec toPureSpec() &&;
        bool operator==(const SizeSpec& other) const = default;
    };
    // Unnamed SizeSpec
    struct PureSpec {
        std::optional<std::size_t> size;
        std::optional<std::size_t> maxOccurrences;
    };
    SizeSpec parseSizeSpec();
};

} // namespace kas
