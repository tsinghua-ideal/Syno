#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>


namespace kas {

class TensorExpression;

class Parser {
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

public:
    using Factor = std::pair<std::string, int>;

    std::string parseIdentifier();
    int parseInteger();
    Factor parseBaseAndPower();
    std::vector<Factor> parseSize();
    std::vector<std::vector<Factor>> parseCommaSeparatedSizes();

    TensorExpression parseFactorExpression();
    TensorExpression parseTermExpression();

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
        PureSpec toPureSpec() const;
        bool operator==(const SizeSpec& other) const = default;
    };
    // Unnamed SizeSpec
    struct PureSpec {
        std::optional<std::size_t> size;
        std::optional<std::size_t> maxOccurrences;
    };
    SizeSpec parseSizeSpec();
};

class ShapeSpecParser {
public:
    using SpecsDict = std::map<std::string, Parser::SizeSpec>;
    using NamedSpecs = std::vector<std::pair<std::string, Parser::PureSpec>>;
private:
    SpecsDict primarySpecs, coefficientSpecs;
    // Parse multiple `SizeSpec`s. `prefix` is for anonymous variables.
    static SpecsDict ParseSpecs(const std::vector<std::string>& specs, const std::string& prefix);
    static NamedSpecs ContractSpecs(const SpecsDict& specs);
public:
    ShapeSpecParser(const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs);
    ShapeSpecParser& addShape(std::string_view shape);
    std::pair<NamedSpecs, NamedSpecs> build() const;
};

} // namespace kas
