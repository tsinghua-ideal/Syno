#pragma once

#include <cstddef>
#include <string>
#include <vector>


namespace kas {

class Parser {
protected:
    enum class Token {
        Identifier, Integer,
        Times, Power, Comma,
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

public:
    Parser(std::string_view buffer);
    std::vector<std::vector<Factor>> parseShape();
    std::vector<std::vector<std::vector<Factor>>> parseShapes();
};

} // namespace kas
