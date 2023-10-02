#pragma once

#include <functional>
#include <sstream>

#include <fmt/format.h>


namespace kas {

class PythonCodePrinter {
    std::ostringstream& oss;
    bool isNewLine = true;
    std::size_t indentLevel;
    void writeIndent() {
        for (std::size_t i = 0; i < indentLevel; ++i) oss << '\t';
    }
public:
    template<typename F>
    void indent(F&& f) {
        const std::size_t oldIndentLevel = indentLevel++;
        std::invoke(f);
        indentLevel = oldIndentLevel;
    };
    template<bool Comma = false, typename F>
    void parens(F&& f) {
        writeLn("(");
        indent(std::forward<F>(f));
        if constexpr (Comma) writeLn("),");
        else writeLn(")");
    };
    PythonCodePrinter(std::ostringstream& oss, std::size_t indentLevel):
        oss { oss },  indentLevel { indentLevel } {}
    template<typename... Args>
    void write(fmt::format_string<Args...> format, Args&&... args) {
        if (isNewLine) {
            writeIndent();
            isNewLine = false;
        }
        fmt::format_to(std::ostreambuf_iterator(oss), format, std::forward<Args>(args)...);
    }
    void writeLn() {
        oss << "\n";
        isNewLine = true;
    }
    template<typename... Args>
    void writeLn(fmt::format_string<Args...> format, Args&&... args) {
        write(format, std::forward<Args>(args)...);
        writeLn();
    }
};

} // namespace kas
