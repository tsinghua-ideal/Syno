#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>


namespace kas {

#define KAS_ASSERT(expr) ((expr) ? ((void)(0)) : kas::AssertImpl(__LINE__, __FILE_NAME__, #expr))

#define KAS_CRITICAL(info) kas::CriticalImpl(__LINE__, __FILE_NAME__, info)
#define KAS_UNIMPLEMENTED() kas::UnimplementedImpl(__LINE__, __FILE_NAME__)
#define KAS_UNREACHABLE() kas::UnreachableImpl(__LINE__, __FILE_NAME__)
#define KAS_WARNING(info) kas::WarningImpl(__LINE__, __FILE_NAME__, info)
#define KAS_WORKING_IN_PROGRESS() kas::WorkingInProgressImpl(__LINE__, __FILE_NAME__)

class [[maybe_unused]] ConsoleColors {
public:
    [[maybe_unused]] static constexpr std::string_view reset = "\033[0m";
    [[maybe_unused]] static constexpr std::string_view black = "\033[30m";
    [[maybe_unused]] static constexpr std::string_view red = "\033[31m";
    [[maybe_unused]] static constexpr std::string_view green = "\033[32m";
    [[maybe_unused]] static constexpr std::string_view yellow = "\033[33m";
    [[maybe_unused]] static constexpr std::string_view blue = "\033[34m";
    [[maybe_unused]] static constexpr std::string_view white = "\033[37m";
    [[maybe_unused]] static constexpr std::string_view clear = "\033[2K\r";
};

// May change to a class structure to record logs
[[nodiscard]] static inline std::ostream& Console() {
    return std::cout;
}

[[nodiscard]] static inline std::ostream& ConsoleError() {
    return std::cerr;
}

template<class CharT, class Traits>
[[nodiscard]] static inline auto& Endl(std::basic_ostream<CharT, Traits>& os) {
    return std::endl(os);
}

[[maybe_unused]] static void WarningImpl(int line, const char* file, std::string_view info) {
    Console() << ConsoleColors::green;
    Console() << "Warning (" << file << ":" << line << "):" << std::endl;
    Console() << " " << info << ConsoleColors::reset << std::endl;
}

[[noreturn]] [[maybe_unused]] static void AssertImpl(int line, const char* file, std::string_view statement) {
    ConsoleError() << ConsoleColors::red;
    ConsoleError() << "Assert error (" << file << ":" << line << "): " << statement;
    ConsoleError() << ConsoleColors::reset << std::endl;
    throw std::runtime_error("KAS assertion failed!");
}

[[noreturn]] [[maybe_unused]] static void UnimplementedImpl(int line, const char* file) {
    ConsoleError() << ConsoleColors::red;
    ConsoleError() << "Unimplemented (" << file << ":" << line << ")";
    ConsoleError() << ConsoleColors::reset << std::endl;
    std::exit(EXIT_FAILURE);
}

[[noreturn]] [[maybe_unused]] static void WorkingInProgressImpl(int line, const char* file) {
    ConsoleError() << ConsoleColors::red;
    ConsoleError() << "Working in progress (" << file << ":" << line << ")";
    ConsoleError() << ConsoleColors::reset << std::endl;
    std::exit(EXIT_FAILURE);
}

[[noreturn]] [[maybe_unused]] static void UnreachableImpl(int line, const char* file) {
    ConsoleError() << ConsoleColors::red;
    ConsoleError() << "Unreachable (" << file << ":" << line << ")";
    ConsoleError() << ConsoleColors::reset << std::endl;
    std::exit(EXIT_FAILURE);
}

[[noreturn]] [[maybe_unused]] static void CriticalImpl(int line, const char* file, std::string_view info) {
    ConsoleError() << ConsoleColors::red;
    ConsoleError() << "Error (" << file << ":" << line << "):" << std::endl;
    ConsoleError() << "  " << info << ConsoleColors::reset << std::endl;
    std::exit(EXIT_FAILURE);
}

} // namespace kas
