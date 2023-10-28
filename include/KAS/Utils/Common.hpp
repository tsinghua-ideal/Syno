#pragma once

#include <iostream>

#include <fmt/core.h>

#define KAS_ASSERT(expr, ...) ((expr) ? ((void)(0)) : ::kas::detail::FormatAndLogAndThrow<::kas::detail::ConsoleType::Error, ::kas::detail::ColorType::Red>(__FILE__, __LINE__, "Assertion failed: " #expr __VA_OPT__(,) __VA_ARGS__))

#define KAS_DEBUG(...) ::kas::detail::FormatAndLog<::kas::detail::ConsoleType::Error, ::kas::detail::ColorType::Green>(__FILE__, __LINE__, "Debug" __VA_OPT__(,) __VA_ARGS__)
#define KAS_WARNING(...) ::kas::detail::FormatAndLog<::kas::detail::ConsoleType::Out, ::kas::detail::ColorType::Green>(__FILE__, __LINE__, "Warning" __VA_OPT__(,) __VA_ARGS__)
#define KAS_UNREACHABLE(...) ::kas::detail::FormatAndLogAndThrow<::kas::detail::ConsoleType::Error, ::kas::detail::ColorType::Red>(__FILE__, __LINE__, "Unreachable" __VA_OPT__(,) __VA_ARGS__)
#define KAS_CRITICAL(...) ::kas::detail::FormatAndLogAndThrow<::kas::detail::ConsoleType::Error, ::kas::detail::ColorType::Red>(__FILE__, __LINE__, "Error" __VA_OPT__(,) __VA_ARGS__)
#define KAS_UNIMPLEMENTED(...) ::kas::detail::FormatAndLogAndThrow<::kas::detail::ConsoleType::Error, ::kas::detail::ColorType::Red>(__FILE__, __LINE__, "Unimplemented" __VA_OPT__(,) __VA_ARGS__)

namespace kas {

namespace detail {

enum class ColorType {
    Reset,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    White,
    Clear,
};
template<ColorType>
[[maybe_unused]] constexpr inline const char *ConsoleColor = "\033[0m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::Reset> = "\033[0m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::Black> = "\033[30m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::Red> = "\033[31m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::Green> = "\033[32m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::Yellow> = "\033[33m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::Blue> = "\033[34m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::White> = "\033[37m";
template<> [[maybe_unused]] constexpr inline const char *ConsoleColor<ColorType::Clear> = "\033[2K\r";

enum class ConsoleType {
    Out,
    Error,
};
// May change to a class structure to record logs
template<ConsoleType consoleType>
[[maybe_unused]] constexpr inline std::ostream& Console = std::cout;
template<> [[maybe_unused]] constexpr inline std::ostream& Console<ConsoleType::Out> = std::cout;
template<> [[maybe_unused]] constexpr inline std::ostream& Console<ConsoleType::Error> = std::cerr;

template<ConsoleType consoleType, ColorType consoleColor, typename... Args>
inline void FormatAndLog(const char *fileName, int line, const char *caption, fmt::format_string<Args...> format, Args&&... args) {
    constexpr auto& console = Console<consoleType>;
    console << ConsoleColor<consoleColor>;
    console << caption << fmt::format(" ({}:{}): ", fileName, line) << std::endl;
    console << fmt::format(format, std::forward<Args>(args)...) << std::endl;
    console << ConsoleColor<ColorType::Reset>;
}

template<ConsoleType consoleType, ColorType consoleColor>
inline void FormatAndLog(const char *fileName, int line, const char *caption) {
    constexpr auto& console = Console<consoleType>;
    console << ConsoleColor<consoleColor>;
    console << caption << fmt::format(" ({}:{})", fileName, line) << std::endl;
    console << ConsoleColor<ColorType::Reset>;
}

template<ConsoleType consoleType, ColorType consoleColor, typename... Args>
[[noreturn]] inline void FormatAndLogAndThrow(const char *fileName, int line, const char *caption, fmt::format_string<Args...> format, Args&&... args) {
    FormatAndLog<consoleType, consoleColor>(fileName, line, caption, format, std::forward<Args>(args)...);
    throw std::runtime_error(caption);
}

template<ConsoleType consoleType, ColorType consoleColor>
[[noreturn]] inline void FormatAndLogAndThrow(const char *fileName, int line, const char *caption) {
    FormatAndLog<consoleType, consoleColor>(fileName, line, caption);
    throw std::runtime_error(caption);
}

template<typename F>
struct Deferred {
    Deferred(F&& f) : f(std::forward<F>(f)) {}
    ~Deferred() { f(); }
    F f;
};

#define KAS_DEFER_CONCAT_IMPL(x, y) x##y
#define KAS_DEFER_CONCAT(x, y) KAS_DEFER_CONCAT_IMPL(x, y)
#define KAS_DEFER ::kas::detail::Deferred KAS_DEFER_CONCAT(_kas_deferred_, __LINE__) = [&]()

} // namespace detail

template<typename... Ts>
struct Match: Ts... { using Ts::operator()...; };
template<typename... Ts>
Match(Ts...) -> Match<Ts...>;

constexpr std::size_t operator""_uz(unsigned long long int x) { return x; }

struct Common {
    static constexpr std::size_t MemoryPoolSize = 65536;
};

} // namespace kas
