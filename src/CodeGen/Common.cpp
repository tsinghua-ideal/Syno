#include <cstdlib>
#include <ranges>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "KAS/CodeGen/Common.hpp"


namespace kas {

int LinkObjects(const std::filesystem::path& dir, const std::string& soName, const std::vector<std::string>& objects) {
    std::string cmdHead = fmt::format("g++ -shared -fPIC -Wl,-soname,{} -o \"{}\" -Wl,--whole-archive", soName, (dir / soName).string());
    std::string cmdBody = fmt::format("{}", fmt::join(
        objects | std::views::transform([&dir](const std::string& obj) {
            return fmt::format("\"{}\"", (dir / obj).string());
        }), " "
    ));
    std::string cmdTail = "-Wl,--no-whole-archive";
    std::string cmd = fmt::format("{} {} {}", cmdHead, cmdBody, cmdTail);
    return std::system(cmd.c_str());
}

} // namespace kas
