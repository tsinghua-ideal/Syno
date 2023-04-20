#include <algorithm>

#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("{}@{:x}", size().toString(ctx), hash());
}

std::ostream& operator<<(std::ostream& os, kas::DimensionType t) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", t);
    return os;
}

std::ostream& operator<<(std::ostream& os, kas::Direction d) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", d);
    return os;
}

std::ostream& operator<<(std::ostream& os, kas::Order o) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", o);
    return os;
}

} // namespace kas
