
#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/Dimension.hpp"


namespace kas {

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("{}@{}", size().toString(ctx), fmt::ptr(inner));
}

} // namespace kas
