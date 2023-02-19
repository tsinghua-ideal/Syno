#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/Dimension.hpp"


namespace kas {

const Size& Iterator::size() const noexcept {
    return domain;
}

const Size& Dimension::size() const noexcept {
    return inner->size();
}

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("{}@{}", size().toString(ctx), fmt::ptr(inner));
}

const Size& DimensionImpl::size() const noexcept {
    return std::visit([](auto&& x) -> const Size& { return x.size(); }, alts);
}

} // namespace kas
