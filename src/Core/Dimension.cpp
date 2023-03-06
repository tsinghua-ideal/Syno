#include <algorithm>

#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string DimensionTypeDescription(DimensionType ty) {
    switch (ty) {
    case DimensionType::Shift:      return "Shift";
    case DimensionType::Stride:     return "Stride";
    case DimensionType::Split:      return "Split";
    case DimensionType::Unfold:     return "Unfold";
    case DimensionType::Merge:      return "Merge";
    case DimensionType::Share:      return "Share";
    case DimensionType::Iterator:   return "Iterator";
    case DimensionType::MapReduce:  return "MapReduce";
    }
    KAS_UNREACHABLE();
}

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("{}@{}", size().toString(ctx), fmt::ptr(inner));
}

Interface::const_iterator Dimension::findIn(const Interface& interface) const {
    auto it = std::lower_bound(interface.begin(), interface.end(), *this);
    // TODO: after switching to computed hash, handle collisions.
    return it;
}

} // namespace kas
