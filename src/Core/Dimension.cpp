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
    return fmt::format("{}@{:x}", size().toString(ctx), hash());
}

} // namespace kas
