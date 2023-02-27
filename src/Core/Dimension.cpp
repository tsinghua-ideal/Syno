
#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/Dimension.hpp"


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
}

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("{}@{}", size().toString(ctx), fmt::ptr(inner));
}

} // namespace kas
