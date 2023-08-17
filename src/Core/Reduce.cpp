#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

std::string Reduce::what(MapType type) {
    switch (type) {
        case MapType::Absolute: return "Absolute";
        case MapType::ArcTan:   return "ArcTan";
        case MapType::Exp:      return "Exp";
        case MapType::Log:      return "Log";
        case MapType::Identity: return "Identity";
        case MapType::Inverse:  return "Inverse";
        case MapType::Negative: return "Negative";
        case MapType::ReLU:     return "ReLU";
        case MapType::Sigmoid:  return "Sigmoid";
        case MapType::Sign:     return "Sign";
        case MapType::MapTypeCount: break;
    }
    KAS_UNREACHABLE();
}

std::string Reduce::what(ReduceType type) {
    switch (type) {
        case ReduceType::Sum:     return "Sum";
        case ReduceType::Max:     return "Max";
        case ReduceType::Mean:    return "Mean";
        case ReduceType::Min:     return "Min";
        case ReduceType::Product: return "Product";
        case ReduceType::ReduceTypeCount: break;
    }
    KAS_UNREACHABLE();
}

Reduce::Reduce(std::size_t priority, const Size& domain, MapType mapType, ReduceType reduceType):
    priority { priority },
    domain { domain },
    mapType { mapType },
    reduceType { reduceType }
{}

std::size_t Reduce::hash() const noexcept {
    using namespace std::string_view_literals;
    constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
    constexpr int ExpectedMaximumReduces = 8;
    std::size_t h = DimensionTypeHash(DimensionType::Reduce);
    HashCombine(h, mapType);
    HashCombine(h, reduceType);
    static const auto reducePriorityHash = std::hash<std::string_view>{}("ReduceIndex"sv);
    HashCombine(h, std::rotl(reducePriorityHash, SizeTypeWidth / ExpectedMaximumReduces * priority));
    HashCombine(h, domain);
    return h;
}

std::string Reduce::whatMap() const {
    return what(mapType);
}
std::string Reduce::whatReduce() const {
    return what(reduceType);
}

} // namespace kas
