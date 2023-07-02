#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

std::string MapReduce::what(MapType type) {
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

std::string MapReduce::what(ReduceType type) {
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

MapReduce::MapReduce(std::size_t priority, const Size& domain, MapType mapType, ReduceType reduceType):
    priority { priority },
    domain { domain },
    mapType { mapType },
    reduceType { reduceType }
{}

std::size_t MapReduce::hash() const noexcept {
    using namespace std::string_view_literals;
    constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
    constexpr int ExpectedMaximumMapReduces = 8;
    std::size_t h = DimensionTypeHash(DimensionType::MapReduce);
    HashCombine(h, mapType);
    HashCombine(h, reduceType);
    static const auto mapReducePriorityHash = std::hash<std::string_view>{}("MapReduceIndex"sv);
    HashCombine(h, std::rotl(mapReducePriorityHash, SizeTypeWidth / ExpectedMaximumMapReduces * priority));
    HashCombine(h, domain);
    return h;
}

std::string MapReduce::whatMap() const {
    return what(mapType);
}
std::string MapReduce::whatReduce() const {
    return what(reduceType);
}

} // namespace kas
