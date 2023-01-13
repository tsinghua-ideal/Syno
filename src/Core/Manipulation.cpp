#include <string>
#include <utility>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string_view Manipulation::what(MapType type) {
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

std::string_view Manipulation::what(ReduceType type) {
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

Manipulation::Manipulation(std::shared_ptr<Iterator> iterator, MapType mapType, ReduceType reduceType):
    iterator { std::move(iterator) },
    mapType { mapType },
    reduceType { reduceType }
{}

std::shared_ptr<Iterator> Manipulation::getIterator() const {
    return iterator;
}

std::string Manipulation::whatMap() const {
    return std::string(what(mapType));
}
std::string Manipulation::whatReduce() const {
    return std::string(what(reduceType));
}

std::string Manipulation::what() const {
    return std::string(what(mapType)) + "+" + std::string(what(reduceType));
}

} // namespace kas
