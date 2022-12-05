#include "KAS/Core/Manipulation.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string ReduceManipulation::what(Type type) {
    switch (type) {
        case Type::Sum:     return "Sum";
        case Type::Max:     return "Max";
        case Type::Mean:    return "Mean";
        case Type::Min:     return "Min";
        case Type::Product: return "Product";
    }
    KAS_UNREACHABLE();
    return {};
}

ReduceManipulation::ReduceManipulation(std::shared_ptr<Iterator> iterator, Type type):
    iterator { std::move(iterator) },
    type { type }
{}

std::string ReduceManipulation::what() const {
    return what(type);
}

std::string MapManipulation::what(Type type) {
    switch (type) {
        case Type::Absolute: return "Absolute";
        case Type::ArcTan:   return "ArcTan";
        case Type::Exp:      return "Exp";
        case Type::Log:      return "Log";
        case Type::Identity: return "Identity";
        case Type::Inverse:  return "Inverse";
        case Type::Mask:     return "Mask";
        case Type::Negative: return "Negative";
        case Type::ReLU:     return "ReLU";
        case Type::Sigmoid:  return "Sigmoid";
        case Type::Sign:     return "Sign";
    }
    KAS_UNREACHABLE();
    return {};
}

MapManipulation::MapManipulation(Type type):
    type { type }
{}

std::string MapManipulation::what() const {
    return what(type);
}

} // namespace kas
