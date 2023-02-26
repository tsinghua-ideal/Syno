#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"


namespace kas {

std::string MapReduceOp::what(MapType type) {
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

std::string MapReduceOp::what(ReduceType type) {
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

std::string MapReduceOp::whatMap() const {
    return what(mapType);
}
std::string MapReduceOp::whatReduce() const {
    return what(reduceType);
}

std::string MapReduceOp::what() const {
    return what(mapType) + "+" + what(reduceType);
}

std::vector<Interface> MapReduceOp::GenerateLastLevelMapReduces(const Shape& outputShape, GenerateOptions options) {
    const BindingContext& ctx = options.ctx;
    auto primaryMeta = ctx.getPrimaryMetadata();
    auto coefficientMeta = ctx.getCoefficientMetadata();
    Size totalSize = outputShape.totalSize();
    auto primary = totalSize.getPrimary();
    auto coefficient = totalSize.getCoefficient();
    const std::size_t primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
    std::map<std::size_t, std::size_t> primaryAllowance;
    for (std::size_t i = 0; i < primaryCount; ++i) {
        // Observe that in the sampling process, the primary variables are generated only by MapReduce. So we can limit it with maximumOccurrence.
        if (static_cast<std::size_t>(primary[i]) < primaryMeta[i].maximumOccurrence) {
            primaryAllowance[i] = primaryMeta[i].maximumOccurrence - static_cast<std::size_t>(primary[i]);
        }
    }
    std::vector<std::size_t> coefficientAllowance;
    for (std::size_t i = 0; i < coefficientCount; ++i) {
        // But we interpret maximumOccurrence for coefficient variables in a different way. If a single coefficient appears for too many times in a tensor, do not mess with it.
        if (std::abs(coefficient[i]) < coefficientMeta[i].maximumOccurrence) {
            coefficientAllowance.emplace_back(i);
        }
    }
    // Place at the end.
    std::size_t pos = outputShape.size();
    std::vector<Interface> res;
    for (const auto& coefficientId: coefficientAllowance) {
        auto c = ctx.getSingleCoefficientVariableSize(coefficientId);
        // For simplicity, we only use Identity and Sum. TODO: add more.
        res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, c, Manipulation::MapType::Identity, Manipulation::ReduceType::Sum));
        for (const auto& [primaryId, allowance]: primaryAllowance) {
            auto p = ctx.getSinglePrimaryVariableSize(primaryId);
            res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, p, Manipulation::MapType::Identity, Manipulation::ReduceType::Sum));
            res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, *c * *p, Manipulation::MapType::Identity, Manipulation::ReduceType::Sum));
            res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, *p / *c, Manipulation::MapType::Identity, Manipulation::ReduceType::Sum));
        }
    }
    return res;
}

} // namespace kas
