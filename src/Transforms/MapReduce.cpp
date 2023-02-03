#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <utility>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

MapReduceShapeOp::MapReduceShapeOp(std::size_t input, std::shared_ptr<Size> size, Manipulation::MapType mapType, Manipulation::ReduceType reduceType):
    input { input },
    size { std::move(size) },
    mapType { mapType },
    reduceType { reduceType }
{}

Shape MapReduceShapeOp::transformShapeInverse(const Shape& output) const {
    return output.replace({}, { std::make_pair(input, size) });
}

void MapReduceShapeOp::transformTensor(TensorView &tensor) const {
    KAS_ASSERT(tensor.getInterfaceIterators().size() > input);
    auto inputIt = tensor[input];
    std::unique_ptr<RepeatLikePrimitiveOp> op { new MapReduceOp { inputIt } };
    KAS_ASSERT(*size == *inputIt->getSize());
    auto outputIt = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, size);
    tensor.replaceInterface({ input }, {});
    // This is special for Reduce: we need to add it to reducedIterators
    tensor.addManipulation(Manipulation { std::move(outputIt), mapType, reduceType });
}

std::string MapReduceShapeOp::description() const {
    std::stringstream ss;
    ss << "MapReduce " << Manipulation::what(mapType) << " " << Manipulation::what(reduceType) << " " << input;
    return ss.str();
}

std::vector<std::unique_ptr<MapReduceShapeOp>> MapReduceShapeOp::generate(const Shape& outputShape, GenerateOptions options) {
    const auto& ctx = options.ctx;
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
    std::vector<std::unique_ptr<MapReduceShapeOp>> res;
    for (const auto& coefficientId: coefficientAllowance) {
        auto c = ctx.getSingleCoefficientVariableSize(coefficientId);
        // For simplicity, we only use ReLU and Sum. TODO: add more.
        res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, c, Manipulation::MapType::ReLU, Manipulation::ReduceType::Sum));
        for (const auto& [primaryId, allowance]: primaryAllowance) {
            auto p = ctx.getSinglePrimaryVariableSize(primaryId);
            res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, p, Manipulation::MapType::ReLU, Manipulation::ReduceType::Sum));
            res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, *c * *p, Manipulation::MapType::ReLU, Manipulation::ReduceType::Sum));
            res.emplace_back(std::make_unique<MapReduceShapeOp>(pos, *p / *c, Manipulation::MapType::ReLU, Manipulation::ReduceType::Sum));
        }
    }
    return res;
}

MapReduceOp::MapReduceOp(std::shared_ptr<Iterator> parent):
    RepeatLikePrimitiveOp { std::move(parent) }
{}

SingleIteratorValue MapReduceOp::value(SingleIteratorValue output) const {
    return output;
}

} // namespace kas
