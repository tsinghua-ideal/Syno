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

Representation::Transform MapReduceShapeOp::transformTensor(TensorView &tensor) const {
    KAS_ASSERT(tensor.getInterfaceIterators().size() > input);
    auto inputIt = tensor[input];
    std::unique_ptr<RepeatLikePrimitiveOp> op { new MapReduceOp { inputIt } };
    KAS_ASSERT(*size == *inputIt->getSize());
    auto outputIt = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, size);
    tensor.replaceInterface({ input }, {});
    // This is special for Reduce: we need to add it to reducedIterators
    tensor.addManipulation(Manipulation { std::move(outputIt), mapType, reduceType });
    return PrimitiveShapeOp::transformTensor(tensor);
}

std::string MapReduceShapeOp::description() const {
    std::stringstream ss;
    ss << "MapReduce " << Manipulation::what(mapType) << " " << Manipulation::what(reduceType) << " " << input;
    return ss.str();
}

MapReduceOp::MapReduceOp(std::shared_ptr<Iterator> parent):
    RepeatLikePrimitiveOp { std::move(parent) }
{}

SingleIteratorValue MapReduceOp::value(SingleIteratorValue output) const {
    return output;
}

} // namespace kas
