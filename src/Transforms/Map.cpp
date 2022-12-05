#include "KAS/Transforms/Map.hpp"
#include "KAS/Core/Manipulation.hpp"


namespace kas {

MapShapeOp::MapShapeOp(MapManipulation::Type type):
    type { type }
{}

Shape MapShapeOp::transformShapeInverse(const Shape& input) const {
    return input;
}

void MapShapeOp::transformTensor(TensorView &tensor) const {
    tensor.addManipulation(Manipulation { MapManipulation { type } });
}

} // namespace kas
