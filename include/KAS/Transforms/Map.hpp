#pragma once

#include <memory>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

// Map does not change the shape of a tensor, this is just for convenience
class MapShapeOp final: public PrimitiveShapeOp {
public:
    MapManipulation::Type type;
    MapShapeOp(MapManipulation::Type type);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;

    static std::vector<std::unique_ptr<MapShapeOp>> generate(const Shape& outputShape);
};

// Map does not affect any iterator, so no need to define a MapOp

} // namespace kas
