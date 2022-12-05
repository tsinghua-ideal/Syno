#pragma once

#include <memory>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

// Map does not change the shape of a tensor, this is just for convenience
class MapShapeOp: public PrimitiveShapeOp {
public:
    MapManipulation::Type type;
    MapShapeOp(MapManipulation::Type type);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

// Map does not affect any iterator, so no need to define a MapOp

} // namespace kas
