#pragma once

#include <memory>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MapReduceShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t input;
    std::shared_ptr<Size> size;
    Manipulation::MapType mapType;
    Manipulation::ReduceType reduceType;
    MapReduceShapeOp(std::size_t input, std::shared_ptr<Size> size, Manipulation::MapType mapType, Manipulation::ReduceType reduceType);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;
    std::string description() const override;

    static std::vector<std::unique_ptr<MapReduceShapeOp>> generate(const Shape& outputShape);
};

class MapReduceOp: public RepeatLikePrimitiveOp {
public:
    MapReduceOp(std::shared_ptr<Iterator> parent);
    SingleIteratorValue value(SingleIteratorValue output) const override;
};

} // namespace kas
