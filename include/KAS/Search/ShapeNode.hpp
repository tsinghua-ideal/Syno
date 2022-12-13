#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class ShapeNode {
public:
    const Shape shape;
    // We are searching bottom-up, so the previous node is actually the child.
    std::shared_ptr<ShapeNode> child;
    // Ops like Shift, Map do not change shapes, so when searching they can be ignored first to match shape, and later inserted in-between.
    std::unique_ptr<PrimitiveShapeOp> shapeOp;

    template<typename T>
    ShapeNode(T&& shape): shape { std::forward<T>(shape) }, child { nullptr }, shapeOp { nullptr } {}
    ShapeNode(std::shared_ptr<ShapeNode> child, std::unique_ptr<PrimitiveShapeOp> shapeOp);

    // Create a TensorView, using the current ShapeNode as the input tensor, the bottom-most shape as the output tensor.
    TensorView buildTensorView(std::size_t tensorId) const;
};

} // namespace kas
