#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class ShapeNode final {
    friend class Sampler;
    FRIEND_TEST(search_tests, shape_node);

protected:
    // This is not complete, and Sampler needs to fill the children and unvisited count.
    ShapeNode(Shape&& shape, bool isFinal);

public:
    struct Next {
        std::unique_ptr<PrimitiveShapeOp> shapeOp;
        std::unique_ptr<ShapeNode> node = nullptr;
        Next(std::unique_ptr<PrimitiveShapeOp> shapeOp);
    };

    const Shape shape;
    // We are searching bottom-up, so the children are actually closer to the input.
    std::vector<Next> children;

    bool isFinal;
    std::size_t countUnvisited = 0;
    // For MCTS. TODO
    // std::size_t score;
};

} // namespace kas
