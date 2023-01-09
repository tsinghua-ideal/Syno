#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/ShapeNode.hpp"


namespace kas {

struct SampleOptions {
public:
    std::size_t countPrimaryVariables = 5;
    std::size_t countCoefficientVariables = 5;
    int depth = 4;
    int dimLowerBound = 1;
    int dimUpperBound = 8;
    SampleOptions() = default;
    void check() const;
};

class Sampler final {
protected:
    Shape inputShape;
    Shape outputShape;
    SampleOptions options;
    BindingContext ctx;

    ShapeNode::Next root;
    // The path from the leaf to the root of the search tree. This can be used to update scores in MCTS.
    std::vector<std::pair<ShapeNode*, ShapeNode::Next*>> path;

    void addNode(const Shape& base, std::size_t depth, ShapeNode::Next& pointer) const;

public:
    BindingContext& getBindingContext();
    Sampler(std::string_view inputShape, std::string_view outputShape, const SampleOptions& options);
    // Sample a TensorView with given shape.
    TensorView sample();
};

} // namespace kas
