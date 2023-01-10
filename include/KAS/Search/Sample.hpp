#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>

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
    FRIEND_TEST(search_tests, sample);

protected:
    Shape inputShape;
    Shape outputShape;
    SampleOptions options;
    BindingContext ctx;

    ShapeNode::Next root;

    void addNode(const Shape& base, std::size_t depth, ShapeNode::Next& pointer) const;

    // Visit a node along a specific path.
    ShapeNode& visit(std::vector<std::size_t> path);

public:
    BindingContext& getBindingContext();
    Sampler(std::string_view inputShape, std::string_view outputShape, const SampleOptions& options);
    // Sample a TensorView with given shape, returns along with the path to the node.
    std::pair<TensorView, std::vector<std::size_t>> randomSample();

    // The following apis can be provided for Python bindings.
    bool isFinal(std::vector<std::size_t> path);
    std::size_t countChildren(std::vector<std::size_t> path);
    std::pair<TensorView, std::shared_ptr<CodeGenContext>> realize(std::vector<std::size_t> path);
};

} // namespace kas
