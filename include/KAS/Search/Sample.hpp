#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/Representation.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/ShapeNode.hpp"


namespace kas {

struct SampleOptions {
public:
    using Seed = std::mt19937::result_type;
    Seed seed = 42;
    std::size_t countPrimaryVariables = 5;
    std::size_t countCoefficientVariables = 5;
    int depth = 4;
    int dimLowerBound = 1;
    int dimUpperBound = 8;
    void check() const;
};

class Sampler final {
    FRIEND_TEST(search_tests, sample);

protected:
    std::mt19937 rng;

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
    // Returns a random path from the current node to a final node.
    std::vector<std::size_t> randomSubPathFromNode(ShapeNode& node, std::size_t depth);

    // The following apis can be provided for Python bindings.
    std::vector<std::size_t> randomPathWithPrefix(std::vector<std::size_t> prefix);
    bool isFinal(std::vector<std::size_t> path);
    std::size_t countChildren(std::vector<std::size_t> path);
    std::tuple<TensorView, std::shared_ptr<CodeGenContext>, Representation> realize(std::vector<std::size_t> path);
    std::tuple<TensorView, std::shared_ptr<CodeGenContext>, Representation> randomSample();
};

} // namespace kas
