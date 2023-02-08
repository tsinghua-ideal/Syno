#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/ShapeNode.hpp"


namespace kas {

struct SampleOptions {
public:
    using Seed = std::mt19937::result_type;
    Seed seed = 42;
    std::size_t depth = 4;
    std::size_t dimLowerBound = 1;
    std::size_t dimUpperBound = 8;
    void check() const;
};

class Sampler final {
    FRIEND_TEST(search_tests, sample);

protected:
    std::mt19937 rng;

    BindingContext ctx;
    SampleOptions options;
    const Shape inputShape;
    const Shape outputShape;

    ShapeNode::Next root;

    void addNode(const Shape& base, std::size_t depth, ShapeNode::Next& pointer) const;

    // Visit a node along a specific path.
    ShapeNode& visit(const std::vector<std::size_t>& path);
    // Visit the pointer that is pointing to `visit(path)`.
    ShapeNode::Next& visitPointer(const std::vector<std::size_t>& path);

private:
    Sampler(std::vector<std::string> inputShape, std::vector<std::string> outputShape, std::vector<std::pair<std::string, Parser::PureSpec>> primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>> coefficientSpecs, const SampleOptions& options);
    Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const SampleOptions& options, std::map<std::string, Parser::SizeSpec> primaryVars, std::map<std::string, Parser::SizeSpec> coefficientVars);
public:
    // A specification has the following forms:
    // <literal-value> [: <max-occurrencens>]
    // <variable-name> [= <literal-value>] [: <max-occurrencens>]
    Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const SampleOptions& options);
    BindingContext& getBindingContext();
    // Returns a random path from the current node to a final node.
    std::vector<std::size_t> randomSubPathFromNode(ShapeNode& node, std::size_t depth);

    // The following apis can be provided for Python bindings.
    std::vector<std::size_t> randomPathWithPrefix(std::vector<std::size_t> prefix);
    bool isFinal(const std::vector<std::size_t>& path);
    std::size_t childrenCount(const std::vector<std::size_t>& path);
    std::map<std::string, std::size_t> childrenTypes(const std::vector<std::size_t>& path);
    std::string nodeString(const std::vector<std::size_t>& path);
    std::string opString(const std::vector<std::size_t>& path);
    std::string opType(const std::vector<std::size_t>& path);
    std::tuple<TensorView, std::shared_ptr<CodeGenContext>> realize(const std::vector<std::size_t>& path);
    std::tuple<TensorView, std::shared_ptr<CodeGenContext>> randomSample();
};

} // namespace kas
