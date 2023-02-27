#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/Stage.hpp"


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
    FRIEND_TEST(search_tests, sampler);

protected:
    std::mt19937 rng;
    template<typename T>
    T random(T upper) {
        std::uniform_int_distribution<T> dist { 0, upper - 1 };
        return dist(rng);
    }

    BindingContext ctx;
    SampleOptions options;
    Shape inputShape;
    Shape outputShape;

    std::vector<Iterator> outputIterators;
    Interface root;

    StageStore store;

    std::vector<MapReduceOp::Base> reduces;
    std::vector<Stage> bases; // The `MapReduce`s are generated first.

    // Start from a MapReduce.
    Node visitBase(std::size_t index);

    // Visit a node along a specific path, but DO NOT visit the last node using this API! Because we do not know whether is is a TensorView.
    Node visitFromNode(const Node& node, std::span<const std::size_t> path) const;

    // Same as above, but visits a base first.
    Node visitFromRoot(const std::vector<std::size_t>& path);

    // Visit a node along a specific path, but stops at the last node, so we can verify whether it is a TensorView.
    std::pair<Node, std::size_t> visitFromNodeButStopAtLast(const Node& node, std::span<const std::size_t> path);

    // Same as above, but visits a base first.
    std::pair<Node, std::size_t> visitFromRootButStopAtLast(const std::vector<std::size_t>& path);

    std::variant<Stage *, TensorView *> visit(const std::vector<std::size_t>& path);

private:
    Sampler(std::vector<std::string> inputShape, std::vector<std::string> outputShape, std::vector<std::pair<std::string, Parser::PureSpec>> primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>> coefficientSpecs, const SampleOptions& options);
    Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const SampleOptions& options, std::map<std::string, Parser::SizeSpec> primaryVars, std::map<std::string, Parser::SizeSpec> coefficientVars);
public:
    // A specification has the following forms:
    // <literal-value> [: <max-occurrencens>]
    // <variable-name> [= <literal-value>] [: <max-occurrencens>]
    Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const SampleOptions& options);
    Sampler(const Sampler&) = delete;
    Sampler(Sampler&&) = delete;
    inline BindingContext& getBindingContext() { return ctx; }
    inline Shape& getInputShape() { return inputShape; }
    inline Shape& getOutputShape() { return outputShape; }
    inline const SampleOptions& getOptions() const { return options; }
    inline DimensionStore& getDimStore() { return store.dimStore(); }
    inline StageStore& getStageStore() { return store; }

    // The following apis can be provided for Python bindings.
    // The path is intended to visit a TensorView, but it may fail, in which case we rely on the search algorithm to penalize it.
    std::vector<std::size_t> randomPathWithPrefix(const std::vector<std::size_t>& prefix);
    bool isFinal(const std::vector<std::size_t>& path);
    std::size_t childrenCount(const std::vector<std::size_t>& path);
    std::map<std::string, std::size_t> childrenTypes(const std::vector<std::size_t>& path);
    std::string nodeString(const std::vector<std::size_t>& path);
    std::string opString(const std::vector<std::size_t>& path);
    std::string opType(const std::vector<std::size_t>& path);
    TensorView *realize(const std::vector<std::size_t>& path);
    TensorView *randomSample();
};

} // namespace kas
