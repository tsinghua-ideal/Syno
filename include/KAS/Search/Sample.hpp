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
    std::size_t maximumTensors = 2;
    void check() const;
};

class Sampler final {
    std::mt19937 rng;
    template<typename T>
    T random(T upper) {
        std::uniform_int_distribution<T> dist { 0, upper - 1 };
        return dist(rng);
    }

    BindingContext ctx;
    SampleOptions options;
    Colors::Options colorOptions;
    Shape inputShape;
    Shape outputShape;

    std::vector<Iterator> outputIterators;
    Interface root;

    StageStore store;

    std::vector<MapReduceOp::Base> reduces;
    std::vector<Stage> originalBases;
    std::vector<std::pair<std::size_t, Stage *>> bases; // The `MapReduce`s are generated first.

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

    inline std::size_t getBaseCount() const { return bases.size(); }
    std::vector<Next> getNextBases() const;
    Stage *getBase(std::size_t key) const;

    // The following APIs can be provided for Python bindings.
    Node visit(const std::vector<Next>& path);
    // The path is intended to visit a TensorView, but it may fail, in which case we rely on the search algorithm to penalize it.
    Node randomNodeWithPrefix(const std::vector<Next>& prefix);
};

} // namespace kas
