#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/ShapeNode.hpp"


namespace kas {

struct SampleOptions {
public:
    int countPrimaryVariables;
    int countCoefficientVariables;
    int depth;
    int dimLowerBound;
    int dimUpperBound;
    SampleOptions(int countPrimaryVariables, int countCoefficientVariables, int depth, int dimLowerBound, int dimUpperBound);
};

using SampleCallback = std::function<void(TensorView)>;

class Sampler final {
protected:
    Shape inputShape;
    Shape outputShape;
    SampleOptions options;
    BindingContext ctx;
    // Sample a tensor by DFS.
    void dfsSample(std::shared_ptr<ShapeNode> node, int depth, const SampleCallback& callback);
    // On the leaf node, we are about to construct a TensorView. But the shape maybe not compatible with the given input. So we need to apply some invertible transforms to obtain a compatible shape.
    void finalize(std::shared_ptr<ShapeNode> node, const SampleCallback& callback);
public:
    BindingContext& getBindingContext();
    Sampler(std::string_view inputShape, std::string_view outputShape, const SampleOptions& options);
    // Sample a TensorView with given shape.
    void sample(const SampleCallback& callback);
};

} // namespace kas
