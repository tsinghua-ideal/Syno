#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

void SampleOptions::check() const {
    KAS_ASSERT(countPrimaryVariables > 0);
    KAS_ASSERT(countCoefficientVariables >= 0);
    KAS_ASSERT(depth >= 0);
    KAS_ASSERT(dimLowerBound >= 1);
    KAS_ASSERT(dimUpperBound >= dimLowerBound);
}

// TODO: do not DFS. Use MCTS.
void Sampler::dfsSample(std::shared_ptr<ShapeNode> node, int depth, const SampleCallback& callback) {
    if (depth >= options.depth) {
        finalize(node, callback);
        return;
    }
    auto recursion = [&](std::unique_ptr<PrimitiveShapeOp> newOp) {
        dfsSample(std::make_shared<ShapeNode>(node, std::move(newOp)), depth + 1, callback);
    };
    const auto& shape = node->shape;
    if (shape.size() < options.dimUpperBound) {
        // Try increasing dimension, by performing
        // Share^{-1}
        for (std::size_t i = 0; i < shape.size(); ++i) {
            recursion(std::make_unique<ShareShapeOp>(0, i + 1, i));
        }
        // Reduce^{-1}, TODO
        // Merge^{-1}, TODO
    }
    if (shape.size() > options.dimLowerBound) {
        // Try decreasing dimension, by performing
        // Split^{-1}
        for (std::size_t i = 0; i < shape.size(); ++i) {
            for (std::size_t j = i + 1; j < shape.size(); ++j) {
                recursion(std::make_unique<SplitShapeOp>(i, i, j));
            }
        }
        // Unfold^{-1}, TODO
    }
    // Try changing dimension size, by performing
    // Stride^{-1}, TODO
}

void Sampler::finalize(std::shared_ptr<ShapeNode> node, const SampleCallback& callback) {
    auto finalizations = FinalizeShapeOp::generate(node->shape, { .desired = inputShape });
    for (auto& f: finalizations) {
        auto finalNode = ShapeNode { node, std::move(f) };
        // TODO: get rid of this "t".
        auto tensorView = finalNode.buildTensorView(ctx.addTensor("t"));
        tensorView.setDefaultAccesses(ctx);
        tensorView.evaluateTensorAccess(ctx);
        callback(tensorView);
    }
    return;
}

BindingContext& Sampler::getBindingContext() {
    return ctx;
}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const SampleOptions& options):
    options { options },
    ctx { options.countPrimaryVariables, options.countCoefficientVariables }
{
    this->options.check();
    this->outputShape = ctx.getShapeFromNames(Shape::parseNames(outputShape));
    this->inputShape = ctx.getShapeFromNames(Shape::parseNames(inputShape));
}

void Sampler::sample(const SampleCallback& callback) {
    auto root = std::make_shared<ShapeNode>(outputShape);
    dfsSample(root, 0, callback);
}

} // namespace kas
