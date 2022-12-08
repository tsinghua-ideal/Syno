#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Transforms.hpp"


namespace kas {

SampleOptions::SampleOptions(int countPrimaryVariables, int countCoefficientVariables, int depth, int dimLowerBound, int dimUpperBound):
    countPrimaryVariables { countPrimaryVariables },
    countCoefficientVariables { countCoefficientVariables },
    depth { depth },
    dimLowerBound { dimLowerBound },
    dimUpperBound { dimUpperBound }
{
    KAS_ASSERT(countPrimaryVariables > 0);
    KAS_ASSERT(countCoefficientVariables >= 0);
    KAS_ASSERT(depth >= 0);
    KAS_ASSERT(dimLowerBound >= 1);
    KAS_ASSERT(dimUpperBound >= dimLowerBound);
}

void Sampler::dfsSample(std::shared_ptr<ShapeNode> node, int depth, const SampleCallback& callback) const {
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
        for (int i = 0; i < shape.size(); ++i) {
            recursion(std::make_unique<ShareShapeOp>(0, i + 1, i));
        }
        // Reduce^{-1}, TODO
        // Merge^{-1}, TODO
    }
    if (shape.size() > options.dimLowerBound) {
        // Try decreasing dimension, by performing
        // Split^{-1}
        for (int i = 0; i < shape.size(); ++i) {
            for (int j = i + 1; j < shape.size(); ++j) {
                recursion(std::make_unique<SplitShapeOp>(i, i, j));
            }
        }
        // Unfold^{-1}, TODO
    }
    // Try changing dimension size, by performing
    // Stride^{-1}, TODO
}

void Sampler::finalize(std::shared_ptr<ShapeNode> node, const SampleCallback& callback) const {
    // Here we need to match the shape.
    std::vector<int> remap(inputShape.size(), -1);
    std::set<int> used;
    for (int i = 0; i < inputShape.size(); ++i) {
        auto coarseMatchingSizes = node->shape.findSize(*inputShape[i]);
        std::vector<int> matchingSizes;
        std::copy_if(coarseMatchingSizes.begin(), coarseMatchingSizes.end(), std::back_inserter(matchingSizes), [&](int i) { return !used.contains(i); });
        if (!matchingSizes.empty()) {
            auto candidate = matchingSizes[0]; // Just add some randomness. TODO
            used.insert(candidate);
            remap[i] = candidate;
        } else {
            auto coarseMultipleSizes = node->shape.findMultipleOfSize(*inputShape[i], ctx);
            std::vector<int> multipleSizes;
            std::copy_if(coarseMultipleSizes.begin(), coarseMultipleSizes.end(), std::back_inserter(multipleSizes), [&](int i) { return !used.contains(i); });
            if (!multipleSizes.empty()) {
                auto candidate = multipleSizes[0];
                remap[i] = candidate;
                node = std::make_shared<ShapeNode>(node, std::make_unique<MergeShapeOp>(node->shape.size(), candidate, candidate, inputShape[i]));
            } else {
                return;
            }
        }
    }
    KAS_ASSERT(std::all_of(remap.begin(), remap.end(), [](int i) { return i != -1; }));
    auto tensorView = node->buildTensorView(remap);
    tensorView.evaluateTensorAccess(ctx);
    callback(tensorView);
    return;
}

BindingContext& Sampler::getBindingContext() {
    return ctx;
}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const SampleOptions& options):
    options { options },
    ctx { options.countPrimaryVariables, options.countCoefficientVariables }
{
    this->outputShape = ctx.getShapeFromNames(Shape::parseNames(outputShape));
    this->inputShape = ctx.getShapeFromNames(Shape::parseNames(inputShape));
}

void Sampler::sample(const SampleCallback& callback) const {
    auto root = std::make_shared<ShapeNode>(outputShape);
    dfsSample(root, 0, callback);
}

} // namespace kas
