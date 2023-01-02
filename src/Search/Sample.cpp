#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Transforms/Finalize.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

void SampleOptions::check() const {
    KAS_ASSERT(countPrimaryVariables > 0);
    KAS_ASSERT(countCoefficientVariables >= 0);
    KAS_ASSERT(depth >= 0);
    KAS_ASSERT(dimLowerBound >= 1);
    KAS_ASSERT(dimUpperBound >= dimLowerBound);
}

// Here, the depth means the depth of base shape.
void Sampler::addNode(const Shape& base, std::size_t depth, ShapeNode::Next& pointer) const {
    bool isFinal = pointer.shapeOp->isFinalizeOp();
    pointer.node.reset(new ShapeNode { pointer.shapeOp->transformShapeInverse(base), isFinal });
    if (isFinal) {
        return;
    }
    const Shape& shape = pointer.node->shape;
    std::vector<ShapeNode::Next> result;
    for (auto& f:
        FinalizeShapeOp::generate(shape, { .desired = inputShape })
    ) result.emplace_back(std::move(f));
    if (depth < options.depth) {
        // Try increasing dimension, by performing
        // Share^{-1}
        for (auto& s:
            ShareShapeOp::generate(shape, { .dimUpperBound = options.dimUpperBound })
        ) result.emplace_back(std::move(s));
        // Reduce^{-1}, TODO
        // Merge^{-1}, TODO
        // Try decreasing dimension, by performing
        // Split^{-1}
        for (auto& s:
            SplitShapeOp::generate(shape, { .dimLowerBound = options.dimLowerBound })
        ) result.emplace_back(std::move(s));
        // Unfold^{-1}, TODO
        // Try changing dimension size, by performing
        // Stride^{-1}, TODO
        // Or do not change the shape at all, by performing
        // Map^{-1}, TODO
        // Shift^{-1}, TODO
    }
    pointer.node->countUnvisited = result.size();
    pointer.node->children = std::move(result);
}

BindingContext& Sampler::getBindingContext() {
    return ctx;
}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const SampleOptions& options):
    options { options },
    ctx { options.countPrimaryVariables, options.countCoefficientVariables },
    root { std::make_unique<IdentityShapeOp>() }
{
    this->options.check();
    path.reserve(options.depth + 1); // An addition layer for FinalizeShapeOp.
    this->outputShape = ctx.getShapeFromNames(Shape::parseNames(outputShape));
    this->inputShape = ctx.getShapeFromNames(Shape::parseNames(inputShape));
}

TensorView Sampler::sample() {
    path.clear();
    std::random_device rd;
    std::mt19937 rng { rd() };
    if (root.node == nullptr) {
        addNode(outputShape, 0, root);
    }
    const auto recursion = [this, &rng](const auto& self, ShapeNode& current, int depth) -> std::optional<TensorView> {
        if (current.isFinal) {
            // TODO: get rid of this "t".
            auto tensorId = ctx.addTensor("t");
            // First build a pure tensor of input shape.
            auto tensor = std::make_shared<PureTensor>(tensorId, current.shape);
            // Then start to build a view of this tensor.
            return { TensorView { tensor } };
        } else {
            // Here we just randomly iterate. In MCTS, use UCB. TODO
            std::vector<std::size_t> childrenIds(current.children.size(), 0);
            std::iota(childrenIds.begin(), childrenIds.end(), 0);
            std::shuffle(childrenIds.begin(), childrenIds.end(), rng);
            for (std::size_t i: childrenIds) {
                auto& child = current.children[i];
                if (child.node == nullptr) {
                    addNode(current.shape, depth, child);
                }
                if (auto result = self(self, *child.node, depth + 1)) {
                    path.emplace_back(&current, &child);
                    child.shapeOp->transformTensor(result.value());
                    return std::move(result);
                }
            }
        }
        return std::nullopt;
    };
    if (auto result = recursion(recursion, *root.node, 0)) {
        auto& tensorView = result.value();
        tensorView.finishConstruction();
        tensorView.setDefaultAccesses(ctx);
        tensorView.evaluateTensorAccess(ctx);
        return std::move(tensorView);
    } else {
        KAS_CRITICAL("Cannot sample a tensor.");
    }
}

} // namespace kas
