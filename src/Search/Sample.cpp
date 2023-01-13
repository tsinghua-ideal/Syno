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
        // MapReduce^{-1}, TODO
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
        // Shift^{-1}, TODO
    }
    pointer.node->children = std::move(result);
}

ShapeNode& Sampler::visit(std::vector<std::size_t> path) {
    if (root.node == nullptr) {
        addNode(outputShape, 0, root);
    }
    ShapeNode* node = root.node.get();
    for (std::size_t depth = 0; depth < path.size(); ++depth) {
        std::size_t offset = path[depth];
        if (node->isFinal) {
            throw std::runtime_error("Cannot visit a child of a final node.");
        }
        if (offset >= node->children.size()) {
            throw std::runtime_error("Invalid path.");
        }
        ShapeNode::Next& next = node->children[offset];
        if (next.node == nullptr) {
            addNode(node->shape, depth + 1, next);
        }
        node = next.node.get();
    }
    return *node;
}

BindingContext& Sampler::getBindingContext() {
    return ctx;
}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const SampleOptions& options):
    rng { options.seed },
    options { options },
    ctx { options.countPrimaryVariables, options.countCoefficientVariables },
    root { std::make_unique<IdentityShapeOp>() }
{
    this->options.check();
    this->outputShape = ctx.getShapeFromNames(Shape::parseNames(outputShape));
    this->inputShape = ctx.getShapeFromNames(Shape::parseNames(inputShape));
}

std::vector<std::size_t> Sampler::randomSubPathFromNode(ShapeNode& node, std::size_t depth) {
    std::vector<std::size_t> path;
    // Returns true on success.
    const auto recursion = [this, &path](const auto& self, ShapeNode& current, std::size_t depth) -> bool {
        if (current.isFinal) {
            return true;
        } else {
            // Randomly iterate.
            std::vector<std::size_t> childrenIds(current.children.size(), 0);
            std::iota(childrenIds.begin(), childrenIds.end(), 0);
            std::shuffle(childrenIds.begin(), childrenIds.end(), rng);
            for (std::size_t i: childrenIds) {
                auto& child = current.children[i];
                if (child.node == nullptr) {
                    addNode(current.shape, depth + 1, child);
                }
                if (self(self, *child.node, depth + 1)) {
                    path.emplace_back(i);
                    return true;
                }
            }
        }
        return false;
    };
    if (recursion(recursion, node, depth)) {
        std::reverse(path.begin(), path.end());
    } else {
        throw std::runtime_error("Cannot find a path from the current node.");
    }
    return path;
}

std::vector<std::size_t> Sampler::randomPathWithPrefix(std::vector<std::size_t> prefix) {
    auto& node = visit(prefix);
    auto suffix = randomSubPathFromNode(node, prefix.size());
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    return prefix;
}

bool Sampler::isFinal(std::vector<std::size_t> path) {
    return visit(path).isFinal;
}

std::size_t Sampler::countChildren(std::vector<std::size_t> path) {
    const ShapeNode& node = visit(path);
    if (node.isFinal) {
        throw std::runtime_error("A final node has no child.");
    }
    return node.children.size();
}

std::tuple<TensorView, std::shared_ptr<CodeGenContext>, Representation> Sampler::realize(std::vector<std::size_t> path) {
    if (root.node == nullptr) {
        addNode(outputShape, 0, root);
    }
    std::shared_ptr<CodeGenContext> cgCtx;
    Representation repr { ctx };
    const auto recursion = [this, &path, &cgCtx, &repr](const auto& self, ShapeNode& current, std::size_t depth) -> TensorView {
        if (depth >= path.size()) {
            if (!current.isFinal) {
                throw std::runtime_error("When realizing a tensor, the path is not ending at a final node.");
            }
            // First divide the shape into input tensor and weight tensor.
            cgCtx = std::make_shared<CodeGenContext>();
            auto [inputS, weightS] = current.shape.cut<2>({ inputShape.size(), current.shape.size() - inputShape.size() });
            auto input = std::make_shared<PureTensor>(cgCtx->addTensor("input"), inputS);
            auto weight = std::make_shared<PureTensor>(cgCtx->addTensor("weight"), weightS);
            // Start to build a view of this tensor.
            auto view = TensorView { { std::move(input), std::move(weight) }, std::move(cgCtx) };
            repr.addShape(current.shape);
            return std::move(view);
        }
        // Follow the path.
        auto& child = current.children.at(path[depth]);
        if (child.node == nullptr) {
            addNode(current.shape, depth + 1, child);
        }
        TensorView result = self(self, *child.node, depth + 1);
        repr.addTransform(child.shapeOp->transformTensor(result));
        repr.addShape(current.shape);
        return std::move(result);
    };
    TensorView result = recursion(recursion, *root.node, 0);
    result.finishConstruction();
    result.setDefaultInterfaceAccess();
    result.evaluateTensorAccess();
    return std::make_tuple(std::move(result), std::move(cgCtx), std::move(repr));
}

std::tuple<TensorView, std::shared_ptr<CodeGenContext>, Representation> Sampler::randomSample() {
    return realize(randomPathWithPrefix({}));
}

} // namespace kas
