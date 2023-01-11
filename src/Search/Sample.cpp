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
    options { options },
    ctx { options.countPrimaryVariables, options.countCoefficientVariables },
    root { std::make_unique<IdentityShapeOp>() }
{
    this->options.check();
    this->outputShape = ctx.getShapeFromNames(Shape::parseNames(outputShape));
    this->inputShape = ctx.getShapeFromNames(Shape::parseNames(inputShape));
}

std::pair<TensorView, std::vector<std::size_t>> Sampler::randomSample() {
    std::vector<std::size_t> path;
    std::random_device rd;
    std::mt19937 rng { rd() };
    if (root.node == nullptr) {
        addNode(outputShape, 0, root);
    }
    const auto recursion = [this, &rng, &path](const auto& self, ShapeNode& current, int depth) -> std::optional<TensorView> {
        if (current.isFinal) {
            // First divide the shape into input tensor and weight tensor.
            auto cgCtx = std::make_shared<CodeGenContext>();
            auto [inputS, weightS] = current.shape.cut<2>({ inputShape.size(), current.shape.size() - inputShape.size() });
            auto input = std::make_shared<PureTensor>(cgCtx->addTensor("input"), inputS);
            auto weight = std::make_shared<PureTensor>(cgCtx->addTensor("weight"), weightS);
            // Start to build a view of this tensor.
            auto view = TensorView { { std::move(input), std::move(weight) }, std::move(cgCtx) };
            view.addIntermediateShape(current.shape.toString(ctx));
            return std::move(view);
        } else {
            // Here we just randomly iterate. In MCTS, use UCB. TODO
            std::vector<std::size_t> childrenIds(current.children.size(), 0);
            std::iota(childrenIds.begin(), childrenIds.end(), 0);
            std::shuffle(childrenIds.begin(), childrenIds.end(), rng);
            for (std::size_t i: childrenIds) {
                auto& child = current.children[i];
                if (child.node == nullptr) {
                    addNode(current.shape, depth + 1, child);
                }
                if (auto result = self(self, *child.node, depth + 1)) {
                    path.emplace_back(i);
                    child.shapeOp->transformTensor(result.value());
                    result.value().addIntermediateShape(current.shape.toString(ctx));
                    return std::move(result);
                }
            }
        }
        return std::nullopt;
    };
    if (auto result = recursion(recursion, *root.node, 0)) {
        auto& tensorView = result.value();
        tensorView.finishConstruction();
        tensorView.setDefaultInterfaceAccess();
        tensorView.evaluateTensorAccess();
        std::reverse(path.begin(), path.end());
        return std::make_pair(std::move(tensorView), std::move(path));
    } else {
        KAS_CRITICAL("Cannot sample a tensor.");
    }
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

std::pair<TensorView, std::shared_ptr<CodeGenContext>> Sampler::realize(std::vector<std::size_t> path) {
    if (root.node == nullptr) {
        addNode(outputShape, 0, root);
    }
    std::shared_ptr<CodeGenContext> cgCtx;
    const auto recursion = [this, &path, &cgCtx](const auto& self, ShapeNode& current, std::size_t depth) -> TensorView {
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
            view.addIntermediateShape(current.shape.toString(ctx));
            return std::move(view);
        }
        // Follow the path.
        auto& child = current.children.at(path[depth]);
        if (child.node == nullptr) {
            addNode(current.shape, depth + 1, child);
        }
        TensorView result = self(self, *child.node, depth + 1);
        child.shapeOp->transformTensor(result);
        result.addIntermediateShape(current.shape.toString(ctx));
        return std::move(result);
    };
    TensorView result = recursion(recursion, *root.node, 0);
    result.finishConstruction();
    result.setDefaultInterfaceAccess();
    result.evaluateTensorAccess();
    return std::make_pair(std::move(result), std::move(cgCtx));
}

} // namespace kas
