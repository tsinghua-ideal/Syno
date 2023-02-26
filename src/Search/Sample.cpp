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

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

void SampleOptions::check() const {
    KAS_ASSERT(dimLowerBound >= 1);
    KAS_ASSERT(dimUpperBound >= dimLowerBound);
}

Node Sampler::visitBase(std::size_t index) {
    return Node { &bases.at(index) };
}

Node Sampler::visitFromNode(const Node& node, std::span<const std::size_t> path) const {
    Node cur = node;
    for (auto i: path) {
        cur = std::get<Stage *>(cur.next(i));
    }
    return cur;
}

std::pair<Node, std::size_t> Sampler::visitButStopAtLast(const std::vector<std::size_t>& path) {
    KAS_ASSERT(path.size() >= 2, "The path must have at least two elements, for MapReduce and Finalize.");
    auto span = std::span(path).subspan(1, path.size() - 2);
    return { visitFromNode(visitBase(path[0]), span), path.back() };
}

Sampler::Sampler(std::vector<std::string> inputShape, std::vector<std::string> outputShape, std::vector<std::pair<std::string, Parser::PureSpec>> primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>> coefficientSpecs, const SampleOptions& options):
    rng { options.seed },
    ctx { primarySpecs.size(), coefficientSpecs.size() },
    options { options },
    inputShape { [&]() {
        ctx.applySpecs(primarySpecs, coefficientSpecs);
        return ctx.getShapeFromNames(inputShape);
    }() },
    outputShape { ctx.getShapeFromNames(outputShape) }
{
    this->options.check();
}

namespace {
    void parseSpecs(const std::vector<std::string>& specs, std::map<std::string, Parser::SizeSpec>& names, const char *prefix) {
        std::size_t unnamed = 0;
        for (const auto& spec: specs) {
            auto result = Parser(spec).parseSizeSpec();
            auto name = result.name();
            if (name) {
                names[*name] = std::move(result);
            } else {
                names[std::string(prefix) + std::to_string(unnamed++)] = std::move(result);
            }
        }
    }
    auto getShapeParsingCallback(std::map<std::string, Parser::SizeSpec>& primaryVars, const std::map<std::string, Parser::SizeSpec>& coefficientVars) {
        return [&](const std::string& newName) {
            if (!coefficientVars.contains(newName) && !primaryVars.contains(newName)) {
                // We have to add a default spec for the name.
                primaryVars[newName] = Parser::SizeSpec { .quantity = newName, .maxOccurrences = std::nullopt };
            }
        };
    }
    std::vector<std::pair<std::string, Parser::PureSpec>> contractSpecs(std::map<std::string, Parser::SizeSpec>& specs) {
        std::vector<std::pair<std::string, Parser::PureSpec>> result;
        for (auto&& [name, spec]: specs) {
            result.emplace_back(name, std::move(spec).toPureSpec());
        }
        return result;
    }
}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const SampleOptions& options, std::map<std::string, Parser::SizeSpec> primaryVars, std::map<std::string, Parser::SizeSpec> coefficientVars):
    Sampler {
        [&]() {
            parseSpecs(primarySpecs, primaryVars, "x_");
            parseSpecs(coefficientSpecs, coefficientVars, "c_");
            return Size::parseNames(inputShape, getShapeParsingCallback(primaryVars, coefficientVars));
        }(),
        Size::parseNames(outputShape, getShapeParsingCallback(primaryVars, coefficientVars)),
        contractSpecs(primaryVars),
        contractSpecs(coefficientVars),
        options,
    }
{}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const SampleOptions& options):
    Sampler { inputShape, outputShape, primarySpecs, coefficientSpecs, options, {}, {} }
{
    // TODO: fill the bases.
    KAS_UNIMPLEMENTED();
}

BindingContext& Sampler::getBindingContext() {
    return ctx;
}

std::vector<std::size_t> Sampler::randomPathWithPrefix(const std::vector<std::size_t>& prefix) {
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
        std::ranges::reverse(path);
    } else {
        throw std::runtime_error("Cannot find a path from the current node.");
    }
    return path;
}

std::vector<std::size_t> Sampler::randomPathWithPrefix(std::vector<std::size_t> prefix) {
    auto& node = visit(prefix);
    auto suffix = randomSubPathFromNode(node, prefix.size());
    std::ranges::copy(suffix, std::back_inserter(prefix));
    return prefix;
}

bool Sampler::isFinal(const std::vector<std::size_t>& path) {
    return visit(path).isFinal;
}

std::size_t Sampler::childrenCount(const std::vector<std::size_t>& path) {
    const ShapeNode& node = visit(path);
    if (node.isFinal) {
        throw std::runtime_error("A final node has no child.");
    }
    return node.children.size();
}

std::map<std::string, std::size_t> Sampler::childrenTypes(const std::vector<std::size_t>& path) {
    const ShapeNode& node = visit(path);
    if (node.isFinal) {
        return {};
    }
    std::map<std::string, std::size_t> result;
    for (const auto& child: node.children) {
        result[child.shapeOp->type()]++;
    }
    return result;
}

std::string Sampler::nodeString(const std::vector<std::size_t>& path) {
    return visit(path).shape.toString(ctx);
}

std::string Sampler::opString(const std::vector<std::size_t>& path) {
    return visitPointer(path).shapeOp->description();
}

std::string Sampler::opType(const std::vector<std::size_t>& path) {
    return visitPointer(path).shapeOp->type();
}

std::tuple<TensorView, std::shared_ptr<CodeGenContext>> Sampler::realize(const std::vector<std::size_t>& path) {
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
            // If no parameter is needed, just return the input tensor.
            if (current.shape.size() == inputShape.size()) {
                return TensorView { { std::make_shared<PureTensor>(cgCtx->addTensor("input"), current.shape) }, std::move(cgCtx) };
            }
            auto [inputS, weightS] = current.shape.cut<2>({ inputShape.size(), current.shape.size() - inputShape.size() });
            auto input = std::make_shared<PureTensor>(cgCtx->addTensor("input"), inputS);
            auto weight = std::make_shared<PureTensor>(cgCtx->addTensor("weight"), weightS);
            // Start to build a view of this tensor.
            return TensorView { { std::move(input), std::move(weight) }, std::move(cgCtx) };
        }
        // Follow the path.
        auto& child = current.children.at(path[depth]);
        if (child.node == nullptr) {
            addNode(current.shape, depth + 1, child);
        }
        TensorView result = self(self, *child.node, depth + 1);
        child.shapeOp->transformTensor(result);
        return result;
    };
    TensorView result = recursion(recursion, *root.node, 0);
    result.finishConstruction();
    result.setDefaultInterfaceAccess();
    result.evaluateTensorAccess();
    return std::make_tuple(std::move(result), std::move(cgCtx));
}

std::tuple<TensorView, std::shared_ptr<CodeGenContext>> Sampler::randomSample() {
    return realize(randomPathWithPrefix({}));
}

} // namespace kas
