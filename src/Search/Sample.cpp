#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

void SampleOptions::check() const {
    KAS_ASSERT(dimLowerBound >= 1);
    KAS_ASSERT(dimUpperBound >= dimLowerBound);
    KAS_ASSERT(maximumTensors >= 1);
}

Node Sampler::visitBase(std::size_t index) {
    return Node { &bases.at(index) };
}

Node Sampler::visitFromNode(const Node& node, std::span<const std::size_t> path) const {
    Node cur = node;
    for (auto i: path) {
        cur = Node::AssertNotFinal(cur.next(i));
    }
    return cur;
}

Node Sampler::visitFromRoot(const std::vector<std::size_t>& path) {
    KAS_ASSERT(path.size() >= 1, "The path must have at least one element, for MapReduce.");
    return visitFromNode(visitBase(path.front()), std::span(path).subspan(1));
}

std::pair<Node, std::size_t> Sampler::visitFromNodeButStopAtLast(const Node& node, std::span<const std::size_t> path) {
    KAS_ASSERT(path.size() >= 1, "The path must have at least one elements, for Finalize.");
    auto span = path.first(path.size() - 1);
    return { visitFromNode(node, span), path.back() };
}

std::pair<Node, std::size_t> Sampler::visitFromRootButStopAtLast(const std::vector<std::size_t>& path) {
    KAS_ASSERT(path.size() >= 2, "The path must have at least two elements, for MapReduce and Finalize.");
    return visitFromNodeButStopAtLast(visitBase(path.front()), std::span(path).subspan(1));
}

std::variant<Stage *, TensorView *> Sampler::visit(const std::vector<std::size_t>& path) {
    KAS_ASSERT(path.size() >= 1, "The path must have at least one element, for MapReduce.");
    auto [node, last] = visitFromRootButStopAtLast(path);
    return node.next(last);
}

Sampler::Sampler(std::vector<std::string> inputShape, std::vector<std::string> outputShape, std::vector<std::pair<std::string, Parser::PureSpec>> primarySpecs, std::vector<std::pair<std::string, Parser::PureSpec>> coefficientSpecs, const SampleOptions& options):
    rng { options.seed },
    ctx { primarySpecs.size(), coefficientSpecs.size() },
    options { options },
    colorOptions { .maximumTensors = options.maximumTensors },
    inputShape { [&]() {
        ctx.applySpecs(primarySpecs, coefficientSpecs);
        return ctx.getShapeFromNames(inputShape);
    }() },
    outputShape { ctx.getShapeFromNames(outputShape) }
{
    this->options.check();
    for (std::size_t index = 0; const auto& domain: this->outputShape) {
        outputIterators.emplace_back(index++, domain);
    }
    for (auto& it: outputIterators) {
        root.emplace_back(&it);
    }
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
    reduces = MapReduceOp::GenerateLastLevelMapReduces(this->outputShape, { this->ctx, this->options.dimUpperBound });
    for (auto& r: reduces) {
        ColoredInterface temp;
        std::ranges::copy(root | std::views::transform([](const Dimension& dim) { return ColoredDimension { dim, Colors::Unknown }; }), std::back_inserter(temp.items)); // Maybe we can determine the colors here? TODO. Use it below.
        std::ranges::copy(r | std::views::transform([](MapReduceOp& m) { return ColoredDimension{ &m, Colors::Unknown }; }), std::back_inserter(temp.items));
        std::ranges::sort(temp.items, Dimension::HashLessThan{}, ColoredDimension::Projection{});
        this->bases.emplace_back(std::move(temp), Colors(colorOptions), *this, 0);
    }
}

std::vector<std::size_t> Sampler::randomPathWithPrefix(const std::vector<std::size_t>& prefix) {
    std::vector<std::size_t> path = prefix;
    Node cur;
    if (prefix.size() == 0) {
        auto first = random(bases.size());
        path.emplace_back(first);
        cur = visitBase(first);
    } else if (prefix.size() == 1) {
        auto first = prefix[0];
        cur = visitBase(first);
    } else {
        auto [node, last] = visitFromRootButStopAtLast(prefix);
        cur = node;
        if (node.isFinal(last)) {
            return path;
        }
    }
    // Recursively visit children.
    while (true) {
        auto cnt = cur.countChildren();
        if (cnt == 0) {
            break; // Fail. No child to visit.
        }
        auto next = random(cnt);
        path.emplace_back(next);
        if (cur.isFinal(next)) {
            break; // Success. Found final node.
        }
        cur = Node::AssertNotFinal(cur.next(next));
    };
    return path;
}

bool Sampler::isFinal(const std::vector<std::size_t>& path) {
    if (path.size() < 2) {
        return false;
    }
    auto [node, last] = visitFromRootButStopAtLast(path);
    return node.isFinal(last);
}

std::size_t Sampler::childrenCount(const std::vector<std::size_t>& path) {
    if (path.size() == 0) {
        return bases.size();
    }
    // If we actually visit a final node, this will crash.
    return visitFromRoot(path).countChildren();
}

std::map<std::string, std::size_t> Sampler::childrenTypes(const std::vector<std::size_t>& path) {
    if (path.size() == 0) {
        return { { DimensionTypeDescription(DimensionType::MapReduce), bases.size() } };
    } else if (path.size() == 1) {
        return visitBase(path[0]).childrenTypes();
    }
    auto [node, last] = visitFromRootButStopAtLast(path);
    if (node.isFinal(last)) {
        return {};
    }
    node = Node::AssertNotFinal(node.next(last));
    return node.childrenTypes();
}

std::string Sampler::nodeString(const std::vector<std::size_t>& path) {
    if (path.size() == 0) {
        return outputShape.toString(ctx);
    } else if (path.size() == 1) {
        return visitBase(path[0]).shapeDescription(ctx);
    }
    auto [node, last] = visitFromRootButStopAtLast(path);
    struct visitor {
        const BindingContext& ctx;
        std::string operator()(Stage *s) {
            return Node(s).shapeDescription(ctx);
        }
        std::string operator()(TensorView *t) {
            return t->getShape().toString(ctx);
        }
    };
    return std::visit(visitor { ctx }, node.next(last));
}

std::string Sampler::opString(const std::vector<std::size_t>& path) {
    if (path.size() == 0) {
        return "Root";
    } else if (path.size() == 1) {
        return DimensionTypeDescription(DimensionType::MapReduce); // TODO: Add detailed description.
    }
    auto [node, last] = visitFromRootButStopAtLast(path);
    return node.opDescription(last);
}

std::string Sampler::opType(const std::vector<std::size_t>& path) {
    if (path.size() == 0) {
        return "Root";
    }
    if (path.size() == 1) {
        return DimensionTypeDescription(DimensionType::MapReduce);
    } else {
        auto [node, last] = visitFromRootButStopAtLast(path);
        return node.opType(last);
    }
}

TensorView *Sampler::realize(const std::vector<std::size_t>& path) {
    struct visitor {
        TensorView *operator()(Stage *s) {
            return nullptr;
        }
        TensorView *operator()(TensorView *t) {
            return t;
        }
    };
    return std::visit(visitor{}, visit(path));
}

TensorView *Sampler::randomSample() {
    return realize(randomPathWithPrefix({}));
}

} // namespace kas
