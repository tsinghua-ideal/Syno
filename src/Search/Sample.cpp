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
    std::vector<std::pair<std::string, Parser::PureSpec>> contractSpecs(std::map<std::string, Parser::SizeSpec>& specs) {
        std::vector<std::pair<std::string, Parser::PureSpec>> result;
        for (auto&& [name, spec]: specs) {
            result.emplace_back(name, std::move(spec).toPureSpec());
        }
        return result;
    }
}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const std::vector<std::map<std::string, std::size_t>>& allMappings, const SampleOptions& options):
    rng { options.seed },
    options { options },
    colorOptions { .maximumTensors = options.maximumTensors }
{
    // First parse the variable names in specifications. Unnamed variables are named by x_i and c_i.
    std::map<std::string, Parser::SizeSpec> primaryVars;
    std::map<std::string, Parser::SizeSpec> coefficientVars;
    parseSpecs(primarySpecs, primaryVars, "x_");
    parseSpecs(coefficientSpecs, coefficientVars, "c_");

    // Then we collect the variables in in/out shapes. In case of new variable, add it to primary.
    auto onNewName = [&](const std::string& newName) {
        if (!coefficientVars.contains(newName) && !primaryVars.contains(newName)) {
            // We have to add a default spec for the name.
            primaryVars[newName] = Parser::SizeSpec { .quantity = newName, .maxOccurrences = std::nullopt };
        }
    };
    std::vector<std::string> inputShapeNames = Size::parseNames(inputShape, onNewName);
    std::vector<std::string> outputShapeNames = Size::parseNames(outputShape, onNewName);

    // Put the specs in order.
    auto contractedPrimarySpecs = contractSpecs(primaryVars);
    auto contractedCoefficientSpecs = contractSpecs(coefficientVars);

    // Apply the specs to all variables.
    ctx = BindingContext { contractedPrimarySpecs.size(), contractedCoefficientSpecs.size() };
    ctx.applySpecs(contractedPrimarySpecs, contractedCoefficientSpecs);

    // Parse shape from names. TODO: add arithmetics support.
    this->inputShape = ctx.getShapeFromNames(inputShapeNames);
    this->outputShape = ctx.getShapeFromNames(outputShapeNames);

    this->options.check();
    // Initialize the output iterators.
    for (std::size_t index = 0; const auto& domain: this->outputShape) {
        outputIterators.emplace_back(index++, domain);
    }
    // DO NOT modify root after this, because Dimension references by address these iterators.
    for (const auto& it: outputIterators) {
        root.emplace_back(&it);
    }

    // Generate MapReduce's.
    reduces = MapReduceOp::GenerateLastLevelMapReduces(this->outputShape, { this->ctx, this->options.dimUpperBound });
    // DO NOT modify reduces and originalBases after this, because Dimension references by address these iterators.
    for (const auto& r: reduces) {
        ColoredInterface temp;
        std::ranges::copy(root | std::views::transform([](const Dimension& dim) { return ColoredDimension { dim, Colors::Unknown }; }), std::back_inserter(temp.items)); // Maybe we can determine the colors here? TODO. Use it below.
        std::ranges::copy(r | std::views::transform([](const MapReduceOp& m) { return ColoredDimension{ &m, Colors::Unknown }; }), std::back_inserter(temp.items));
        std::ranges::sort(temp.items, Dimension::HashLessThan{}, ColoredDimension::Projection{});
        originalBases.emplace_back(std::move(temp), Colors(colorOptions), *this, 0);
    }
    auto hasher = std::hash<Interface>{};
    std::ranges::move(std::views::iota(static_cast<std::size_t>(0), originalBases.size()) | std::views::transform([&](std::size_t baseIndex) { return std::pair{ hasher(originalBases[baseIndex].getInterface().toDimensions()), baseIndex }; }), std::back_inserter(bases));
    std::ranges::sort(bases, std::less{}, &std::pair<std::size_t, std::size_t>::first);
}

std::vector<Next> Sampler::getNextBases() const {
    std::vector<Next> result;
    std::ranges::move(bases | std::views::transform([](const auto& base) { return Next { Next::Type::MapReduce, base.first }; }), std::back_inserter(result));
    return result;
}

std::size_t Sampler::getBaseIndex(std::size_t key) const {
    auto it = std::ranges::lower_bound(bases, key, std::less{}, &std::pair<std::size_t, std::size_t>::first);
    KAS_ASSERT(it != bases.end() && it->first == key, "Specified MapReduce not found.");
    return it->second;
}

Stage *Sampler::getBase(std::size_t key) {
    return &originalBases[getBaseIndex(key)];
}

const MapReduceOp::Base& Sampler::getReduce(std::size_t key) const {
    return reduces[getBaseIndex(key)];
}

Node Sampler::visit(const std::vector<Next>& path) {
    Node n { this };
    for (const auto& next: path) {
        n = n.getChild(next);
    }
    return n;
}

std::pair<std::vector<Next>, Node> Sampler::randomNodeWithPrefix(const std::vector<Next>& prefix) {
    std::vector<Next> path = prefix;
    Node cur = visit(prefix);
    // Recursively visit children.
    while (true) {
        auto cnt = cur.countChildren();
        if (cnt == 0) {
            break;
        }
        auto next = cur.getChildrenHandles()[random(cnt)];
        path.emplace_back(next);
        cur = cur.getChild(next);
    };
    return { std::move(path), std::move(cur) };
}

} // namespace kas
