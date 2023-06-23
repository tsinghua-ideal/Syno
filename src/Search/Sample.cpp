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
    KAS_ASSERT(maxStridedDimSize > 1);
    KAS_ASSERT(maxUnfoldKernelSize > 1);
    KAS_ASSERT(minimumUnfoldRatio >= 1.0f);
    KAS_ASSERT(minimumMergeRatio >= 1.0f);
    KAS_ASSERT(disallowSplitRAboveUnfold + disallowUnfoldLAboveSplit <= 1);
    KAS_ASSERT(disallowMergeWithLargeBlockAboveUnfold + disallowUnfoldLAboveMergeR <= 1);
    KAS_ASSERT(disallowSplitRAboveStride + disallowStrideAboveSplit <= 1);
    KAS_ASSERT(disallowMergeWithLargeBlockAboveStride + disallowStrideAboveMergeR <= 1);
    KAS_ASSERT(disallowUnfoldLAboveShift + disallowShiftAboveUnfold <= 1);
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

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const std::vector<std::map<std::string, std::size_t>>& allMappings, const std::vector<std::pair<std::size_t, std::size_t>>& fixedIODims, const SampleOptions& options):
    rng { options.seed },
    options { options }
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
    ctx.setMaxVariablesInSize(options.maximumVariablesInSize);
    ctx.setMaxVariablesPowersInSize(options.maximumVariablesPowersInSize);

    // Parse shape from names. TODO: add arithmetics support.
    this->inputShape = ctx.getShape(inputShapeNames);
    this->outputShape = ctx.getShape(outputShapeNames);

    // Apply the mappings to obtain concrete consts.
    ctx.applyMappings(allMappings);

    this->options.check();
    // Initialize the output iterators.
    for (std::size_t index = 0; const auto& domain: this->outputShape) {
        outputIterators.emplace_back(index++, domain);
    }

    // Check that the bound I/O dimensions are of the same size, and collect fixedDimensions.
    std::vector<std::size_t> boundOutputDimensions;
    for (auto [i, o]: fixedIODims) {
        KAS_ASSERT(this->inputShape[i] == this->outputShape[o], "Bound I/O dimensions must be of the same size.");
        fixedDimensions.emplace_back(i, &outputIterators[i]);
        boundOutputDimensions.emplace_back(o);
    }
    // Sort fixedDimensions by input index. We will insert them back into tensors in FinalizeOp::buildTensorView().
    std::ranges::sort(fixedDimensions, std::less{}, &FixedDimension::index);
    std::ranges::sort(boundOutputDimensions);
    // To bind input and output dimensions, remove fixed dimensions from inputShape and outputShape.
    for (std::size_t i: fixedDimensions | std::views::reverse | std::views::transform(&FixedDimension::index)) {
        this->inputShape.sizes.erase(this->inputShape.sizes.begin() + i);
    }
    for (std::size_t o: boundOutputDimensions | std::views::reverse) {
        this->outputShape.sizes.erase(this->outputShape.sizes.begin() + o);
    }

    // DO NOT modify root after this, because Dimension references by address these iterators.
    for (const auto& it: outputIterators) {
        // Exclude the bound dimensions.
        if (std::ranges::binary_search(boundOutputDimensions, it.getIndex())) {
            continue;
        }
        root.emplace_back(&it);
    }

    // Generate MapReduce's. This recursively calls MapReduceOp::Generate().
    rootStage = std::make_unique<ReductionStage>(*this);
}

TensorExpression Sampler::getExpressionForTensorNum(std::size_t num) const {
    std::string_view expr;
    switch (num) {
    case 1: expr = options.expressionOneTensor; break;
    case 2: expr = options.expressionTwoTensors; break;
    case 3: expr = options.expressionThreeTensors; break;
    case 4: expr = options.expressionFourTensors; break;
    default: KAS_CRITICAL("Unsupported number of tensors: {}", num);
    }
    return Parser(expr).parseTensorExpression();
}

Size Sampler::getTotalOutputSize() const {
    Size result = outputShape.totalSize();
    if (!fixedDimensions.empty()) {
        using FixedDimensionsShapeView = AbstractShape<const std::vector<FixedDimension>&, [](const FixedDimension& fd) -> const Size& { return fd.dim.size(); }>;
        result = result * FixedDimensionsShapeView { fixedDimensions }.totalSize();
    }
    return result;
}

std::optional<Node> Sampler::visit(const std::vector<Next>& path) {
    Node n { this, rootStage.get() };
    for (const auto& next: path) {
        auto nextNode = n.getChild(next);
        if (!nextNode) {
            return std::nullopt;
        }
        n = *nextNode;
    }
    return n;
}

std::optional<std::pair<std::vector<Next>, Node>> Sampler::randomNodeWithPrefix(const std::vector<Next>& prefix) {
    std::vector<Next> path = prefix;
    std::optional<Node> optCur = visit(prefix);
    if (!optCur) {
        return std::nullopt;
    }
    Node cur = *optCur;
    // Recursively visit children.
    while (true) {
        auto cnt = cur.countChildren();
        if (cnt == 0) {
            auto stage = cur.tryAsStage();
            KAS_ASSERT(!stage || stage->getFinalizability() == AbstractStage::Finalizability::No);
            break;
        }
        auto next = cur.getChildrenHandles()[random(cnt)];
        path.emplace_back(next);
        auto nextNode = cur.getChild(next);
        if (!nextNode) {
            return std::nullopt;
        }
        cur = *nextNode;
    };
    return std::optional<std::pair<std::vector<Next>, Node>>(std::in_place, std::move(path), std::move(cur));
}

void Sampler::ConvertTensorViewToSearchableOrder(std::vector<std::vector<Dimension>>& tensorView) {
    // First sort the weights in order of hash. This somewhat duplicates the functionality in Forward::buildTensorView(). TODO
    std::ranges::for_each(tensorView | std::views::drop(1), [](std::vector<Dimension>& dims) {
        std::ranges::sort(dims, Dimension::HashLessThan{});
    });
}

std::vector<Next> Sampler::convertTensorViewToPath(const std::vector<std::vector<Dimension>>& tensorView) const {
    Graph::Builder builder;
    builder.addTopmost(tensorView | std::views::join);
    Graph graph = builder.build();

    std::vector<Next> result;
    // To obtain the path, we need to follow the 3 stages of searching.

    // First, ReductionStage.
    {
        for (const MapReduce *op: graph.getMapReduceIterators()) {
            result.emplace_back(Next::Type::MapReduce, op->hash());
        }
    }

    // Next, NormalStage.
    {
        std::set<Dimension, Dimension::HashLessThan> completed;
        Graph::AttributeMap<bool> added;
        // Bottom-up.
        auto dfs = [&](const auto& self, const Dimension& dim) -> void {
            if (completed.contains(dim)) return;
            completed.emplace(dim);
            graph.visitAlong(dim, Direction::Down).match(
                [&](const RepeatLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[RepeatLikeOp::Branch::Output]);
                    result.emplace_back(Next::TypeOf(v.op.getType()), v.op.opHash());
                },
                [&](const SplitLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[SplitLikeOp::Branch::OutputLhs]);
                    self(self, v[SplitLikeOp::Branch::OutputRhs]);
                    result.emplace_back(Next::TypeOf(v.op.getType()), v.op.opHash());
                },
                [&](const MergeLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[MergeLikeOp::Branch::Output]);
                    result.emplace_back(Next::TypeOf(v.op.getType()), v.op.opHash());
                }
            );
        };
        // The reverse is just a simple fix. We need to generate Next's in canonical order of ShareOp! See ShareOp::IsSharedDimensionCanonical(). TODO.
        for (const auto& tensor: tensorView | std::views::reverse) {
            for (const Dimension& dim: tensor) {
                dfs(dfs, dim);
            }
        }
    }

    // Finally, Finalize.
    {
        // The fixed dimensions should be removed first.
        std::vector<std::vector<Dimension>> tensors;
        std::ranges::copy(tensorView, std::back_inserter(tensors));
        auto& inputTensor = tensors.at(0);
        for (const auto& [i, _]: fixedDimensions | std::views::reverse) {
            inputTensor.erase(inputTensor.begin() + i);
        }
        result.emplace_back(Next::Type::Finalize, NextFinalizeSlot::GetKey(tensors));
    }

    return result;
}

} // namespace kas
