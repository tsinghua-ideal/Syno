#include "KAS/Transforms/Contraction.hpp"
#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

std::size_t ContractionOp::Dimwise::hash() const noexcept {
    // TODO!!!
}

std::weak_ordering ContractionOp::Dimwise::operator<=>(const Dimwise& other) const noexcept {
    auto hash = share->output.hash() <=> other.share->output.hash();
    if (hash != 0) {
        return hash;
    }
    return (expand != nullptr) <=> (other.expand != nullptr);
}

std::size_t ContractionOp::opHash() const noexcept {
    // TODO!!!
}

bool ContractionOp::canApplyToInterface(const GraphHandle& interface) const {
    return std::ranges::all_of(dimwiseOps, [&](const Dimwise& dimwise) {
        return dimwise.share->canApplyToInterface(interface);
    });
}

void ContractionOp::applyToInterface(GraphHandle& interface) const {
    for (const Dimwise& dimwise: dimwiseOps) {
        dimwise.share->applyToInterface(interface);
        if (dimwise.expand != nullptr) {
            dimwise.expand->applyToInterface(interface);
        }
    }
}

GraphHandle ContractionOp::appliedToInterface(const GraphHandle& interface) const {
    auto newInterface = interface;
    applyToInterface(newInterface);
    return newInterface;
}

std::string ContractionOp::description(const BindingContext& ctx) const {
    // TODO!!!
}

std::string ContractionOp::descendantsDescription(const BindingContext& ctx) const {
    // TODO!!!
}

ContractionOp::Analysis ContractionOp::Analyze(const GraphHandle& interface, const Graph& graph) {
    int maxWeightId = 0;
    int numShares = 0;
    for (const ShareOp *shareOp: graph.getOpsOfType<ShareOp>()) {
        maxWeightId = std::max(maxWeightId, shareOp->getRhsOrigin());
        ++numShares;
    }
    const bool hasContracted = maxWeightId > 0;
    auto simpleViewSearchable = Topmost(std::vector<Dimension>{}, interface.getExpansions());
    auto full = Topmost(std::vector<Dimension>{}, interface.getExpansions());
    Graph::DimensionSet other;
    for (const Dimension& dim: interface.getDimensions()) {
        if (dim.is(DimensionTypeWithOrder::ShareR)) continue;
        if (hasContracted) {
            const auto& existingShareDescendents = graph.colorOf(dim).getTags();
            const bool isLatest = std::ranges::find_if(
                existingShareDescendents,
                [&](const ShareOp *share) {
                    return share->getRhsOrigin() == maxWeightId;
                },
                [](const MergeLikeOp *op) { return dynamic_cast<const ShareOp *>(op); }
            ) != existingShareDescendents.end();
            if (isLatest) {
                simpleViewSearchable.getDimensions().emplace_back(dim);
            } else {
                other.emplace(dim);
            }
        } else {
            simpleViewSearchable.getDimensions().emplace_back(dim);
        }
        full.getDimensions().emplace_back(dim);
    }
    return {
        .maxWeightId = maxWeightId,
        .numShares = numShares,
        .simpleViewSearchable = std::move(simpleViewSearchable),
        .other = std::move(other),
        .full = std::move(full),
    };
}

std::vector<const ContractionOp *> ContractionOp::Generate(PrimitiveOpStore& store, ContractionOpStore& contractionStore, const GenerateOptions& options) {
    // TODO!!!
}

} // namespace kas
