#include "KAS/Transforms/Contraction.hpp"
#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Ranges.hpp"


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

ContractionOp::SharedCandidateType ContractionOp::GetSharedCandidateType(Dimension dim) {
    while (auto s = dim.tryAs<ShareOp::Input>()) {
        KAS_ASSERT(s->getOrder() == Order::Left);
        dim = s->getOp()->output;
    }
    switch (dim.type()) {
    case DimensionType::Merge:
    case DimensionType::Iterator:
        return SharedCandidateType::Merge;
    case DimensionType::Reduce:
        return SharedCandidateType::WeightsSharing;
    default:
        return SharedCandidateType::Normal;
    }
}

ContractionOp::Analysis ContractionOp::Analyze(const BindingContext& ctx, const Graph& graph, const GraphHandle& interface) {
    int maxWeightId = 0;
    int numShares = 0;
    Size numelOuter = Size::Identity(ctx);
    for (const ShareOp *shareOp: graph.getOpsOfType<ShareOp>()) {
        maxWeightId = std::max(maxWeightId, shareOp->getRhsOrigin());
        ++numShares;
        graph.visitAlong(shareOp->getInputL(), Direction::Up).match(Match {
            [&](const ExpandVertex&, auto) {
                // Expand + Share pattern.
                // We still have to find out if it is actually merge or a low-rank decomposition.
                if (GetSharedCandidateType(shareOp->output) == SharedCandidateType::Merge) {
                    numelOuter *= shareOp->output.size();
                }
            },
            []<typename T>(const T&, auto) requires(!std::same_as<T, ExpandVertex>) {},
        });
    }
    const bool hasContracted = maxWeightId > 0;
    auto simpleViewSearchable = Topmost(std::vector<Dimension>{}, interface.getExpansions());
    auto full = Topmost(std::vector<Dimension>{}, interface.getExpansions());
    Graph::DimensionSet other;
    std::vector<SharedCandidateType> candidateTypes;
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
        candidateTypes.emplace_back(GetSharedCandidateType(dim));
    }
    return {
        .maxWeightId = maxWeightId,
        .numShares = numShares,
        .simpleViewSearchable = std::move(simpleViewSearchable),
        .other = std::move(other),
        .full = std::move(full),
        .candidateTypes = std::move(candidateTypes),
        .numelOuter = numelOuter,
    };
}

ContractionOp::Enumerator ContractionOp::Enumerator::assign(std::optional<ContractionType> type, const Allowance& newAllowance, const Size& newNumelOuter) const {
    auto newAssigned = assigned;
    newAssigned.emplace_back(type);
    return {
        .options = options,
        .allowance = newAllowance,
        .numelOuter = newNumelOuter,
        .numShares = numShares + type.has_value(),
        .assigned = std::move(newAssigned),
    };
}

const ContractionOp *ContractionOp::Enumerator::apply() const {
    const auto& available = options.available;
    KAS_ASSERT(available.size() == assigned.size());

    // Why not allow 1? Later explained.
    if (numShares <= 1) return nullptr;
    std::vector<std::size_t> outer, inner;
    for (std::size_t i = 0; i < available.size(); ++i) {
        const auto& assignment = assigned[i];
        if (!assignment.has_value()) continue;
        switch (*assignment) {
        case ContractionType::Outer:
            outer.emplace_back(i);
            break;
        case ContractionType::Inner:
            inner.emplace_back(i);
            break;
        }
    }

    // A simple rule. We must include at least one outer and one inner.
    // Because this is what a matmul requires.
    // Without matmul, we can only get outer product or hadamard product or weighted sum.
    // TODO! Think over this.
    if (outer.empty() || inner.empty()) return nullptr;

    // TODO!!! Canonical order of ContractionOp!

    auto& store = options.store;
    auto& contractionStore = options.contractionStore;
    std::vector<Dimwise> result;
    for (std::size_t i: outer) {
        auto shareOp = store.get<ShareOp>(available[i].dim, options.weightId);
        auto expandOp = store.get<ExpandOp>(shareOp->getInputL());
        result.emplace_back(shareOp, expandOp);
    }
    for (std::size_t i: inner) {
        auto shareOp = store.get<ShareOp>(available[i].dim, options.weightId);
        result.emplace_back(shareOp, nullptr);
    }
    return contractionStore.get(std::move(result));
}

Generator<const ContractionOp *> ContractionOp::Enumerator::generate() const {
    const auto& available = options.available;
    // Guard.
    if (available.empty()) co_return;
    if (available.size() == assigned.size()) {
        // Done.
        auto result = apply();
        if (result) {
            co_yield result;
        }
        co_return;
    }

    // First try without this.
    {
        auto withoutThis = assign(std::nullopt, allowance, numelOuter);
        for (const ContractionOp *op: withoutThis.generate()) co_yield op;
    }
    const CandidateDimension& next = available[assigned.size()];
    // Then with this.
    {
        // Check numShares.
        if (numShares >= options.maxShares) co_return;
        // Check allowance.
        if (!allowance.shareWithinAllowance(next.dim.size())) {
            co_return;
        }
        const auto newAllowance = allowance.shared(next.dim.size());
        // Anyhow, a simple Share is OK.
        {
            auto withThis = assign(ContractionType::Inner, newAllowance, numelOuter);
            for (const ContractionOp *op: withThis.generate()) co_yield op;
        }
        if (next.type == SharedCandidateType::Merge) {
            // Outer product.
            const auto newNumelOuter = numelOuter * next.dim.size();
            if (newNumelOuter.upperBoundEst(options.ctx) <= options.maxExpansionMergeMultiplier) {
                auto withThis = assign(ContractionType::Outer, newAllowance, newNumelOuter);
                for (const ContractionOp *op: withThis.generate()) co_yield op;
            }
        } else if (next.type == SharedCandidateType::WeightsSharing) {
            // Low-rank decomposition.
            // We need to restrict this pattern.
            auto weightsSharing = next.dim.size().upperBoundEst(options.ctx);
            if (
                weightsSharing >= options.minExpansionWeightsSharingDimSize &&
                weightsSharing <= options.maxExpansionWeightsSharingDimSize
            ) {
                auto withThis = assign(ContractionType::Outer, newAllowance, numelOuter);
                for (const ContractionOp *op: withThis.generate()) co_yield op;
            }
        }
    }
}

std::vector<const ContractionOp *> ContractionOp::Generate(PrimitiveOpStore& store, ContractionOpStore& contractionStore, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const Analysis& analysis = options.analysis;
    const int nextWeightId = analysis.maxWeightId + 1;
    if (options.maximumTensors <= nextWeightId) {
        // No more tensors.
        return {};
    }

    const Graph& graph = options.graph;
    const Allowance& allowance = options.allowance;
    const std::vector<Dimension>& fullDims = analysis.full.getDimensions();
    const std::vector<SharedCandidateType>& candidateTypes = analysis.candidateTypes;

    // TODO!!! Canonicalization of contractions!
    std::vector<CandidateDimension> available;
    for (std::size_t i = 0; i < fullDims.size(); ++i) {
        if (graph.colorOf(fullDims[i]).isDataDiscarding()) continue;
        available.emplace_back(fullDims[i], candidateTypes[i]);
    }

    Enumerator::Options enumeratorOptions {
        .store = store,
        .contractionStore = contractionStore,
        .ctx = options.ctx,
        .weightId = nextWeightId,
        .maxShares = options.maxShares,
        .maxExpansionMergeMultiplier = options.maxExpansionMergeMultiplier,
        .maxExpansionWeightsSharingDimSize = options.maxExpansionWeightsSharingDimSize,
        .minExpansionWeightsSharingDimSize = options.minExpansionWeightsSharingDimSize,
        .available = available,
    };

    Enumerator enumerator {
        .options = enumeratorOptions,
        .allowance = allowance,
        .numelOuter = analysis.numelOuter,
        .numShares = 0,
        .assigned = {},
    };

    auto result = ranges::to<std::vector<const ContractionOp *>>(enumerator.generate());
    CountSuccessfulGenerations += result.size();
    return result;
}

} // namespace kas
