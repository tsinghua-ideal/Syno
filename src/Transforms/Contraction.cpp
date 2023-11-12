#include "KAS/Transforms/Contraction.hpp"
#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/OperationStore.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

bool ContractionOp::isEqual(const Operation& other) const {
    return dimwiseOps == static_cast<const ContractionOp&>(other).dimwiseOps;
}

std::weak_ordering ContractionOp::Dimwise::operator<=>(const Dimwise& other) const noexcept {
    auto hash = share->opHash() <=> other.share->opHash();
    if (hash != 0) {
        return hash;
    }
    return (expand != nullptr) <=> (other.expand != nullptr);
}

std::size_t ContractionOp::Dimwise::hash() const noexcept {
    using namespace std::string_view_literals;
    static const std::size_t ContractionDimwiseHash = std::hash<std::string_view>{}("ContractionDimwise"sv);
    std::size_t h = ContractionDimwiseHash;
    HashCombineRaw(h, share->opHash());
    static const std::size_t ContractionHasExpansionHash = std::hash<std::string_view>{}("ContractionHasExpansion"sv);
    static const std::size_t ContractionNoExpansionHash = std::hash<std::string_view>{}("ContractionNoExpansion"sv);
    if (expand != nullptr) {
        HashCombineRaw(h, ContractionHasExpansionHash);
    } else {
        HashCombineRaw(h, ContractionNoExpansionHash);
    }
    return h;
}

ContractionType ContractionOp::Dimwise::type() const noexcept {
    return expand ? ContractionType::Outer : ContractionType::Inner;
}

std::string ContractionOp::Dimwise::description(const BindingContext& ctx) const {
    if (expand) {
        return fmt::format(
            "[], {} -> {}",
            share->getInputR().description(ctx),
            share->output.description(ctx)
        );
    } else {
        return share->description(ctx);
    }
}

std::string ContractionOp::Dimwise::descendantsDescription(const BindingContext& ctx) const {
    if (expand) {
        return fmt::format(
            "{}, {} -> {}",
            expand->descendantsDescription(ctx),
            share->getInputR().descendantsDescription(ctx),
            share->output.descendantsDescription(ctx)
        );
    } else {
        return share->descendantsDescription(ctx);
    }
}

std::size_t ContractionOp::opHash() const noexcept {
    using namespace std::string_view_literals;
    static const std::size_t ContractionHash = std::hash<std::string_view>{}("Contraction"sv);
    std::size_t h = ContractionHash;
    HashCombine(h, dimwiseOps.size());
    for (const Dimwise& dimwise: dimwiseOps) {
        HashCombineRaw(h, dimwise.hash());
    }
    return h;
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

std::string ContractionOp::description(const BindingContext& ctx) const {
    return fmt::format(
        "[{}]@Contraction{}",
        fmt::join(
            dimwiseOps | std::views::transform([&ctx](const Dimwise& dimwise) {
                return dimwise.description(ctx);
            }),
            "]["
        ),
        opHash()
    );
}

std::string ContractionOp::descendantsDescription(const BindingContext& ctx) const {
    return fmt::format(
        "[{}]@Contraction{}",
        fmt::join(
            dimwiseOps | std::views::transform([&ctx](const Dimwise& dimwise) {
                return dimwise.descendantsDescription(ctx);
            }),
            "]["
        ),
        opHash()
    );
}

ContractionOp::SharedCandidateType ContractionOp::GetSharedCandidateType(Dimension dim) {
    Dimension bottom = dim;
    while (auto s = bottom.tryAs<ShareOp::Input>()) {
        KAS_ASSERT(s->getOrder() == Order::Left);
        bottom = s->getOp()->output;
    }
    switch (bottom.type()) {
    case DimensionType::Merge:
    case DimensionType::Iterator:
        return SharedCandidateType::Merge;
    case DimensionType::Reduce:
        if (bottom == dim) {
            // Not yet.
            return SharedCandidateType::Normal;
        } else {
            return SharedCandidateType::WeightsSharing;
        }
    default:
        return SharedCandidateType::Normal;
    }
}

ContractionOp::Analysis ContractionOp::Analyze(const BindingContext& ctx, const Graph& graph, const GraphHandle& interface) {
    int maxWeightId = 0;
    std::optional<Dimension> lastWeightLeader;
    int numShares = 0;
    Size numelOuter = Size::Identity(ctx);
    auto globalComp = Dimension::GlobalLessThan(graph);
    for (const ShareOp *shareOp: graph.getOpsOfType<ShareOp>()) {
        int newWeightId = shareOp->getRhsOrigin();
        const Dimension& newWeightDim = shareOp->output;
        if (maxWeightId < newWeightId) {
            maxWeightId = newWeightId;
            lastWeightLeader = newWeightDim;
        } else if (maxWeightId == newWeightId && globalComp(newWeightDim, lastWeightLeader.value())) {
            lastWeightLeader = newWeightDim;
        }
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
        .lastWeightLeader = std::move(lastWeightLeader),
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
    int lastWeight = 0;
    std::optional<Dimension> leader;
    auto globalComp = Dimension::GlobalLessThan(options.graph);
    for (std::size_t i = 0; i < available.size(); ++i) {
        const auto& assignment = assigned[i];
        if (!assignment.has_value()) continue;
        const auto& selection = available[i];
        lastWeight = std::max(lastWeight, selection.lastWeight);
        if (!leader || globalComp(selection.dim, *leader)) {
            leader = selection.dim;
        }
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

    // Canonical order of ContractionOp.
    if (lastWeight + 1 != options.weightId) {
        // Require the leader of weights to be ordered.
        if (!globalComp(options.lastWeightLeader.value(), leader.value())) {
            return nullptr;
        }
    }

    auto& store = options.store;
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
    return store.get<ContractionOp>(std::move(result));
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

std::vector<const ContractionOp *> ContractionOp::Generate(OperationStore& store, const GenerateOptions& options) {
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

    std::vector<CandidateDimension> available;
    KAS_ASSERT(fullDims.size() == candidateTypes.size());
    for (std::size_t i = 0; i < fullDims.size(); ++i) {
        const auto& color = graph.colorOf(fullDims[i]);
        if (color.isDataDiscarding()) continue;
        int lastWeight = ranges::fold_left(
            color.getTags()
            | std::views::transform([](const MergeLikeOp *op) { return dynamic_cast<const ShareOp *>(op)->getRhsOrigin(); }),
            0,
            [](int a, int b) { return std::max(a, b); }
        );
        available.emplace_back(fullDims[i], candidateTypes[i], lastWeight);
    }

    Enumerator::Options enumeratorOptions {
        .store = store,
        .ctx = options.ctx,
        .graph = graph,
        .weightId = nextWeightId,
        .lastWeightLeader = analysis.lastWeightLeader,
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

bool ContractionOp::NoMoreContractions(const Graph& graph, std::size_t maximumTensors) {
    int maxWeightId = 0;
    for (const ShareOp *shareOp: graph.getOpsOfType<ShareOp>()) {
        int newWeightId = shareOp->getRhsOrigin();
        if (maxWeightId < newWeightId) {
            maxWeightId = newWeightId;
        }
    }
    return maximumTensors <= maxWeightId + 1;
}

void ViewAndContraction::addView(const PrimitiveOp *op) {
    views.emplace_back(op);
}

void ViewAndContraction::addExpand(const ExpandOp *op) {
    if (auto share = op->output.tryAs<ShareOp::Input>(); share) {
        // This belongs to ContractionOp.
        auto shareOp = share->getDerivedOp<ShareOp>();
        auto [_, inserted] = dimwiseContractions.try_emplace(shareOp, op);
        KAS_ASSERT(inserted);
    } else {
        // This is a Repeat.
        addView(op);
    }
}

void ViewAndContraction::addShare(const ShareOp *op) {
    // This belongs to ContractionOp.
    auto [it, inserted] = dimwiseContractions.try_emplace(op, nullptr);
    // If the ShareOp is already included, it must have been added by ExpandOp.
    KAS_ASSERT(inserted || it->second != nullptr);
}

const ContractionOp *ViewAndContraction::toContractionOp(OperationStore& store) const {
    if (dimwiseContractions.empty()) return nullptr;
    std::vector<ContractionOp::Dimwise> dimwiseOps;
    for (const auto& [share, expand]: dimwiseContractions) {
        dimwiseOps.emplace_back(share, expand);
    }
    return store.get<ContractionOp>(std::move(dimwiseOps));
}

ContractionExtractor::ContractionExtractor(OperationStore& store, const Graph& graph):
    // Only the input in the beginning.
    DependentCutSetDiscoverer(graph, graph.getTopmost().getDimensions()),
    store { store } {}

void ContractionExtractor::afterExclusionHook(const PrimitiveOp *op) {
    if (auto expand = dynamic_cast<const ExpandOp *>(op); expand) {
        last().addExpand(expand);
    } else if (auto share = dynamic_cast<const ShareOp *>(op); share) {
        last().addShare(share);
    } else {
        last().addView(op);
    }
}

void ContractionExtractor::extract(const Topmost& bottommost) {
    std::map<int, std::vector<Dimension>> layers;
    for (const ShareOp *shareOp: graph.getOpsOfType<ShareOp>()) {
        layers[shareOp->getRhsOrigin()].emplace_back(shareOp->output);
    }
    KAS_ASSERT(layers.find(0) == layers.end());
    for (const Dimension& output: bottommost.getDimensions()) {
        layers[0].emplace_back(output);
    }
    for (const auto& [layer, dimensions]: layers | std::views::reverse) {
        alternatingLayers.emplace_back();
        include(dimensions);
        if (last().empty()) {
            alternatingLayers.pop_back();
        }
    }
    std::ranges::reverse(alternatingLayers);
    std::ranges::for_each(alternatingLayers, [](ViewAndContraction& layer) {
        std::ranges::reverse(layer.views);
    });
}

std::vector<const Operation *> ContractionExtractor::serialize() const {
    std::vector<const Operation *> result;
    for (const auto& layer: alternatingLayers) {
        if (auto contraction = layer.toContractionOp(store); contraction) {
            result.emplace_back(contraction);
        }
        std::ranges::copy(layer.views, std::back_inserter(result));
    }
    return result;
}

std::vector<std::vector<const Operation *>> ContractionExtractor::layers() const {
    std::vector<std::vector<const Operation *>> layers;
    for (const auto& lattice: alternatingLayers) {
        if (auto contraction = lattice.toContractionOp(store); contraction) {
            layers.emplace_back();
            layers.back().emplace_back(contraction);
        }
        if (!lattice.views.empty()) {
            layers.emplace_back();
            auto& layer = layers.back();
            for (const PrimitiveOp *view: lattice.views) {
                layer.emplace_back(view);
            }
        }
    }
    return layers;
}

} // namespace kas
