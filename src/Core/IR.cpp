#include "KAS/Core/Expand.hpp"
#include "KAS/Core/IR.hpp"
#include "KAS/Core/Pass.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

IR IR::Build(const std::vector<Topmost>& tensors, const BindingContext& ctx, bool constractOneTensorEachTime) {
    auto builder = IRBuilder(tensors);
    auto schemes = builder.plausibleContractionSchemes(constractOneTensorEachTime);

    // TODO: make this an option.
    constexpr std::size_t MaxVRAM = 40_uz * 1024 * 1024 * 1024;

    IR current;
    std::size_t optimal = std::numeric_limits<std::size_t>::max();
    bool OOM = true;
    for (auto scheme: schemes) {
        auto result = builder.build(scheme, ctx);

        auto thisOOM = result.getVRAMUsage(ctx) > MaxVRAM;
        auto flops = result.getFLOPs(ctx);

        if (!OOM && thisOOM) {
            // Since there is already a result that does not overflow, we keep it.
            continue;
        } else if (OOM && !thisOOM) {
            // This result does not OOM. We have to pick it.
            OOM = false;
            current = std::move(result);
            optimal = flops;
            continue;
        }
        // Otherwise, OOM == thisOOM. Just compare.

        if (flops < optimal) {
            current = std::move(result);
            optimal = flops;
        } else if (flops == optimal) {
            const auto currentStages = current.numStages(), resultStages = result.numStages();
            if (currentStages < resultStages) {
                // More stages enable schedulers to do more optimizations.
                current = std::move(result);
            } else if (currentStages == resultStages) {
                // TODO: Tie breaker.
                ++CountEqualFLOPs;
                // KAS_WARNING("Equal FLOPs {} and stages {} for two contraction schemes!", optimal, currentStages);
            }
        }
    }

    if (OOM) {
        ++CountVRAMExceeded;
    }
    if (!current) {
        // No, we cannot contract one tensor at a time.
        ++CountWithInterdependentShares;
        return Build(tensors, ctx, false);
    }

    return current;
}

IR IR::copy() const {
    std::map<Tensor, Tensor> oldToNew;
    auto dfs = [&](const auto& self, const Tensor& tensor) -> void {
        if (oldToNew.contains(tensor)) return;
        for (const Tensor& input: tensor.inputs()) {
            self(self, input);
        }
        Tensor newTensor = tensor.clone(oldToNew);
        auto [_, inserted] = oldToNew.try_emplace(tensor, newTensor);
        KAS_ASSERT(inserted);
    };
    dfs(dfs, outputTensor);
    auto newInputs = ranges::to<std::vector<Tensor>>(
        inputTensors
        | std::views::transform([&](const Tensor& t) {
            return oldToNew.at(t);
        })
    );
    return { expansions, std::move(newInputs), oldToNew.at(outputTensor) };
}

Graph IR::buildGraph() const {
    return GraphBuilder()
        .addDimensions(inputTensors | std::views::transform(&Tensor::output) | std::views::join)
        .addExpansions(expansions | std::views::join)
        .build();
}

std::size_t IR::getFLOPs(const BindingContext& ctx, const ConcreteConsts& consts) const {
    std::size_t flops = 0;
    bottomTopForEach([&](const Tensor& tensor) {
        flops += tensor.getFLOPs(ctx, consts);
    });
    return flops;
}
std::size_t IR::getFLOPs(const BindingContext& ctx) const {
    std::size_t flops = 0;
    bottomTopForEach([&](const Tensor& tensor) {
        flops += tensor.getFLOPs(ctx);
    });
    return flops;
}

std::size_t IR::getVRAMUsage(const BindingContext& ctx) const {
    std::size_t numel = 0;
    bottomTopForEach([&](const Tensor& tensor) {
        numel += (tensor.output().empty() ? Size::Identity(ctx) : ShapeView(tensor.output()).totalSize()).evalSumAllConsts(ctx);
    });
    // 2 for gradients.
    return 2 * numel * sizeof(float);
}

std::size_t IR::numStages() const {
    std::size_t stages = 0;
    bottomTopForEach([&](const Tensor& tensor) {
        stages += !tensor.isInputTensor();
    });
    return stages;
}

bool ContractionScheme::contractMoreThanOneTensorEachTime() const {
    return std::ranges::any_of(contractions, [](const std::vector<std::size_t>& contraction) {
        return contraction.size() > 1;
    });
}

Generator<ContractionScheme> IRBuilder::plausibleContractionSchemes(const std::vector<std::vector<bool>>& laterThan, std::vector<std::size_t> remaining) const {
    if (remaining.empty()) {
        co_yield ContractionScheme { };
        co_return;
    }

    // If j is later than k (laterThan[j][k] == true), then k must be in a former or the same contraction group as that of j.
    const std::size_t mask = (1_uz << remaining.size()) - 1;
    for (std::size_t i = 1; i <= mask; ++i) {
        std::vector<std::size_t> contraction, next;
        for (std::size_t j = 0; j < remaining.size(); ++j) {
            if ((i >> j) & 1) {
                contraction.emplace_back(remaining[j]);
            } else {
                next.emplace_back(remaining[j]);
            }
        }
        bool valid = true;
        for (std::size_t j: contraction) {
            for (std::size_t k: next) {
                if (laterThan[j][k]) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }
        if (valid) {
            for (ContractionScheme scheme: plausibleContractionSchemes(laterThan, std::move(next))) {
                scheme.cons(contraction);
                co_yield std::move(scheme);
            }
        }
    }
}

IR IRBuilder::initial(const ContractionScheme& scheme) const {
    std::vector<Tensor> inputs;
    for (const Topmost& topmost: inputTensors) {
        inputs.emplace_back(TensorImpl::CreateInput(topmost.getDimensions()));
    }

    Tensor current = inputs.at(0);
    auto contractor = TensorContractor(graph, current.output());
    bool isFirst = true;
    for (const auto& contraction: scheme.contractions) {
        KAS_ASSERT(!contraction.empty() || isFirst, "Only when we do early reduction, we can have an empty contraction group.");

        std::vector<Tensor> currentInputs = { current };

        // Add newly contracted tensors to inputs.
        for (std::size_t index: contraction) {
            currentInputs.emplace_back(inputs.at(index));
        }
        // Collect all the required dimensions.
        contractor.contract(
            currentInputs
            | std::views::drop(1) // skip the current tensor.
            | std::views::transform(&Tensor::output)
        );

        // And apply reductions.
        contractor.reduce();
        current = TensorImpl::CreateView(currentInputs, Bottommost(contractor.build()));

        // Remove all the reductions, so we can continue to the next contraction.
        contractor.removeReductions();

        isFirst = false;
    }

    // If we have not reached the output, perform required views. Note that by this time we have done all reductions.
    bool allDimensionsCollected = DimensionSetEqual(contractor.build(), graph.getOutputIterators());
    if (
        !allDimensionsCollected ||
        // Special case where we do not even need to perform any contraction, reduction or view.
        (allDimensionsCollected && current.isInputTensor() && !std::ranges::equal(current.output(), graph.getOutputIterators()))
    ) {
        contractor.fill();
        auto finalOutput = Bottommost(contractor.build());
        KAS_ASSERT(DimensionSetEqual(finalOutput.getOutput(), graph.getOutputIterators()));
        current = TensorImpl::CreateView(std::vector<Tensor>{current}, std::move(finalOutput));
    }

    // We do not need to adjust the layout here, because rfactor pass will overwrite that anyway.
    // We adjust the output layout in layout pass.

    auto expansions = ranges::to<std::vector<std::vector<const Expand *>>>(
        inputTensors
        | std::views::transform(static_cast<const std::vector<const Expand *>& (Topmost::*)() const>(&Topmost::getExpansions))
    );
    return { std::move(expansions), std::move(inputs), std::move(current) };
}

void IRBuilder::rfactor(IR& ir, const BindingContext& ctx) const {
    (RFactorIRPass(ctx, graph))(ir);
}

void IRBuilder::optimizeLayout(IR& ir) const {
    (OptimizeLayoutIRPass(graph))(ir);
}

IRBuilder::IRBuilder(const std::vector<Topmost>& tensors):
    inputTensors { tensors },
    graph(GraphBuilder().addTopmosts(tensors).build())
{}

Generator<ContractionScheme> IRBuilder::plausibleContractionSchemes(bool constractOneTensorEachTime) const {
    const std::size_t numTensors = inputTensors.size();

    std::vector<Graph::CompactIndices> tensorFeatures;
    for (const auto& topmost: inputTensors) {
        tensorFeatures.emplace_back(graph.getAncestors(topmost.getAllDimensions()));
    }
    const auto& inputFeatures = tensorFeatures.at(0);

    // Create a dictionary.
    std::map<std::size_t, std::size_t> dimToOrigin;
    for (std::size_t i = 0; i < numTensors; ++i) {
        tensorFeatures[i].foreach([&](std::size_t index) {
            auto [it, inserted] = dimToOrigin.try_emplace(index, i);
            KAS_ASSERT(inserted || it->second == i);
        });
    }
    auto getOrigin = [&](const Dimension& dim) {
        std::set<std::size_t> origin;
        const auto dimFeatures = graph.getAncestors(dim);
        dimFeatures.foreach([&](std::size_t index) {
            origin.insert(dimToOrigin.at(index));
        });
        return origin;
    };

    bool earlyReduction = std::ranges::any_of(graph.getReduceIterators(), [&](const Reduce *reduction) {
        return inputFeatures.contains(graph.getAncestors(reduction));
    });

    std::vector<std::vector<bool>> laterThan(numTensors, std::vector<bool>(numTensors));
    // The first tensor must be the earliest, which is by default the case.
    for (const MergeLikeOp *op: graph.getOpsOfType<MergeLikeOp>(DimensionType::Share)) {
        const Dimension lhs = op->getInputL(), rhs = op->getInputR();
        const auto lhsFeatures = graph.getAncestors(lhs), rhsFeatures = graph.getAncestors(rhs);

        // Required by canonicalization.
        KAS_ASSERT(lhsFeatures.intersects(inputFeatures) && rhsFeatures.disjoint(inputFeatures));

        // First find the tensor that we are contracting.
        const auto rhsOriginSet = getOrigin(rhs);
        KAS_ASSERT(rhsOriginSet.size() == 1); // required by canonicalization.
        const auto rhsOrigin = *rhsOriginSet.begin();
        // Then the existing contracted tensors.
        const auto lhsOriginSet = getOrigin(lhs);
        for (std::size_t lhsOrigin: lhsOriginSet) {
            // Require that rhs be contracted later than lhs.
            laterThan[rhsOrigin][lhsOrigin] = true;
        }
    }

    for (ContractionScheme scheme:
        plausibleContractionSchemes(
            laterThan,
            ranges::to<std::vector<std::size_t>>(std::views::iota(1_uz, numTensors))
        )
    ) {
        if (earlyReduction) {
            scheme.contractions.insert(scheme.contractions.begin(), std::vector<std::size_t>{});
        }
        if (constractOneTensorEachTime && scheme.contractMoreThanOneTensorEachTime()) {
            continue;
        }
        co_yield std::move(scheme);
    }

    co_return;
}

IR IRBuilder::build(const ContractionScheme& scheme, const BindingContext& ctx) const {
    auto ir = initial(scheme);
    rfactor(ir, ctx);
    optimizeLayout(ir);
    return ir;
}

} // namespace kas
