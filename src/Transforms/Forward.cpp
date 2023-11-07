#include "KAS/Core/Graph.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Transforms/Forward.hpp"


namespace kas {

namespace Forward {

void Expand::notifyParent() {
    KAS_ASSERT(evaluated());
    BackwardDimension outputDim = get();
    op = factory.getStore().get<::kas::ExpandOp>(outputDim);
}

const ::kas::ExpandOp *Expand::getOp() const {
    KAS_ASSERT(op);
    return op;
}

std::string Dimension::sizeToString() const {
    return getSize().toString(inner->getFactory().getBindingContext());
}

void Dimension::output(std::size_t index) {
    set(getFactory().createIterator(getSize(), index));
}

void Dimension::reduce(Reduce::ReduceType reduceType) {
    set(getFactory().createReduce(getSize(), reduceType));
}

const Iterator *Factory::createIterator(const Size& domain, std::size_t index) {
    auto it = std::make_unique<Iterator>(index, domain);
    auto ptr = it.get();
    iterators.emplace_back(std::move(it));
    bottommost.emplace_back(ptr);
    return ptr;
}
const Reduce *Factory::createReduce(const Size& domain, Reduce::ReduceType reduceType) {
    auto op = store.get<::kas::ReduceOp>(domain, reduceType);
    auto multiplicity = op->getMultiplicity(bottommost);
    auto backDim = op->getInput(multiplicity);
    bottommost.emplace_back(backDim);
    return dynamic_cast<const Reduce *>(backDim.getInnerPointer());
}

void Factory::inputs(const std::vector<std::vector<Dimension>>& tensors) {
    KAS_ASSERT(std::ranges::all_of(tensors, [](const auto& t) { return !t.empty(); }));

    KAS_ASSERT(topmosts.empty(), "You must not call inputs() twice!");
    topmosts.resize(tensors.size());

    std::map<Dimension, int> dimToTensorId;
    for (int i = 0; i < tensors.size(); ++i) {
        for (const Dimension& dim: tensors[i]) {
            dimToTensorId.try_emplace(dim, i);
        }
    }
    std::vector<int> tensorIdToFinalTensorId(tensors.size(), -1);
    tensorIdToFinalTensorId[0] = 0;

    while (!unresolvedShareOps.empty()) {
        // First find next rhsOrigin.
        const Graph graph = GraphBuilder().addDimensions(
            unresolvedShareOps
            | std::views::transform([](const std::pair<BackwardDimension, ShareOp *>& pair) {
                return pair.first;
            })
        ).build();
        auto origins = ::kas::ShareOp::GetRhsOrigins(graph);
        int nextRhsOrigin = origins.size() + 1;
        KAS_ASSERT(origins.empty() || *origins.rbegin() == nextRhsOrigin - 1);

        // Then find the tensor to label.
        std::optional<std::pair<BackwardDimension, ShareOp *>> least;
        auto comp = BackwardDimension::GlobalLessThan(graph);
        for (const std::pair<BackwardDimension, ShareOp *>& pair: unresolvedShareOps) {
            const auto& [dim, share] = pair;
            if (!least || comp(dim, least->first)) {
                least = pair;
            }
        }
        auto selectedWeightDim = least->second->getInputRhs();
        int targetTensorId = dimToTensorId.at(selectedWeightDim);
        int& finalTensorId = tensorIdToFinalTensorId[targetTensorId];
        KAS_ASSERT(finalTensorId == -1);
        finalTensorId = nextRhsOrigin;

        // Now label the tensors.
        while (true) {
            bool hasProgress = false;
            for (auto it = unresolvedShareOps.begin(); it != unresolvedShareOps.end(); ++it) {
                auto [dim, share] = *it;
                int tensorId = dimToTensorId.at(share->getInputRhs());
                int finalTensorId = tensorIdToFinalTensorId.at(tensorId);
                if (finalTensorId != -1) {
                    unresolvedShareOps.erase(it);
                    hasProgress = true;
                    share->setRhsOrigin(finalTensorId);
                    share->proceedNotification(*this);
                    break;
                }
            }
            if (!hasProgress) break;
        }
    }

    {
        // Sanity check.
        auto finalTensorIds = tensorIdToFinalTensorId;
        std::ranges::sort(finalTensorIds);
        KAS_ASSERT(std::ranges::equal(finalTensorIds, std::views::iota(0, static_cast<int>(finalTensorIds.size()))));
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        int finalTensorId = tensorIdToFinalTensorId[i];
        for (const Dimension& dim: tensors[i]) {
            if (auto expand = dim.asExpanded(); expand) {
                topmosts[finalTensorId].getExpansions().emplace_back(expand->getOp());
            } else {
                topmosts[finalTensorId].getDimensions().emplace_back(dim);
            }
        }
    }
    // Sort the dimensions by hash.
    // TODO: what if we change the order, due to performance considerations?
    topmosts[0].sortExpansions();
    std::ranges::for_each(topmosts | std::views::drop(1), [](Topmost& topmost) { topmost.sort(); });
}

TensorView& Factory::buildTensorView(TensorExpression blending) {
    KAS_ASSERT(!this->result, "Factory must not be used twice!");
    KAS_ASSERT(!topmosts.empty(), "You must first call inputs() before building!");
    this->result = std::make_unique<TensorView>(topmosts, std::move(blending), ctx);
    return *this->result;
}

void MergeOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    auto op = factory.getStore().get<::kas::MergeOp>(outputDim, inputRhs.getSize());
    inputLhs.set(op->getInputL());
    inputRhs.set(op->getInputR());
}
Dimension MergeOp::Create(const Dimension& lhs, const Dimension& rhs) {
    auto op = std::unique_ptr<MergeOp> { new MergeOp { lhs, rhs } };
    auto output = Output::Create(lhs.getFactory(), lhs.getSize() * rhs.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void ShareOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    factory.registerUnresolvedShareOp(outputDim, this);
}
void ShareOp::proceedNotification(Factory& factory) {
    KAS_ASSERT(rhsOrigin >= 1);
    BackwardDimension outputDim = output.lock()->get();
    auto op = factory.getStore().get<::kas::ShareOp>(outputDim, rhsOrigin);
    inputLhs.set(op->getInputL());
    inputRhs.set(op->getInputR());
}
Dimension ShareOp::Create(const Dimension& lhs, const Dimension& rhs) {
    auto op = std::unique_ptr<ShareOp> { new ShareOp { lhs, rhs } };
    KAS_ASSERT(lhs.getSize() == rhs.getSize(), "Shared dimensions must be the same size.");
    auto output = Output::Create(lhs.getFactory(), lhs.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void ShiftOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    input.set(factory.getStore().get<::kas::ShiftOp>(outputDim, shift)->getInput());
}
Dimension ShiftOp::Create(const Dimension& input, int shift) {
    auto op = std::unique_ptr<ShiftOp> { new ShiftOp { input, shift } };
    auto output = Output::Create(input.getFactory(), input.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void SplitOp::onNotification(Factory& factory) {
    if (outputLhs.lock()->evaluated() && outputRhs.lock()->evaluated()) {
        BackwardDimension outputLhsDim = outputLhs.lock()->get();
        BackwardDimension outputRhsDim = outputRhs.lock()->get();
        input.set(factory.getStore().get<::kas::SplitOp>(outputLhsDim, outputRhsDim)->getInput());
    }
}
std::pair<Dimension, Dimension> SplitOp::Create(const Dimension& input, const Size& block) {
    auto op = std::shared_ptr<SplitOp> { new SplitOp { input } };
    auto outputLhs = Output::Create(input.getFactory(), input.getSize() / block, op, Order::Left);
    auto outputRhs = Output::Create(input.getFactory(), block, op, Order::Right);
    return { Dimension(std::move(outputLhs)), Dimension(std::move(outputRhs)) };
}

void StrideOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    input.set(factory.getStore().get<::kas::StrideOp>(outputDim, stride)->getInput());
}
Dimension StrideOp::Create(const Dimension& input, const Size& stride) {
    auto op = std::unique_ptr<StrideOp> { new StrideOp { input, stride } };
    auto output = Output::Create(input.getFactory(), input.getSize() / stride, std::move(op));
    return Dimension(std::move(output));
}

void UnfoldOp::onNotification(Factory& factory) {
    if (outputLhs.lock()->evaluated() && outputRhs.lock()->evaluated()) {
        BackwardDimension outputLhsDim = outputLhs.lock()->get();
        BackwardDimension outputRhsDim = outputRhs.lock()->get();
        input.set(factory.getStore().get<::kas::UnfoldOp>(outputLhsDim, outputRhsDim)->getInput());
    }
}
std::pair<Dimension, Dimension> UnfoldOp::Create(const Dimension& input, const Size& window) {
    auto op = std::shared_ptr<UnfoldOp> { new UnfoldOp { input } };
    auto outputLhs = Output::Create(input.getFactory(), input.getSize(), op, Order::Left);
    auto outputRhs = Output::Create(input.getFactory(), window, op, Order::Right);
    return { Dimension(std::move(outputLhs)), Dimension(std::move(outputRhs)) };
}

} // namespace Forward

} // namespace kas
