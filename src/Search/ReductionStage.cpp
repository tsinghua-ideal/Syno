#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

ReductionStage::ReductionStage(Sampler& sampler, std::vector<const MapReduceOp *>&& reductions):
    sampler { sampler },
    reductions { std::move(reductions) }
{
    const auto& options = sampler.getOptions();

    // First create the corresponding NormalStage.
    nStage = std::make_unique<NormalStage>(sampler, this->reductions);

    // Then attempt to generate new reductions.
    if (this->reductions.size() == options.maximumReductions) {
        return;
    }
    auto nextReductions = MapReduceOp::Generate(sampler.getReductionStore(), this->reductions, {
        .ctx = sampler.getBindingContext(),
        .dimUpperBound = options.dimUpperBound,
        .outputSize = sampler.getTotalOutputSize(),
        .maxFLOPs = options.maxFLOPs,
    });
    this->nextReductions.fill(nextReductions, [&](const MapReduceOp *op) -> NextReductionSlot {
        return NextReductionSlot({NextReductionSlot::GetKey(op)}, std::make_unique<ReductionStage>(*this, op));
    });
}

std::size_t ReductionStage::hash() const {
    using namespace std::string_view_literals;
    auto h = std::hash<std::string_view>{}("ReductionStage"sv);
    HashCombine(h, reductions.size());
    for (const auto *op: reductions) {
        HashCombineRaw(h, op->hash());
    }
    return h;
}

Interface ReductionStage::toInterface() const {
    auto interface = sampler.getRootInterface();
    std::ranges::copy(reductions, std::back_inserter(interface));
    return interface;
}

std::vector<Next> ReductionStage::getChildrenHandles() const {
    std::vector<Next> handles = nextReductions.toNexts();
    handles.emplace_back(Next::Type::MapReduce, StopReductionToken);
    return handles;
}

const ReductionStage::NextReductionSlot& ReductionStage::getChildSlot(std::size_t key) const {
    return nextReductions.getSlot(key);
}

Node ReductionStage::getChild(Next next) const {
    KAS_ASSERT(next.type == Next::Type::MapReduce);
    if (next.key == StopReductionToken) {
        return Node { &sampler, nStage.get() };
    }
    return Node { &sampler, getChildSlot(next.key).next.get() };
}

std::string ReductionStage::getChildDescription(std::size_t key) const {
    if (key == StopReductionToken) {
        return "StopGeneratingReductions";
    }
    return getChildSlot(key).next->lastReduction()->description(sampler.getBindingContext());
}

std::string ReductionStage::description() const {
    return nStage->description();
}

} // namespace kas
