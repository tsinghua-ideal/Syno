#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

void ReductionStage::expand() {
    const auto& options = sampler.getOptions();

    // First create the corresponding NormalStage.
    nStage = std::make_unique<NormalStage>(toInterface(), *this, std::nullopt);

    // Then attempt to generate new reductions.
    if (reductions.size() == options.maximumReductions) {
        return;
    }
    auto nextReductions = MapReduceOp::Generate(sampler.getReductionStore(), reductions, {
        .ctx = sampler.getBindingContext(),
        .dimUpperBound = options.dimUpperBound,
        .outputSize = sampler.getTotalOutputSize(),
        .maxFLOPs = options.maxFLOPs,
    });
    this->nextReductions.fill(nextReductions, [&](const MapReduceOp *op) -> NextReductionSlot {
        return NextReductionSlot({NextReductionSlot::GetKey(op)}, std::make_unique<ReductionStage>(*this, op));
    });
}

void ReductionStage::removeDeadChildrenFromSlots() {
    nextReductions.remove([&](const NextReductionSlot& slot) {
        return slot.next->getFinalizability() == Finalizability::No;
    });
}

void ReductionStage::removeAllChildrenFromSlots() {
    KAS_ASSERT(nStage->getFinalizability() == Finalizability::No);
    nextReductions.clear();
}

AbstractStage::Finalizability ReductionStage::checkForFinalizableChildren() const {
    bool allNo = true;
    bool foundYes = false;
    auto acc = [&](Finalizability fin) {
        if (fin == Finalizability::Yes) {
            foundYes = true;
            allNo = false;
        } else if (fin == Finalizability::Maybe) {
            allNo = false;
        }
    };
    acc(nStage->getFinalizability());
    nextReductions.forEach([&](const NextReductionSlot& slot) {
        if (foundYes) {
            return;
        }
        acc(slot.next->getFinalizability());
    });
    if (foundYes) {
        return Finalizability::Yes;
    } else if (allNo) {
        return Finalizability::No;
    } else {
        return Finalizability::Maybe;
    }
}

std::size_t ReductionStage::uncheckedCountChildren() {
    return nextReductions.size() + nStage->countChildren();
}

std::vector<Next> ReductionStage::uncheckedGetChildrenHandles() {
    std::vector<Next> handles = nextReductions.toNexts();
    std::ranges::move(nStage->getChildrenHandles(), std::back_inserter(handles));
    return handles;
}

const ReductionStage::NextReductionSlot& ReductionStage::getChildSlot(std::size_t key) {
    return nextReductions.getSlot(key);
}

Node ReductionStage::uncheckedGetChild(Next next) {
    if(next.type == Next::Type::MapReduce) {
        return Node { &sampler, getChildSlot(next.key).next.get() };
    } else {
        return nStage->getChild(next);
    }
}

std::string ReductionStage::uncheckedGetChildDescription(Next next) {
    if (next.type == Next::Type::MapReduce) {
        return getChildSlot(next.key).next->lastReduction()->description(sampler.getBindingContext());
    } else {
        return nStage->getChildDescription(next);
    }
}

ReductionStage::ReductionStage(ReductionStage& current, const MapReduceOp *nextReduction):
    AbstractStage { current, Next::Type::MapReduce }
{
    reductions = current.reductions;
    reductions.emplace_back(nextReduction);
    expand();
}

ReductionStage::ReductionStage(Sampler& sampler):
    AbstractStage { sampler }
{
    expand();
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
    std::ranges::sort(interface, Dimension::HashLessThan{});
    return interface;
}

std::size_t ReductionStage::countChildren() {
    return guarded([this] { return uncheckedCountChildren(); });
}

std::vector<Next> ReductionStage::getChildrenHandles() {
    return guarded([this] { return uncheckedGetChildrenHandles(); });
}

Node ReductionStage::getChild(Next next) {
    return guarded([=, this] { return uncheckedGetChild(next); });
}

std::string ReductionStage::getChildDescription(Next next) {
    return guarded([=, this] { return uncheckedGetChildDescription(next); });
}

std::string ReductionStage::description() const {
    return nStage->description();
}

} // namespace kas
