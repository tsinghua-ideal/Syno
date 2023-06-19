#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ReductionStore.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

ReductionStore::ReductionStore(PrimitiveOpStore& store, const std::vector<const MapReduce *>& current, const MapReduceOp::GenerateOptions& options) {
    if (current.size() == options.maximumReductions) {
        return;
    }
    auto nextReductions = MapReduceOp::Generate(store, current, options);
    this->nextReductions.fill(nextReductions, [&](const MapReduceOp *op) -> NextReductionSlot {
        auto nextCurrent = current;
        nextCurrent.push_back(op->getRaw());
        return NextReductionSlot({NextReductionSlot::GetKey(op)}, op, std::make_unique<ReductionStore>(store, nextCurrent, options));
    });
    this->nextReductions.checkHashCollisionAndRemove();
}

std::vector<const MapReduceOp *> ReductionStore::retrieve(std::span<const MapReduce * const> current) const {
    if (current.empty()) {
        return ranges::to<std::vector<const MapReduceOp *>>(nextReductions.getRawSlots() | std::views::transform(&NextReductionSlot::op));
    } else {
        return nextReductions.findSlot(current[0]->hash())->nextStore->retrieve(current.subspan(1));
    }
}

} // namespace kas