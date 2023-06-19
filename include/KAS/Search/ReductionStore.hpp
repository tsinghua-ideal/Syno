#pragma once

#include "KAS/Core/MapReduce.hpp"
#include "KAS/Transforms/MapReduce.hpp"
#include "KAS/Search/Node.hpp"


namespace kas {

class ReductionStore;

struct NextReductionSlot: NextSlot<Next::Type::MapReduce> {
    const MapReduceOp *op;
    std::unique_ptr<ReductionStore> nextStore;
    static std::size_t GetKey(const MapReduceOp *op) { return op->opHash(); }
};

class ReductionStore {
    NextSlotStore<NextReductionSlot> nextReductions;

public:
    ReductionStore(PrimitiveOpStore& store, const std::vector<const MapReduce *>& current, const MapReduceOp::GenerateOptions& options);

    std::vector<const MapReduceOp *> retrieve(std::span<const MapReduce * const> current) const;
};

} // namespace kas
