#pragma once

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"

namespace kas {

class DimVisitor {
public:
    virtual void visit(const Iterator& dim);
    virtual void visit(const MapReduceOp& dim);
    virtual void visit(const RepeatLikeOp::Input& dim);
    virtual void visit(const SplitLikeOp::Input& dim);
    virtual void visit(const MergeLikeOp::Input& dim);
    inline void visit(const Dimension& dim) {
        dim.getInnerPointer()->accept(*this);
    }
};

} // namespace kas
