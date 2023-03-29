#include "KAS/Core/DimVisitor.hpp"


namespace kas {

void Iterator::accept(DimVisitor& visitor) const {
    visitor.visit(*this);
}
void MapReduceOp::accept(DimVisitor& visitor) const {
    visitor.visit(*this);
}
void RepeatLikeOp::Input::accept(DimVisitor& visitor) const {
    visitor.visit(*this);
}
void SplitLikeOp::Input::accept(DimVisitor& visitor) const {
    visitor.visit(*this);
}
void MergeLikeOp::Input::accept(DimVisitor& visitor) const {
    visitor.visit(*this);
}

void DimVisitor::visit(const Iterator& dim) {}
void DimVisitor::visit(const MapReduceOp& dim) {}
void DimVisitor::visit(const RepeatLikeOp::Input& dim) {}
void DimVisitor::visit(const SplitLikeOp::Input& dim) {}
void DimVisitor::visit(const MergeLikeOp::Input& dim) {}

} // namespace kas
