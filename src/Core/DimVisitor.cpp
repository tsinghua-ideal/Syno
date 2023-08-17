#include "KAS/Core/DimVisitor.hpp"


namespace kas {

void Iterator::accept(DimVisitor& visitor) const {
    visitor.visit(*this);
}
void Reduce::accept(DimVisitor& visitor) const {
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
void DimVisitor::visit(const Reduce& dim) {}
void DimVisitor::visit(const RepeatLikeOp::Input& dim) {
    dim.getOp()->output.accept(*this);
}
void DimVisitor::visit(const SplitLikeOp::Input& dim) {
    dim.getOp()->outputLhs.accept(*this);
    dim.getOp()->outputRhs.accept(*this);
}
void DimVisitor::visit(const MergeLikeOp::Input& dim) {
    dim.getOp()->output.accept(*this);
}

} // namespace kas
