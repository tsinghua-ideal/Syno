#include "KAS/Search/Node.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Stage.hpp"


namespace kas {

std::string Next::toString() const {
    return fmt::format("{}({})", type, key);
}

std::string Next::description(const Node& node) const {
    const BindingContext& ctx = node.sampler->getBindingContext();
    return node.match<std::string>(
        [&](ReductionStage *rStage) {
            KAS_ASSERT(type == Type::MapReduce);
            return rStage->getChildDescription(key);
        },
        [&](Stage *stage) {
            switch (type) {
            case Type::Shift: return stage->getChildSlot<ShiftOp>(key).op->description(ctx);
            case Type::Stride: return stage->getChildSlot<StrideOp>(key).op->description(ctx);
            case Type::Split: return stage->getChildSlot<SplitOp>(key).op->description(ctx);
            case Type::Unfold: return stage->getChildSlot<UnfoldOp>(key).op->description(ctx);
            case Type::Merge: return stage->getChildSlot<MergeOp>(key).op->description(ctx);
            case Type::Share: return stage->getChildSlot<ShareOp>(key).op->description(ctx);
            case Type::Finalize: return stage->getChildFinalizeSlot(key).finalization.description(ctx);
            default: KAS_UNREACHABLE();
            }
        },
        [](std::shared_ptr<TensorView> tensor) -> std::string { KAS_UNREACHABLE(); }
    );
}

std::map<Next::Type, std::size_t> Next::CountTypes(const std::vector<Next>& nexts) {
    std::map<Type, std::size_t> result;
    for (auto&& next: nexts) {
        ++result[next.type];
    }
    return result;
}

std::shared_ptr<TensorView> Node::asFinal() const {
    return std::get<std::shared_ptr<TensorView> >(inner);
}

std::unique_ptr<Kernel> Node::realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options) const {
    auto final = asFinal();
    if (!final) {
        return nullptr;
    }
    return std::make_unique<Kernel>(*final, sampler->getBindingContext(), allMappings, std::move(options));
}

std::size_t Node::estimateTotalFLOPsAsFinal() const {
    auto final = asFinal();
    const auto& allConsts = sampler->getBindingContext().getAllConsts();
    std::size_t result = 0;
    for (const auto& consts: allConsts) {
        result += final->getFLOPs(consts);
    }
    return result;
}

std::size_t Node::countChildren() const {
    return match<std::size_t>(
        [](ReductionStage *rStage) { return rStage->countChildren(); },
        [](Stage *stage) { return stage->countChildren(); },
        [](std::shared_ptr<TensorView> tensor) { return 0; }
    );
}

std::vector<Next> Node::getChildrenHandles() const {
    return match<std::vector<Next>>(
        [](ReductionStage *rStage) { return rStage->getChildrenHandles(); },
        [](Stage *stage) { return stage->getChildrenHandles(); },
        [](std::shared_ptr<TensorView> tensor) { return std::vector<Next>{}; }
    );
}

Node Node::getChild(Next next) const {
    return match<Node>(
        [&](ReductionStage *rStage) { return rStage->getChild(next); },
        [&](Stage *stage) { return stage->getChild(next); },
        [](std::shared_ptr<TensorView> tensor) -> Node { KAS_UNREACHABLE(); }
    );
}

std::string Node::toString() const {
    const BindingContext& ctx = sampler->getBindingContext();
    return match<std::string>(
        [](ReductionStage *rStage) { return rStage->description(); },
        [](Stage *stage) { return stage->description(); },
        [&](std::shared_ptr<TensorView> tensor) { return tensor->description(ctx); }
    );
}

} // namespace kas
