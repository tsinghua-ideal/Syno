#include "KAS/Search/Node.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Stage.hpp"


namespace kas {

std::string Next::toString() const {
    return fmt::format("{}({})", type, key);
}

std::string Next::description(const Node& node) const {
    const BindingContext& ctx = node.sampler->getBindingContext();
    return node.match<std::string>(
        [&]() {
            return fmt::format("{}",
                fmt::join(node.sampler->getReduce(key)
                | std::views::transform([&](const MapReduceOp& op) {
                    return op.description(ctx);
                }), ", ")
            );
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
        [](TensorView *tensor) -> std::string { KAS_UNREACHABLE(); }
    );
}

std::map<Next::Type, std::size_t> Next::CountTypes(const std::vector<Next>& nexts) {
    std::map<Type, std::size_t> result;
    for (auto&& next: nexts) {
        ++result[next.type];
    }
    return result;
}

TensorView *Node::asFinal() const {
    return std::get<TensorView *>(inner);
}

std::unique_ptr<Kernel> Node::realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options) const {
    auto final = asFinal();
    if (!final) {
        return nullptr;
    }
    return std::make_unique<Kernel>(*final, sampler->getBindingContext(), allMappings, std::move(options));
}

std::size_t Node::countChildren() const {
    return match<std::size_t>(
        [&]() { return sampler->getBaseCount(); },
        [](Stage *stage) { return stage->countChildren(); },
        [](TensorView *tensor) { return 0; }
    );
}

std::vector<Next> Node::getChildrenHandles() const {
    return match<std::vector<Next>>(
        [&]() { return sampler->getNextBases(); },
        [](Stage *stage) { return stage->getChildrenHandles(); },
        [](TensorView *tensor) { return std::vector<Next>{}; }
    );
}

Node Node::getChild(Next next) const {
    return match<Node>(
        [&]() { return Node { sampler, sampler->getBase(next.key) }; },
        [&](Stage *stage) { return stage->getChild(next); },
        [](TensorView *tensor) -> Node { KAS_UNREACHABLE(); }
    );
}

std::string Node::toString() const {
    const BindingContext& ctx = sampler->getBindingContext();
    return match<std::string>(
        [&]() { return sampler->getOutputShape().toString(ctx); },
        [&](Stage *stage) { return stage->description(ctx); },
        [&](TensorView *tensor) { return tensor->description(ctx); }
    );
}

} // namespace kas
