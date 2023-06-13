#include "KAS/Search/Node.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Stage.hpp"
#include "KAS/Utils/Ranges.hpp"


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

bool Node::operator==(const Node& rhs) const {
    if (inner.index() != rhs.inner.index()) {
        return false;
    }
    return match<bool>(
        [&](ReductionStage *rStage) { // Because we have uniquified them.
            return rStage == std::get<ReductionStage *>(rhs.inner);
        },
        [&](Stage *stage) { // Because we have uniquified them.
            return stage == std::get<Stage *>(rhs.inner);
        },
        [&](std::shared_ptr<TensorView> tensor) {
            return *tensor == *std::get<std::shared_ptr<TensorView>>(rhs.inner);
        }
    );
}

std::size_t Node::hash() const {
    using namespace std::string_view_literals;
    auto h = std::hash<std::string_view>{}("Node"sv);
    HashCombine(h, inner.index());
    auto hashRetriever = [](const auto& content) {
        return content->hash();
    };
    auto contentHash = std::visit(hashRetriever, inner);
    HashCombineRaw(h, contentHash);
    return h;
}

std::shared_ptr<TensorView> Node::asFinal() const {
    return std::get<std::shared_ptr<TensorView>>(inner);
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

void Node::generateGraphviz(const std::string& dir, const std::string& name) const {
    const auto& ctx = sampler->getBindingContext();
    match<void>(
        [&](ReductionStage *rStage) {
            auto interface = rStage->toInterface();
            GraphvizGen gen { interface, ctx };
            gen.generate(dir, name);
        },
        [&](Stage *stage) {
            auto interface = ranges::to<Interface>(stage->getInterface().toDimensions());
            GraphvizGen gen { interface, ctx };
            gen.generate(dir, name);
        },
        [&](std::shared_ptr<TensorView>) -> void {
            generateGraphvizAsFinal(dir, name);
        }
    );
}

void Node::generateGraphvizAsFinal(const std::string& dir, const std::string& name) const {
    auto final = asFinal();
    GraphvizGen gen { *final, sampler->getBindingContext() };
    gen.generate(dir, name);
}

std::string Node::getNestedLoopsAsFinal() const {
    auto final = asFinal();
    return final->printNestedLoopsForAll(sampler->getBindingContext());
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
