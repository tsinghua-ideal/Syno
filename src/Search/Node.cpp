#include "KAS/Search/Node.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

std::string Next::toString() const {
    return fmt::format("{}({})", type, key);
}

std::map<Next::Type, std::size_t> Next::CountTypes(const std::vector<Next>& nexts) {
    std::map<Type, std::size_t> result;
    for (auto&& next: nexts) {
        ++result[next.type];
    }
    return result;
}

bool Arc::operator==(const Arc& rhs) const {
    if (inner.index() != rhs.inner.index()) {
        return false;
    }
    return match<bool>(
        [&](const PrimitiveOp *op) {
            return PrimitiveOpEqual{}(op, rhs.as<PrimitiveOp>());
        },
        [&](const FinalizeOp *op) {
            return *op == *rhs.as<FinalizeOp>();
        }
    );
}
std::size_t Arc::hash() const {
    using namespace std::string_view_literals;
    static const auto arcHash = std::hash<std::string_view>{}("Arc"sv);
    auto h = arcHash;
    HashCombine(h, inner.index());
    auto contentHash = match<std::size_t>(
        [](const PrimitiveOp *op) {
            return op->opHash();
        },
        [](const FinalizeOp *op) {
            return op->hash();
        }
    );
    HashCombineRaw(h, contentHash);
    return h;
}
Next Arc::toNext() const {
    return match<Next>(
        [&](const PrimitiveOp *op) -> Next {
            return Next::FromOp(op);
        },
        [&](const FinalizeOp *op) -> Next {
            return { Next::Type::Finalize, op->hash() };
        }
    );
}
std::string Arc::toString() const {
    const auto& ctx = sampler->getBindingContext();
    return match<std::string>(
        [&](auto op) -> std::string {
            return op->description(ctx);
        },
        [&](auto op) -> std::string {
            return op->description(ctx);
        }
    );
}

bool Node::operator==(const Node& rhs) const {
    if (inner.index() != rhs.inner.index()) {
        return false;
    }
    return match<bool>(
        [&](ReductionStage *rStage) { // Because we have uniquified them.
            return rStage == std::get<ReductionStage *>(rhs.inner);
        },
        [&](NormalStage *nStage) { // Because we have uniquified them.
            return nStage == std::get<NormalStage *>(rhs.inner);
        },
        [&](FinalStage *fStage) { // Because it is uniquely determined by the previous stage.
            return fStage == std::get<FinalStage *>(rhs.inner);
        }
    );
}

std::size_t Node::hash() const {
    using namespace std::string_view_literals;
    static const auto nodeHash = std::hash<std::string_view>{}("Node"sv);
    auto h = nodeHash;
    HashCombine(h, inner.index());
    auto hashRetriever = [](const auto& content) {
        return content->hash();
    };
    auto contentHash = std::visit(hashRetriever, inner);
    HashCombineRaw(h, contentHash);
    return h;
}

FinalStage *Node::asFinalStage() const {
    return std::get<FinalStage *>(inner);
}

std::unique_ptr<Kernel> Node::realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, CodeGenOptions options, const std::filesystem::path& directory, const std::string& name) const {
    auto final = asFinalStage();
    return std::make_unique<Kernel>(sampler->getBindingContext(), final->value, allMappings, std::move(options), directory, name);
}

std::size_t Node::estimateTotalFLOPsAsFinal() const {
    auto final = asFinalStage();
    return final->value.getFLOPs(sampler->getBindingContext());
}

void Node::generateGraphviz(const std::filesystem::path& path, const std::string& name) const {
    const auto& ctx = sampler->getBindingContext();
    match<void>(
        [&](ReductionStage *rStage) {
            GraphvizGen gen { rStage->getInterface().getRaw(), ctx };
            gen.generate(path, name);
        },
        [&](NormalStage *nStage) {
            GraphvizGen gen { nStage->getInterface().getRaw(), ctx };
            gen.generate(path, name);
        },
        [&](FinalStage *) -> void {
            generateGraphvizAsFinal(path, name);
        }
    );
}

void Node::generateGraphvizAsFinal(const std::filesystem::path& path, const std::string& name) const {
    auto final = asFinalStage();
    GraphvizGen gen { final->value, sampler->getBindingContext() };
    gen.generate(path, name);
}

std::string Node::getNestedLoopsAsFinal() const {
    auto final = asFinalStage();
    return final->value.printNestedLoopsForAll(sampler->getBindingContext());
}

std::size_t Node::countChildren() const {
    return match<std::size_t>(
        [](AbstractStage *stage) { return stage->countChildren(); },
        [](FinalStage *stage) { return 0; }
    );
}

std::vector<Next> Node::getChildrenHandles() const {
    return match<std::vector<Next>>(
        [](AbstractStage *stage) { return stage->getChildrenHandles(); },
        [](FinalStage *stage) { return std::vector<Next>{}; }
    );
}

std::vector<Arc> Node::getChildrenArcs() const {
    return match<std::vector<Arc>>(
        [](AbstractStage *stage) { return stage->getChildrenArcs(); },
        [](FinalStage *stage) { return std::vector<Arc>{}; }
    );
}

std::optional<Arc> Node::getArcFromHandle(Next next) const {
    return match<std::optional<Arc>>(
        [&](AbstractStage *stage) { return stage->getArcFromHandle(next); },
        [](FinalStage *stage) -> std::optional<Arc> { return std::nullopt; }
    );
}

std::optional<Node> Node::getChild(Next next) const {
    return match<std::optional<Node>>(
        [&](AbstractStage *stage) { return stage->getChild(next); },
        [](FinalStage *stage) -> std::optional<Node> { return std::nullopt; }
    );
}

std::vector<std::optional<Node>> Node::getChildren(const std::vector<Next>& nexts) const {
    return match<std::vector<std::optional<Node>>>(
        [&](AbstractStage *stage) { return stage->getChildren(nexts); },
        [&](FinalStage *stage) {
            return std::vector<std::optional<Node>>(nexts.size());
        }
    );
}

bool Node::canAcceptArc(Arc arc) const {
    return match<bool>(
        [&](AbstractStage *stage) { return stage->canAcceptArc(arc); },
        [](FinalStage *stage) -> bool { return false; }
    );
}

Node Node::getChildFromArc(Arc arc) const {
    return match<Node>(
        [&](AbstractStage *stage) { return stage->getChild(arc); },
        [](FinalStage *stage) -> Node { KAS_UNREACHABLE(); }
    );
}

std::vector<Next> Node::getPossiblePath() const {
    std::optional<Next> finalNext;
    AbstractStage *stage = match<AbstractStage *>(
        [](AbstractStage *stage) {
            return stage;
        },
        [&](FinalStage *stage) {
            const NextFinalizeSlot& slot = stage->getSlot();
            finalNext = slot;
            return &stage->parent;
        }
    );
    const Graph graph = stage->getInterface().buildGraph();
    auto result = Sampler::ConvertOpsToNexts(Sampler::ConvertGraphToOps(graph));
    if (finalNext) {
        result.emplace_back(*finalNext);
    }
    return result;
}

std::vector<Arc> Node::getComposingArcs() const {
    std::optional<Arc> finalArc;
    AbstractStage *stage = match<AbstractStage *>(
        [](AbstractStage *stage) {
            return stage;
        },
        [&](FinalStage *stage) {
            const NextFinalizeSlot& slot = stage->getSlot();
            finalArc = { sampler, &slot.finalization };
            return &stage->parent;
        }
    );
    const Graph graph = stage->getInterface().buildGraph();
    auto result = sampler->convertOpsToArcs(Sampler::ConvertGraphToOps(graph));
    if (finalArc) {
        result.emplace_back(*finalArc);
    }
    return result;
}

void Node::expandSync(int layers) const {
    match<void>(
        [&](AbstractStage *stage) {
            stage->expandSync(layers);
        },
        [](FinalStage *) -> void {
            return;
        }
    );
}

void Node::expandWithArcs(ThreadPool<LatticeTask>& expander, const std::vector<Arc>& arcs) const {
    // For some reason we have to know whether the next stage is a deadend or not.
    // That is to say we have to expand our children!
    // So, as you wish :(.
    for (Next next: getChildrenHandles()) {
        auto child = getChild(next);
        if (child) {
            child->countChildren();
        }
    }
    // Continue.
    if (arcs.empty()) return;
    std::size_t success = 0;
    for (std::size_t index = 0; const Arc& arc: arcs) {
        if (!canAcceptArc(arc)) {
            ++index;
            continue;
        }
        auto child = getChildFromArc(arc);
        std::vector<Arc> remainingArcs = arcs;
        remainingArcs.erase(remainingArcs.begin() + index);
        expander.add(LatticeTask { child, std::move(remainingArcs) });
        ++success;
        ++index;
    }
    KAS_ASSERT(success > 0);
}

void Node::expandToSync(Node target) const {
    if (*this == target) {
        return;
    }
    std::vector<Arc> remainingReductions, remainingOthers;
    {
        auto bottomArcs = ranges::to<std::unordered_set<Arc, Arc::Hash>>(match<std::vector<Arc>>(
            [&](AbstractStage *) { return getComposingArcs(); },
            [](FinalStage *) -> std::vector<Arc> { KAS_UNREACHABLE(); }
        ));
        auto topArcs = ranges::to<std::unordered_set<Arc, Arc::Hash>>(target.match<std::vector<Arc>>(
            [&](AbstractStage *) { return target.getComposingArcs(); },
            // If the target is a TensorView, we only need to expand to its predecessor.
            [&](FinalStage *stage) -> std::vector<Arc> { return Node(sampler, &stage->parent).getComposingArcs(); }
        ));
        std::size_t removed = 0;
        for (const Arc& arc: topArcs) {
            if (bottomArcs.contains(arc)) {
                ++removed;
            } else {
                if (arc.tryAs<ReduceOp>()) {
                    remainingReductions.emplace_back(arc);
                } else {
                    remainingOthers.emplace_back(arc);
                }
            }
        }
        KAS_ASSERT(removed == bottomArcs.size());
    }
    auto& expander = sampler->getLatticeExpander();
    Node normalBottom = *this;
    if (!remainingReductions.empty()) {
        expander.add(LatticeTask { *this, remainingReductions });
        for (const Arc& arc: remainingReductions) {
            normalBottom = normalBottom.getChildFromArc(arc);
        }
    }
    if (!remainingOthers.empty()) {
        expander.add(LatticeTask { normalBottom, remainingOthers });
    }
    expander.sync();
}

void Node::expand(int layers) const {
    match<void>(
        [&](AbstractStage *stage) {
            stage->expand(layers);
        },
        [](FinalStage *) -> void {
            return;
        }
    );
}

std::optional<std::string> Node::getChildDescription(Next next) const {
    return match<std::optional<std::string>>(
        [&](AbstractStage *stage) -> std::optional<std::string> {
            auto arc = stage->getArcFromHandle(next);
            if (!arc) return std::nullopt;
            return arc->toString();
        },
        [](FinalStage *stage) -> std::string {
            KAS_UNREACHABLE();
        }
    );
}

bool Node::isDeadEnd() const {
    return match<bool>(
        [](AbstractStage *stage) { return stage->getFinalizability() == Finalizability::No; },
        [](FinalStage *stage) { return false; }
    );
}

bool Node::discoveredFinalDescendant() const {
    return match<bool>(
        [](AbstractStage *stage) { return stage->getFinalizability() == Finalizability::Yes; },
        [](FinalStage *stage) { return true; }
    );
}

std::string Node::toString() const {
    return match<std::string>(
        [](AbstractStage *stage) { return stage->description(); },
        [](FinalStage *stage) { return stage->description(); }
    );
}

std::string Node::debugToGraphviz() const {
    return match<std::string>(
        [&](AbstractStage *stage) { return GraphvizGen(stage->getInterface().getRaw(), sampler->getBindingContext()).print("preview"); },
        [&](FinalStage *stage) { return GraphvizGen(stage->value, sampler->getBindingContext()).print("preview"); }
    );
}

} // namespace kas
