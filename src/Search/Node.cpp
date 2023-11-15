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
        [&](const Operation *op) {
            // We have uniquify them.
            return *op == *rhs.as<Operation>();
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
    constexpr std::size_t NumVariants = std::variant_size_v<decltype(inner)>;
    constexpr std::size_t NumBits = std::numeric_limits<std::size_t>::digits / NumVariants;
    HashCombine(h, 1_uz << (NumBits * inner.index()));
    auto contentHash = match<std::size_t>(
        [](const Operation *op) {
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
        [&](const Operation *op) -> Next {
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

Node::Node(Sampler *sampler, ReductionStage *rStage):
    sampler { sampler }, inner { rStage } {}
Node::Node(Sampler *sampler, NormalStage *nStage):
    sampler { sampler }, inner { nStage }
{
    KAS_ASSERT(!nStage->isEmbeddedInReductionStage(), "You should not use the embedded stage in Node!");
}
Node::Node(Sampler *sampler, FinalStage *fStage):
    sampler { sampler }, inner { fStage } {}

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

AbstractStage *Node::asNonFinalStage() const {
    return match<AbstractStage *>(
        [](AbstractStage *stage) { return stage; },
        [](FinalStage *) -> AbstractStage * { KAS_UNREACHABLE(); }
    );
}

FinalStage *Node::asFinalStage() const {
    return std::get<FinalStage *>(inner);
}

std::unique_ptr<Kernel> Node::realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, CodeGenOptions options, const std::filesystem::path& directory, const std::string& name) const {
    auto final = asFinalStage();
    return std::make_unique<Kernel>(sampler->getBindingContext(), final->value, final->pyTorchSpecializedIR, allMappings, std::move(options), directory, name);
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

Node Node::arbitraryParent() const {
    return match<Node>(
        [&](AbstractStage *stage) {
            auto parent = stage->arbitraryParent();
            if (auto rStage = dynamic_cast<ReductionStage *>(parent)) {
                return Node(sampler, rStage);
            } else if (auto nStage = dynamic_cast<NormalStage *>(parent)) {
                if (nStage->isEmbeddedInReductionStage()) {
                    return Node(sampler, dynamic_cast<ReductionStage *>(nStage->arbitraryParent()));
                } else {
                    return Node(sampler, nStage);
                }
            } else {
                KAS_UNREACHABLE();
            }
        },
        [&](FinalStage *stage) {
            return Node(sampler, &stage->parent);
        }
    );
}

GraphHandle Node::getInterface() const {
    return match<GraphHandle>(
        [&](AbstractStage *stage) { return stage->getInterface(); },
        [&](FinalStage *stage) { return stage->parent.getInterface(); }
    );
}

void Node::recomputeShapeDistance() const {
    match<void>(
        [&](AbstractStage *stage) { stage->recomputeShapeDistance(); },
        [](FinalStage *) {}
    );
}

ShapeDistance Node::getShapeDistance() const {
    return match<ShapeDistance>(
        [&](AbstractStage *stage) { return stage->getShapeDistance(); },
        [&](FinalStage *stage) -> ShapeDistance {
            return { 0, stage->value.getFLOPs(sampler->getBindingContext()) };
        }
    );
}

std::size_t Node::depth() const {
    return match<std::size_t>(
        [](AbstractStage *stage) { return stage->getDepth(); },
        [](FinalStage *stage) { return stage->parent.getDepth() + 1; }
    );
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

std::optional<Node> Node::getChildFromArc(Arc arc) const {
    return match<std::optional<Node>>(
        [&](AbstractStage *stage) { return stage->getChild(arc); },
        [](FinalStage *stage) -> std::optional<Node> { KAS_UNREACHABLE(); }
    );
}

std::vector<std::optional<Node>> Node::getChildrenFromArcs(const std::vector<Arc>& arcs) const {
    return match<std::vector<std::optional<Node>>>(
        [&](AbstractStage *stage) { return stage->getChildren(arcs); },
        [&](FinalStage *stage) -> std::vector<std::optional<Node>> { KAS_UNREACHABLE(); }
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
    auto result = Sampler::ConvertOpsToNexts(Sampler::ConvertGraphToOps(graph, sampler->getOpStore()));
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
    auto result = sampler->convertOpsToArcs(Sampler::ConvertGraphToOps(graph, sampler->getOpStore()));
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

void Node::expandWithArcs(ThreadPool<LatticeTask>& expander, const LatticeTask& task) const {
    const auto& arcs = task.arcs;
    expand(2);
    // Continue.
    if (arcs.empty()) return;
    [[maybe_unused]] std::size_t success = 0;
    for (std::size_t index = 0; const Arc& arc: arcs) {
        if (!canAcceptArc(arc)) {
            ++index;
            continue;
        }
        // This must not be null! Otherwise the pruning algorithm is wrong. (But it might throw for non-final lattices.)
        auto optChild = getChildFromArc(arc);
        if (!optChild) {
            // This is really wrong!
            ++CountCanAcceptArcButPruned;
            ++index;
            continue;
        }
        auto child = *optChild;
        std::vector<Arc> remainingArcs = arcs;
        remainingArcs.erase(remainingArcs.begin() + index);
        if (task.pool.add(remainingArcs.size(), child)) {
            expander.add(LatticeTask { task.pool, child, std::move(remainingArcs) });
        }
        ++success;
        ++index;
    }
    // Due to maxEnumerations, we cannot guarantee that this is a lattice.
    // KAS_ASSERT(success > 0);
}

std::vector<Node> Node::expandToSync(Node target) const {
    if (target.isFinal()) target = target.arbitraryParent();

    // We have partitioned the huge lattice into several smaller lattices.
    // We need to find the bottom node of each lattice.
    const Graph targetGraph = target.getInterface().buildGraph();
    auto& store = sampler->getOpStore();

    // TODO: we simply expand from root. We really should only expand from this node.
    const auto latticesOps = Sampler::ConvertGraphToOpLayers(targetGraph, store);
    KAS_ASSERT(ranges::fold_left(latticesOps | std::views::transform([](const auto& l) { return l.size(); }), 0_uz, std::plus<>{}) == target.depth());

    std::vector<std::vector<Arc>> latticesArcs;
    std::vector<LatticePool> latticePools;
    for (const auto& ops: latticesOps) {
        latticesArcs.emplace_back(sampler->convertOpsToArcs(ops));
        latticePools.emplace_back(ops.size());
    }
    auto& expander = sampler->getLatticeExpander();

    Node bottom = sampler->visit({}).value(); // root.
    std::vector<Node> latticeEndPoints { bottom };
    for (std::size_t i = 0; i < latticesArcs.size(); ++i) {
        auto arcs = latticesArcs[i];
        KAS_ASSERT(!arcs.empty());
        auto& pool = latticePools[i];

        bool win = true;
        bool hasLost = false;
        Node top = bottom;
        int rounds = 0;
        while (true) {
            for (const Arc& arc: arcs) {
                auto child = top.getChildFromArc(arc);
                if (!child) {
                    win = false;
                    break;
                }
                top = child.value();
            }
            if (win) {
                break;
            } else {
                top = bottom;
                auto [_, nextPermutation] = std::ranges::next_permutation(arcs);
                rounds += !nextPermutation;
                KAS_ASSERT(rounds < 2, "Node not reachable! Something really wrong happened!");
                win = true;
                hasLost = true;
            }
        }

        if (hasLost) {
            KAS_WARNING("Non-lattice found!");
            ++CountNonLattices;
        } else {
            ++CountLattices;
        }

        expander.add(LatticeTask { pool, bottom, std::move(arcs) });

        // const auto& arcs = latticesArcs[i];
        // auto& pool = latticePools[i];
        // Node top = ranges::fold_left(arcs, bottom, [&](Node node, const Arc& arc) {
        //     return node.getChildFromArc(arc).value();
        // });
        // expander.add(LatticeTask { pool, bottom, arcs });

        bottom = top;
        latticeEndPoints.emplace_back(bottom);
    }
    KAS_ASSERT(bottom == target);

    expander.sync();
    sampler->getExpander().sync();

    return latticeEndPoints;
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

LatticePool::LatticePool(std::size_t depth):
    depth { depth }, nodesPools(depth) {}
bool LatticePool::add(std::size_t remainingArcs, Node node) {
    auto& [mutex, pool] = nodesPools[remainingArcs];
    std::scoped_lock lock { mutex };
    return pool.insert(node).second;
}

} // namespace kas
