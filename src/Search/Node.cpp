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
            return { Next::TypeOf(op->getType()), op->opHash() };
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
        [&](std::shared_ptr<TensorView> tensor) {
            return *tensor == *std::get<std::shared_ptr<TensorView>>(rhs.inner);
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

AbstractStage *Node::tryAsStage() const {
    return match<AbstractStage *>(
        [](AbstractStage *stage) {
            return stage;
        },
        [](std::shared_ptr<TensorView> tensor) -> AbstractStage * {
            return nullptr;
        }
    );
}

NormalStage *Node::asNormalStage() const {
    return std::get<NormalStage *>(inner);
}

std::shared_ptr<TensorView> Node::asFinal() const {
    return std::get<std::shared_ptr<TensorView>>(inner);
}

std::unique_ptr<Kernel> Node::realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, CodeGenOptions options, const std::filesystem::path& directory, const std::string& name) const {
    auto final = asFinal();
    if (!final) {
        return nullptr;
    }
    return std::make_unique<Kernel>(sampler->getBindingContext(), *final, allMappings, std::move(options), directory, name);
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

void Node::generateGraphviz(const std::filesystem::path& path, const std::string& name) const {
    const auto& ctx = sampler->getBindingContext();
    match<void>(
        [&](ReductionStage *rStage) {
            GraphvizGen gen { rStage->getInterface(), ctx };
            gen.generate(path, name);
        },
        [&](NormalStage *nStage) {
            GraphvizGen gen { nStage->getInterface(), ctx };
            gen.generate(path, name);
        },
        [&](std::shared_ptr<TensorView>) -> void {
            generateGraphvizAsFinal(path, name);
        }
    );
}

void Node::generateGraphvizAsFinal(const std::filesystem::path& path, const std::string& name) const {
    auto final = asFinal();
    GraphvizGen gen { *final, sampler->getBindingContext() };
    gen.generate(path, name);
}

std::string Node::getNestedLoopsAsFinal() const {
    auto final = asFinal();
    return final->printNestedLoopsForAll(sampler->getBindingContext());
}

std::size_t Node::countChildren() const {
    return match<std::size_t>(
        [](AbstractStage *stage) { return stage->countChildren(); },
        [](std::shared_ptr<TensorView> tensor) { return 0; }
    );
}

std::vector<Next> Node::getChildrenHandles() const {
    return match<std::vector<Next>>(
        [](AbstractStage *stage) { return stage->getChildrenHandles(); },
        [](std::shared_ptr<TensorView> tensor) { return std::vector<Next>{}; }
    );
}

std::vector<Arc> Node::getChildrenArcs() const {
    return match<std::vector<Arc>>(
        [](AbstractStage *stage) { return stage->getChildrenArcs(); },
        [](std::shared_ptr<TensorView> tensor) { return std::vector<Arc>{}; }
    );
}

std::optional<Arc> Node::getArcFromHandle(Next next) const {
    return match<std::optional<Arc>>(
        [&](AbstractStage *stage) { return stage->getArcFromHandle(next); },
        [](std::shared_ptr<TensorView> tensor) -> std::optional<Arc> { return std::nullopt; }
    );
}

std::optional<Node> Node::getChild(Next next) const {
    return match<std::optional<Node>>(
        [&](AbstractStage *stage) { return stage->getChild(next); },
        [](std::shared_ptr<TensorView> tensor) -> std::optional<Node> { return std::nullopt; }
    );
}

bool Node::canAcceptArc(Arc arc) const {
    return match<bool>(
        [&](AbstractStage *stage) { return stage->canAcceptArc(arc); },
        [](std::shared_ptr<TensorView> tensor) -> bool { return false; }
    );
}

Node Node::getChildFromArc(Arc arc) const {
    return match<Node>(
        [&](AbstractStage *stage) { return stage->getChild(arc); },
        [](std::shared_ptr<TensorView> tensor) -> Node { KAS_UNREACHABLE(); }
    );
}

std::vector<Next> Node::getPossiblePath() const {
    return match<std::vector<Next>>(
        [](AbstractStage *stage) {
            auto graph = stage->getInterface().buildGraph();
            return Sampler::ConvertGraphToPath(graph);
        },
        [&](std::shared_ptr<TensorView> tensorView) {
            auto tensors = ranges::to<std::vector<std::vector<Dimension>>>(tensorView->getUnderlyingTensorRange());
            return sampler->convertTensorsToPath(tensors);
        }
    );
}

std::vector<Arc> Node::getComposingArcs() const {
    auto possiblePath = getPossiblePath();
    auto arcs = sampler->convertPathToArcs(possiblePath);
    KAS_ASSERT(arcs, "This node is a dead end, so the composing arcs do not exist.");
    return std::move(*arcs);
}

void Node::expandSync(int layers) const {
    match<void>(
        [&](AbstractStage *stage) {
            stage->expandSync(layers);
        },
        [](std::shared_ptr<TensorView>) -> void {
            return;
        }
    );
}

void Node::expand(int layers) const {
    match<void>(
        [&](AbstractStage *stage) {
            stage->expand(layers);
        },
        [](std::shared_ptr<TensorView>) -> void {
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
        [](std::shared_ptr<TensorView> tensor) -> std::string {
            KAS_UNREACHABLE();
        }
    );
}

bool Node::isDeadEnd() const {
    return match<bool>(
        [](AbstractStage *stage) { return stage->getFinalizability() == Finalizability::No; },
        [](std::shared_ptr<TensorView> tensor) { return false; }
    );
}

bool Node::discoveredFinalDescendant() const {
    return match<bool>(
        [](AbstractStage *stage) { return stage->getFinalizability() == Finalizability::Yes; },
        [](std::shared_ptr<TensorView> tensor) { return true; }
    );
}

std::string Node::toString() const {
    const BindingContext& ctx = sampler->getBindingContext();
    return match<std::string>(
        [](AbstractStage *stage) { return stage->description(); },
        [&](std::shared_ptr<TensorView> tensor) { return tensor->description(ctx); }
    );
}

} // namespace kas
