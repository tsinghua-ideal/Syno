#include "KAS/Search/Node.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Stage.hpp"


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

TensorView *Node::asKernel() const {
    return std::get<TensorView *>(inner);
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
        [&](TensorView *tensor) -> Node { KAS_UNREACHABLE(); }
    );
}

} // namespace kas
