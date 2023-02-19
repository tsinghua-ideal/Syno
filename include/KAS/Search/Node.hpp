#pragma once

#include "KAS/Search/Stage.hpp"


namespace kas {

class Node {
    friend class Sampler;
    using PointerType = Stage *;

    PointerType data;
    // We are searching bottom-up, so the children are actually closer to the input.
    std::vector<std::unique_ptr<Node>> nexts;
public:
    inline Node(PointerType data): data { data } {}
};

}