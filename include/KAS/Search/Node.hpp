#pragma once

#include "KAS/Search/Stage.hpp"


namespace kas {

class Node {
    Stage *data;
    // We are searching bottom-up, so the children are actually closer to the input.
    std::vector<std::unique_ptr<Node>> nexts;
public:
    inline Node(Stage *data): data { data } {}

};

}