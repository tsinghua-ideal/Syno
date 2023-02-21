#pragma once

#include "KAS/Search/Stage.hpp"
#include <map>


namespace kas {

class RootNode {
    
};

class Node {
    Stage *data;
    // We are searching bottom-up, so the children are actually closer to the input.
    std::optional<std::map<Dimension, std::unique_ptr<Node>>> nexts;
public:
    inline Node(Stage *data): data { data } {}

};

}