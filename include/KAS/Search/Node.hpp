#pragma once

#include <cstddef>
#include <map>

#include "KAS/Search/Stage.hpp"


namespace kas {

class StageStore;

class Node {
    Stage *stage;

public:
    inline Node(Stage *stage): stage { stage } {}
    inline std::size_t size() const { return stage->size(); }
    inline bool isFinal(std::size_t index) const { return stage->isFinal(index); }
    inline std::variant<Stage *, TensorView *> next(std::size_t index) const { return stage->next(index); }
    inline std::string opDescription(std::size_t index) const { return stage->opDescription(index); }
};

}