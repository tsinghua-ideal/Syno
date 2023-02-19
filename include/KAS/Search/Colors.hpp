#pragma once

#include "KAS/Core/DimensionDecl.hpp"


namespace kas {

class Colors {
    // The dimensions that have no color.
    std::vector<Dimension> clear;
    // The disjoint pairs of set of dimensions, in the sense that they have no common color.
    std::vector<std::pair<std::vector<Dimension>, std::vector<Dimension>>> constraints;
public:
    void substitute(std::initializer_list<Dimension> fro, std::initializer_list<Dimension> to);
    void clearify(std::initializer_list<Dimension> items);
};

} // namespace kas
