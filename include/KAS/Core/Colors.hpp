#pragma once

#include <initializer_list>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class Colors {
    enum Color {
        Clear = 0,
        First = 1,
        Second = 2,
    };
    struct ColoredDimension {
        Dimension dimension;
        // 0 means clear, 1 means the first tensor, 2 means the second tensor, etc.
        // Here we only want to support two tensors at most.
        int color;
    };
    // The dimensions whose colors we have determined.
    std::vector<ColoredDimension> colored;
    // The disjoint pairs of set of dimensions, in the sense that they have no common color.
    std::vector<std::pair<std::vector<Dimension>, std::vector<Dimension>>> constraints;
public:
    void substitute(const Dimension& fro, std::initializer_list<Dimension> to);
    void assign(const Dimension& item, int color);
    void simplify();
};

} // namespace kas
