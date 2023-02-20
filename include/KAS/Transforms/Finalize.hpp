#pragma once

#include <vector>

#include "KAS/Core/Dimension.hpp"


namespace kas {

// Contains multiple finalization options.
class FinalizeOp {
public:
    using Tensors = std::vector<Interface>;
    std::vector<Tensors> options;
};

} // namespace kas
