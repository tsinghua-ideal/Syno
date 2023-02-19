#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/DimensionDecl.hpp"
#include "KAS/Search/Colors.hpp"


namespace kas {

struct Stage {
    // The interface decides the hash. Other properties are computed.
    std::vector<Dimension> interface;
    Colors colors;
    std::vector<std::reference_wrapper<const Size>> missingSizes;
};

} // namespace kas
