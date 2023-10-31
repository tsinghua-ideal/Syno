#pragma once

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

struct DesiredSize {
    Size value;
    bool isUnordered;
    operator const Size&() const { return value; }
};

struct CurrentSize {
    Size value;
    int remainingLength;
    operator const Size&() const { return value; }
};

struct CurrentDimension {
    Dimension value;
    int remainingLength;
    operator CurrentSize() const {
        return { value.size(), remainingLength }; 
    }
    operator const Dimension&() const { return value; }
};

} // namespace kas
