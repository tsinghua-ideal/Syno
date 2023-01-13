#pragma once

#include <string>
#include <vector>

#include "KAS/Core/BindingContext.hpp"


namespace kas {

class Representation {
public:
    using Transform = std::string;

protected:
    const BindingContext& ctx;
    // The description for transforms.
    std::vector<Transform> transforms;
    // The intermediate shapes. The first one is the input shape, the last one is the output shape, so the number of shapes is the number of transforms + 1.
    std::vector<std::string> shapes;

public:
    Representation(const BindingContext& ctx);
    void addTransform(Transform&& transform);
    void addShape(const Shape& shape);
    // Prints the transform process.
    std::string description() const;
};

} // namespace kas
