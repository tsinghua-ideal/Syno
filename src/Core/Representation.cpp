#include <sstream>

#include "KAS/Core/Representation.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Representation::Representation(const BindingContext& ctx):
    ctx { ctx }
{}

void Representation::addTransform(std::string&& transform) {
    transforms.emplace_back(std::move(transform));
}

void Representation::addShape(const Shape& shape) {
    shapes.emplace_back(std::move(shape.toString(ctx)));
}

std::string Representation::description() const {
    KAS_ASSERT(transforms.size() + 1 == shapes.size());
    std::stringstream ss;
    for (std::size_t i = 0; i < transforms.size(); ++i) {
        ss << shapes[i] << '\n';
        ss << transforms[i] << '\n';
    }
    ss << shapes.back() << '\n';
    return ss.str();
}

} // namespace kas
