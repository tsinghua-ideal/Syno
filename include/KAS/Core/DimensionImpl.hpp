#pragma once

#include <variant>

#include "KAS/Core/DimensionDecl.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

using DimensionAlternatives = std::variant<ShareOp, Iterator>;
template<typename T>
concept DimensionAlternative =
    std::same_as<T, std::remove_cvref_t<T>> &&
    requires(DimensionAlternatives x) {
        { std::get<T>(x) } -> std::same_as<T&>;
    };

class DimensionImpl {
public:
    DimensionAlternatives alts;
    const Size& size() const noexcept;
    template<PrimitiveOp T>
    const T& as() const {
        static_assert(DimensionAlternative<T>); // ensure the type is registered
        return std::get<T>(alts);
    }
};

template<DimensionLike T>
bool Dimension::is() const {
    static_assert(DimensionAlternative<T>); // ensure the type is registered
    return std::holds_alternative<T>(inner->alts);
}

template<PrimitiveOp T>
const T& Dimension::as() const {
    return inner->as<T>();
}

} // namespace kas
