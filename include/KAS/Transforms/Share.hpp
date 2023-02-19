#pragma once

#include <functional>
#include <memory>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/DimensionDecl.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareOp {
public:
    Dimension output;
    FirstOrSecond firstOrSecond;
    inline ShareOp(auto&& output, FirstOrSecond firstOrSecond):
        output { std::forward<decltype(output)>(output) },
        firstOrSecond { firstOrSecond }
    {}
    bool operator==(const ShareOp& other) const = default;
    inline const Size& size() const noexcept { return output.size(); }
    DoubleIteratorValue value(const IteratorValue& output) const;
    // Since ShareOp keeps no metadata, the initial hash is the same for all ShareOps.
    consteval std::size_t initialHash() const noexcept { return std::hash<std::string>{}(Type()); }
    consteval static const char *Type() { return "Share"; }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<std::pair<Dimension, Dimension>> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};
static_assert(MergeLikePrimitiveOp<ShareOp>);

} // namespace kas
