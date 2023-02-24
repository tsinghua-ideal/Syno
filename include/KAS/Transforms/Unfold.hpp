#pragma once

#include <memory>
#include <optional>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class UnfoldOp final: public SplitLikePrimitiveOp {
public:
    using SplitLikePrimitiveOp::SplitLikePrimitiveOp;

    inline const Size& size() const noexcept override { return outputLhs.size(); }
    constexpr std::size_t initialHash() const noexcept override { return static_cast<std::size_t>(type()); }
    constexpr DimensionType type() const noexcept override { return DimensionType::Unfold; }

    IteratorValue value(const IteratorValue &outputMajor, const IteratorValue &outputMinor) const override;

    inline bool operator==(const UnfoldOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimLowerBound;
    };
    static std::vector<Dimension> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
