#pragma once

#include <boost/container_hash/hash.hpp>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class SplitOp final: public SplitLikePrimitiveOp {
    Size sz;
public:
    SplitOp(auto&& outputLhs, auto&& outputRhs):
        SplitLikePrimitiveOp { std::forward<decltype(outputLhs)>(outputLhs), std::forward<decltype(outputRhs)>(outputRhs) },
        sz { this->outputLhs.size() * this->outputRhs.size() }
    {}

    inline const Size& size() const noexcept override { return sz; }
    // SplitOp keeps no metadata.
    constexpr std::size_t initialHash() const noexcept override { return boost::hash<std::string>{}("Split"); }
    constexpr DimensionType type() const noexcept override { return DimensionType::Split; }

    IteratorValue value(const IteratorValue &outputMajor, const IteratorValue &outputMinor) const override;
    
    inline bool operator==(const SplitOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        std::size_t dimLowerBound;
    };
    static std::vector<Dimension> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
