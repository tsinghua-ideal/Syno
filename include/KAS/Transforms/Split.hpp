#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class SplitOp final: public SplitLikeOp {
public:
    class Input final: public SplitLikeOp::Input {
    public:
        inline Input(const SplitOp* op):
            SplitLikeOp::Input { op }
        {}
        inline const Size& size() const noexcept override { return getDerivedOp<SplitOp>()->sz; }
        std::size_t hash() const noexcept override;
        constexpr DimensionType type() const noexcept override { return DimensionType::Split; }
    };

protected:
    Size sz;
    Input input;

public:
    SplitOp(auto&& outputLhs, auto&& outputRhs):
        SplitLikeOp { std::forward<decltype(outputLhs)>(outputLhs), std::forward<decltype(outputRhs)>(outputRhs) },
        sz { this->outputLhs.size() * this->outputRhs.size() },
        input { this }
    {}
    inline Dimension getInput() const override { return &input; }
    IteratorValue value(const IteratorValue &outputMajor, const IteratorValue &outputMinor) const override;
    
    inline bool operator==(const SplitOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        std::size_t dimLowerBound;
    };
    static std::vector<std::unique_ptr<SplitOp>> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
