#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class UnfoldOp final: public SplitLikeOp {
public:
    class Input final: public SplitLikeOp::Input {
    public:
        inline Input(const UnfoldOp* op):
            SplitLikeOp::Input { op }
        {}
        inline const Size& size() const noexcept override { return getDerivedOp<UnfoldOp>()->outputLhs.size(); }
        std::size_t hash() const noexcept override;
        constexpr DimensionType type() const noexcept override { return DimensionType::Unfold; }
    };

protected:
    Input input;

public:
    UnfoldOp(auto&& outputLhs, auto&& outputRhs):
        SplitLikeOp { std::forward<decltype(outputLhs)>(outputLhs), std::forward<decltype(outputRhs)>(outputRhs) },
        input { this }
    {}
    inline Dimension getInput() const override { return &input; }
    IteratorValue value(const IteratorValue &outputMajor, const IteratorValue &outputMinor) const override;

    inline bool operator==(const UnfoldOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimLowerBound;
    };
    static std::vector<std::unique_ptr<UnfoldOp>> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
