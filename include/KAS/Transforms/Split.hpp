#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class SplitOp final: public SplitLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Split;
    class Input final: public SplitLikeOp::Input {
    public:
        inline Input(const SplitOp* op):
            SplitLikeOp::Input { op }
        {}
        inline const Size& size() const noexcept override { return getDerivedOp<SplitOp>()->sz; }
        constexpr DimensionType type() const noexcept override { return Type; }
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
    constexpr DimensionType getType() const noexcept override { return Type; }
    constexpr std::size_t initialHash() const noexcept override { return static_cast<std::size_t>(Type); }
    inline Dimension getInput() const override { return &input; }
    IteratorValue value(const IteratorValue &outputMajor, const IteratorValue &outputMinor) const override;

    static std::size_t CountColorTrials;
    static std::size_t CountColorSuccesses;
    bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const override;
    
    inline bool operator==(const SplitOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        std::size_t dimLowerBound;
    };
    static std::vector<const SplitOp *> Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options);
};

} // namespace kas
