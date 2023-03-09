#include <algorithm>

#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Stride.hpp"


namespace kas {

std::size_t StrideOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, stride);
    return h;
}

IteratorValue StrideOp::value(const IteratorValue& output) const {
    auto stride = ConstValueNode::Create(this->stride);
    return stride * output;
}

std::size_t StrideOp::CountColorTrials = 0;
std::size_t StrideOp::CountColorSuccesses = 0;
bool StrideOp::transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    ++CountColorTrials;
    auto& out = interface[output];
    // Stride asserts that the IO dimensions are clear.
    if (out.isSingle()) { // So we must not violate existing constraints.
        return false;
    }
    if (output.size().isGeneral()) { // [Single Statement] We know that general dimension cannot be clear.
        return false;
    }
    // The `substitute` removes output, so actually no need to make it clear.
    colors.assign(interface, output, Colors::Clear);
    colors.substitute(interface, output, { getInput(), Colors::Clear });
    colors.simplify(interface);
    ++CountColorSuccesses;
    return true;
}

std::vector<const StrideOp *> StrideOp::Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors) {
    std::vector<const StrideOp *> result;
    for (std::size_t i = 0; i < outputShape.size(); ++i) {
        const Size& size = outputShape[i].size();
        auto primary = size.getPrimary();
        if (std::ranges::all_of(primary, [](auto x) { return x == 0; })) {
            // Here, we only allow an axis with primary variable to be strided. TODO: relax this?
            continue;
        }
        auto coefficient = size.getCoefficient();
        for (std::size_t j = 0; j < coefficient.size(); ++j) {
            // Here we take one of the coefficient as stride. If you want more, you can add more StrideShapeOp.
            auto stride = Size(primary.size(), coefficient.size());
            stride.getCoefficient()[j] = 1;
            result.emplace_back(store.get<StrideOp>(outputShape[i], stride));
        }
    }
    return result;
}

} // namespace kas
