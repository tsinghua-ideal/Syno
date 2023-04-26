#include <algorithm>

#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Stride.hpp"


namespace kas {

std::size_t StrideOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, stride);
    return h;
}

StrideOp::Values StrideOp::value(const Values& known) const {
    // This is different from other Op's. Because we need to set the initial orientation if allUnoriented.
    if (known.allValued()) return known;
    if (known.allUnoriented()) {
        return {{ Direction::Up, std::monostate{} }};
    }
    auto& [input, output] = known.values;
    auto stride = ConstValueNode::Create(this->stride);
    if (input.isOrientedUp()) { // Note that we must have set this to Up. This is sanity check. If input is valued, output must have been set beforehand.
        if (auto outputV = output.tryValue(); outputV) {
            // Out value -> in value.
            return {{ outputV * stride, outputV }};
        } else if (output.isUnorientedOrOrientedUp()) {
            // OK, but nothing to do.
            return {{ Direction::Up, output }};
        }
    }
    KAS_CRITICAL("Conflicting values for StrideOp: input = {}, output = {}", input, output);
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
    // colors.assign(interface, output, Colors::Clear);
    // colors.substitute(interface, output, { getInput(), Colors::Clear });
    colors.substitute(interface, output, { getInput(), Colors::Unknown });
    colors.simplify(interface);
    ++CountColorSuccesses;
    return true;
}

std::vector<const StrideOp *> StrideOp::Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options) {
    const BindingContext& ctx = options.ctx;
    std::vector<const StrideOp *> result;
    auto checkThenAdd = [&ctx, &result, &store](const Dimension& dim, auto&& block) {
        if ((dim.size() / block).isRealistic(ctx)) { // block is already realistic.
            result.emplace_back(store.get<StrideOp>(dim, std::forward<decltype(block)>(block)));
        }
    };
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
            checkThenAdd(outputShape[i], stride);
        }
    }
    return result;
}

} // namespace kas
