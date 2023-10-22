#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Shift.hpp"


namespace kas {

ShiftOp::ShiftOp(const Dimension& output, int shift):
    RepeatLikeOp { output },
    shift { shift },
    input { this }
{}

std::size_t ShiftOp::initialHash() const noexcept {
    std::size_t h = DimensionTypeHash(Type);
    HashCombine(h, shift);
    return h;
}

ShiftOp::Values ShiftOp::value(const Values& known) const {
    if (known.canSkipDeduction()) return known;
    auto& [input, output] = known.values;
    auto imm = ImmediateValueNode::Create(shift);
    auto size = ConstValueNode::Create(this->output.size());
    if (auto outputV = output.tryValue(); outputV) {
        // Out value -> in value.
        if (input.isUnorientedOrOrientedUp()) { // Check.
            return {{ (outputV + imm) % size, outputV }};
        }
    } else if (auto inputV = input.tryValue(); inputV) {
        // In value -> out value.
        if (output.isUnorientedOrOrientedDown()) { // Check.
            return {{ inputV, (inputV - imm) % size }};
        }
    } else if (output.isOrientedUp()) {
        // Out orientation -> in orientation.
        if (input.isUnorientedOrOrientedUp()) {
            return {{ Direction::Up, Direction::Up }};
        }
    } else if (input.isOrientedDown()) {
        // In orientation -> out orientation.
        if (output.isUnorientedOrOrientedDown()) {
            return {{ Direction::Down, Direction::Down }};
        }
    }
    // Otherwise, conflict.
    KAS_CRITICAL("Conflicting values for ShiftOp: input = {}, output = {}", input, output);
}

bool ShiftOp::ExceedsMaxValidReshapeShiftPattern(const Size& block, int shift, const BindingContext& ctx, float maximumValidReshapeShiftPattern) {
    return boost::rational_cast<float>(block.lowerBoundEst(ctx)) / std::abs(shift) > maximumValidReshapeShiftPattern;
}

std::vector<const ShiftOp *> ShiftOp::Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallows { Reduce, ShareR, Shift };
    if (options.disallowShiftAboveUnfold) disallows.push_back(Unfold);
    auto plausible = interface.filterOut(disallows);

    std::vector<const ShiftOp *> result;
    CountGenerateAttempts += interface.getDimensions().size();
    std::size_t countPlausible = 0;
    constexpr int ShiftValue = 1;
    for (auto&& dim: plausible) {
        ++countPlausible;
        Dimension peek = dim;
        if (auto share = dim.tryAs<ShareOp::Input>(); share) {
            // Canonicalization requires us to go beyond Share.
            peek = share->getOp()->output;
            if (std::ranges::any_of(disallows, [&](auto disallow) { return peek.is(disallow); })) {
                // TODO: we are duplicating filterOut. Make peek more formalized.
                continue;
            }
        }
        if (auto split = peek.tryAs<SplitOp::Input>(); split) {
            // This is a reshape and shift pattern.
            // We would like to see if the reshape is worth this Shift.
            if (ExceedsMaxValidReshapeShiftPattern(
                split->getDerivedOp<SplitOp>()->getBlock(),
                ShiftValue,
                options.ctx,
                options.maximumValidReshapeShiftPattern
            )) {
                // It seems that the reshape RHS is too large, and Shift barely makes a difference compared to being placed underneath this reshape.
                ++CountExceedsMaxValidReshapeShiftPattern;
                continue;
            }
        }
        ++CountSuccessfulGenerations;
        result.emplace_back(store.get<ShiftOp>(dim, ShiftValue));
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return result;
}

} // namespace kas
