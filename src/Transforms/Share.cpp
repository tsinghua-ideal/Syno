#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::pair<IteratorValue, IteratorValue> ShareOp::value(const IteratorValue& output) const {
    return { output, output };
}

bool ShareOp::transformColors(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    // [Single Statement] Only dimensions of sizes with no primary variables can be of clear color.
    auto out = interface.binarySearch(output);
    auto [inputLhs, inputRhs] = getInputs();
    if (output.size().isGeneral()) { // Test if there are any primary variables.
        if (!out->isUnknown()) return false; // [Single Statement] This is a dimension of two colors, and must be Unknown.
        if (options.maximumTensors <= 1) {
            return false; // [Single Statement] There must be at least 2 colors.
        } else if (options.maximumTensors == 2) { // Here we can just assign the colors.
            colors.substitute(interface, output, { inputLhs, Colors::First }, { inputRhs, Colors::Second });
        } else { // We cannot be very sure about this, since there can be many colors.
            colors.substitute(interface, output, { inputLhs, Colors::Unknown }, { inputRhs, Colors::Unknown });
            colors.disjoint(inputLhs, inputRhs);
        }
    } else {
        switch (out->category()) {
        case Colors::Category::Clear: // Pass this clear color over.
            colors.substitute(interface, output, { inputLhs, Colors::Clear }, { inputRhs, Colors::Clear });
            break;
        case Colors::Category::Single:
            fmt::print(stderr, "[Single Statement] No primary variables, so we should have made no assumptions.");
            return false;
        case Colors::Category::Unknown: // Now out->isUnknown(). We do not know the color, so add it to disjoint constraints.
            colors.substitute(interface, output, { inputLhs, Colors::Unknown }, { inputRhs, Colors::Unknown });
            colors.disjoint(inputLhs, inputRhs);
            break;
        }
    }
    colors.simplify(interface);
    return true;
}

std::vector<const ShareOp *> ShareOp::Generate(DimensionStore& store, const ColoredInterface& interface, const Colors& colors, GenerateOptions options) {
    Allowance allowance { Size::Product(interface.getShape()), options.ctx };
    std::vector<const ShareOp *> result;
    if (interface.size() < options.dimUpperBound) {
        for (auto&& dim: interface.items | std::views::transform(ColoredDimension::Projection{})) {
            if (allowance.withinAllowance(dim.size())) {
                result.emplace_back(store.get<ShareOp>(dim));
            }
        }
    }
    return result;
}

} // namespace kas
