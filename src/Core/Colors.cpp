#include "KAS/Core/Colors.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

std::size_t Colors::CountColorInconsistent = 0;

void Colors::setInconsistent() {
    KAS_ASSERT(consistent, "Cannot set inconsistent state twice.");
    consistent = false;
    ++CountColorInconsistent;
}

void Colors::disjoint(const Dimension& lhs, const Dimension& rhs) {
    if (!isConsistent()) return;
    constraints.emplace_back(Interface { lhs }, Interface { rhs });
}

void Colors::substitute(ColoredInterface& interface, const Dimension& fro, ColoredDimension to) {
    if (!isConsistent()) return;
    for (auto& [lhs, rhs]: constraints) {
        WeakOrderedSubstituteVector1To1IfAny(lhs, fro, to.dimension, Dimension::HashLessThan{});
        WeakOrderedSubstituteVector1To1IfAny(rhs, fro, to.dimension, Dimension::HashLessThan{});
    }
    bool res = WeakOrderedSubstituteVector1To1IfAny(interface.items, fro, std::move(to), Dimension::HashLessThan{}, ColoredDimension::Projection{});
    KAS_ASSERT(res);
}

void Colors::substitute(ColoredInterface& interface, const Dimension& fro, ColoredDimension to1, ColoredDimension to2) {
    if (!isConsistent()) return;
    for (auto& [lhs, rhs]: constraints) {
        WeakOrderedSubstituteVector1To2IfAny(lhs, fro, to1.dimension, to2.dimension, Dimension::HashLessThan{});
        WeakOrderedSubstituteVector1To2IfAny(rhs, fro, to1.dimension, to2.dimension, Dimension::HashLessThan{});
    }
    bool res = WeakOrderedSubstituteVector1To2IfAny(interface.items, fro, std::move(to1), std::move(to2), Dimension::HashLessThan{}, ColoredDimension::Projection{});
    KAS_ASSERT(res);
}

void Colors::substitute(ColoredInterface& interface, const Dimension& fro1, const Dimension& fro2, ColoredDimension to) {
    if (!isConsistent()) return;
    for (auto& [lhs, rhs]: constraints) {
        WeakOrderedSubstituteVector2To1IfAny(lhs, fro1, fro2, to.dimension, Dimension::HashLessThan{});
        WeakOrderedSubstituteVector2To1IfAny(rhs, fro1, fro2, to.dimension, Dimension::HashLessThan{});
    }
    bool res = WeakOrderedSubstituteVector2To1IfAny(interface.items, fro1, fro2, std::move(to), Dimension::HashLessThan{}, ColoredDimension::Projection{});
    KAS_ASSERT(res);
}

ColoredDimension& Colors::assign(ColoredInterface& interface, const Dimension& item, int color) {
    auto& target =  interface[item];
    if (!isConsistent()) return target;
    if (!target.isUnknown() && target.color != color) {
        setInconsistent();
    } else {
        target.color = color;
    }
    return target;
}

void Colors::simplify(ColoredInterface& interface) {
    if (!isConsistent()) return;
    for (auto& [lhs, rhs]: constraints) {
        for (auto& item: lhs) {
            if (auto it = WeakOrderedBinarySearch(rhs, item, Dimension::HashLessThan{}); it != rhs.end()) { // The dimensio is disjoint with itself, so it has no color.
                assign(interface, item, Colors::Clear);
            }
        }
    }
    for (auto& [lhs, rhs]: constraints) {
        auto colorAcc = std::vector<int>(options.maximumTensors + 1, 0); // Clear is color.
        auto newLhs = std::vector<Dimension>();
        auto newRhs = std::vector<Dimension>();
        for (auto& item: lhs) {
            auto& target = interface[item];
            switch (target.category()) {
            case Category::Clear:
                break; // Remove from constraint.
            case Category::Single:
                ++colorAcc.at(target.color); // Record color, and keep in constraint.
            case Category::Unknown:
                newLhs.emplace_back(item); // Keep in constraint.
                break;
            }
        }
        for (auto& item: rhs) {
            auto& target = interface[item];
            switch (target.category()) {
            case Category::Clear:
                break; // Remove from constraint.
            case Category::Single:
                if (colorAcc.at(target.color) > 0) { // The color is already present in the lhs.
                    // This violates the constraint.
                    setInconsistent();
                    return;
                }
            case Category::Unknown:
                newRhs.emplace_back(item); // Keep in constraint.
                break;
            }
        }
        lhs = std::move(newLhs);
        rhs = std::move(newRhs);
    }
    std::erase_if(constraints, [](const auto& pair) {
        const auto& [lhs, rhs] = pair;
        return lhs.empty() || rhs.empty();
    });
}

bool Colors::checkFinalization(const std::vector<Interface>& tensors) const {
    // TODO!!!
    return true;
}

} // namespace kas
