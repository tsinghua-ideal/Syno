#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

void IteratorEvaluator::evaluateTensorAccess(TensorView& tensor) {
    const auto& interfaceIterators = tensor.getInterfaceIterators();
    for (std::size_t i = 0; i < interfaceIterators.size(); ++i) {
        auto ii = interfaceIterators[i];
        // Use original access
        valueMap.emplace(ii, tensor.getAccess(i));
        workingSet.push(ii);
    }
    auto manipulations = tensor.getManipulations();
    for (std::size_t i = 0; i < manipulations.size(); i++) {
        const auto& m = manipulations[i];
        // Create new access
        auto ra = std::make_shared<VariableValueNode>(m.iteratorVariableId);
        valueMap.emplace(m.getIterator(), ra);
        workingSet.push(m.getIterator());
    }
    // This can be further optimized if we maintain an ``available'' variable for each node when we traverse the graph, so that there would be no failed ``compute()''. TODO
    while (!workingSet.empty()) {
        auto iterator = workingSet.front();
        workingSet.pop();
        iterator->compute(*this);
    }
    valueMap.clear();
}

} // namespace kas
