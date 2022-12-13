#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

IteratorEvaluator::IteratorEvaluator(BindingContext& bindingContext):
    bindingContext { bindingContext }
{}

void IteratorEvaluator::evaluateTensorAccess(TensorView& tensor) {
    auto interfaceIterators = tensor.getInterfaceIterators();
    for (std::size_t i = 0; i < interfaceIterators.size(); ++i) {
        auto ii = interfaceIterators[i];
        // Use original access
        valueMap.emplace(ii, tensor.getAccess(i));
        workingSet.push(ii);
    }
    auto reducedIterators = tensor.getReducedIterators();
    std::string base = "ri_";
    for (int i = 0; i < reducedIterators.size(); i++) {
        auto ri = reducedIterators[i];
        // Create new access
        auto ra = std::make_shared<VariableValueNode>(bindingContext.addIteratorVariable(base + std::to_string(i)));
        tensor.setReducedAccess(ra, i);
        valueMap.emplace(ri, ra);
        workingSet.push(ri);
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
