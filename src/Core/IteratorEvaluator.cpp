#include <sstream>
#include <string>

#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

IteratorEvaluator::IteratorEvaluator(const BindingContext& bindingContext):
    bindingContext { bindingContext }
{}

void IteratorEvaluator::evaluateTensorAccess(const TensorView& tensor) {
    auto iterators = tensor.getAllIterators();
    for (int i = 0; i < iterators.size(); i++) {
        std::stringstream ss;
        ss << "i_" << i;
        auto iterator = iterators[i];
        valueMap.emplace(iterator, std::make_shared<IteratorValue>(ss.str()));
        workingSet.push(iterator);
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
