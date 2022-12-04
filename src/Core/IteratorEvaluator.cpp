#include <sstream>
#include <string>

#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

IteratorEvaluator::IteratorEvaluator(BindingContext& bindingContext):
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
    while (!workingSet.empty()) {
        auto iterator = workingSet.front();
        workingSet.pop();
        iterator->compute(*this);
    }
    valueMap.clear();
}

} // namespace kas
