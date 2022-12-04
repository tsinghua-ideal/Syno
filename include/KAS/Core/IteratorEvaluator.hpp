#pragma once

#include <map>
#include <memory>
#include <queue>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Shape.hpp"


namespace kas {

using IteratorValueMap = std::map<std::shared_ptr<Iterator>, std::shared_ptr<IteratorValue>>;

class IteratorEvaluator {
public:
    BindingContext& bindingContext;
    IteratorValueMap valueMap;
    std::queue<std::shared_ptr<Iterator>> workingSet;
    IteratorEvaluator(BindingContext& bindingContext);

    void evaluateTensorAccess(const TensorView& tensor);
};

} // namespace kas
