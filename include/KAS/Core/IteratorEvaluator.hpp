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
    const BindingContext& bindingContext;
    IteratorValueMap valueMap;
    std::queue<std::shared_ptr<Iterator>> workingSet;
    IteratorEvaluator(const BindingContext& bindingContext);

    void evaluateTensorAccess(const TensorView& tensor);
};

} // namespace kas
