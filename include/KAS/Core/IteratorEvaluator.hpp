#pragma once

#include <map>
#include <memory>
#include <queue>
#include <vector>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Shape.hpp"


namespace kas {

using IteratorValueMap = std::map<std::shared_ptr<Iterator>, IteratorValue>;

class IteratorEvaluator {
public:
    IteratorValueMap valueMap;
    std::queue<std::shared_ptr<Iterator>> workingSet;
    IteratorEvaluator() = default;

    void evaluateTensorAccess(TensorView& tensor);
};

} // namespace kas
