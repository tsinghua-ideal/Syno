#pragma once

#include <cstddef>
#include <sstream>
#include <variant>
#include <memory>
#include <map>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

using IteratorTransform = std::variant<
    std::unique_ptr<RepeatLikePrimitiveOp>,
    std::shared_ptr<SplitLikePrimitiveOp>,
    std::unique_ptr<MergeLikePrimitiveOp>,
    TensorStub
>;

class IteratorEvaluator;

class Iterator: public std::enable_shared_from_this<Iterator> {
protected:
    IteratorTransform parent;
    std::shared_ptr<Size> size;
public:
    Iterator(IteratorTransform parent, std::shared_ptr<Size> size);

    std::shared_ptr<Size> getSize() const;

    // Returns true on success
    bool compute(IteratorEvaluator& iteratorEvaluator);
};

} // namespace kas
