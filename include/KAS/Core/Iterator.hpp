#pragma once

#include <variant>
#include <memory>
#include <map>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

using IteratorTransform = std::variant<
    std::unique_ptr<RepeatLikePrimitiveOp>,
    std::unique_ptr<SplitLikePrimitiveOp>,
    std::unique_ptr<MergeLikePrimitiveOp>,
    TensorStub
>;

class IteratorEvaluator;

class Iterator: public std::enable_shared_from_this<Iterator> {
public:
    IteratorTransform parent;
    std::shared_ptr<Size> size;
    Iterator(IteratorTransform parent, std::shared_ptr<Size> size);
    // Returns true on success
    bool compute(IteratorEvaluator& iteratorEvaluator);
};

class IteratorValue {
public:
    // Now we simplify the AST to a string. TODO
    std::string content;
};

} // namespace kas
