#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"
#include <memory>


namespace kas {

Iterator::Iterator(IteratorTransform parent, std::shared_ptr<Size> size):
    parent { std::move(parent) },
    size { std::move(size) }
{}

namespace {
    class IteratorTransformVisitor {
    public:
        std::queue<std::shared_ptr<Iterator>>& workingSet;
        IteratorValueMap& valueMap;
        const std::shared_ptr<Iterator>& thisIterator;
        bool operator()(std::unique_ptr<RepeatLikePrimitiveOp>& repeatLikeOp) {
            auto result = valueMap.find(repeatLikeOp->parent);
            if (result != valueMap.end()) {
                return true;
            }
            auto input = repeatLikeOp->value(
                valueMap.at(thisIterator)
            );
            valueMap.emplace(repeatLikeOp->parent, input);
            workingSet.push(repeatLikeOp->parent);
            return true;
        }
        bool operator()(std::unique_ptr<SplitLikePrimitiveOp>& splitLikeOp) {
            auto result = valueMap.find(splitLikeOp->parent);
            if (result != valueMap.end()) {
                return true;
            }
            auto outputLhs = valueMap.find(splitLikeOp->childLhs.lock());
            auto outputRhs = valueMap.find(splitLikeOp->childRhs.lock());
            if (outputLhs == valueMap.end() || outputRhs == valueMap.end()) {
                // Values of children not computed yet, so retry later
                workingSet.push(thisIterator);
                return false;
            }
            auto input = splitLikeOp->value({
                outputLhs->second, 
                outputRhs->second
            });
            valueMap.emplace(splitLikeOp->parent, input);
            workingSet.push(splitLikeOp->parent);
            return true;
        }
        bool operator()(std::unique_ptr<MergeLikePrimitiveOp>& mergeLikeOp) {
            auto result = valueMap.find(mergeLikeOp->parentLhs);
            if (result != valueMap.end()) {
                return true;
            }
            auto [inputLhs, inputRhs] = mergeLikeOp->value(
                valueMap.at(thisIterator)
            );
            valueMap.emplace(mergeLikeOp->parentLhs, inputLhs);
            workingSet.push(mergeLikeOp->parentLhs);
            valueMap.emplace(mergeLikeOp->parentRhs, inputRhs);
            workingSet.push(mergeLikeOp->parentRhs);
            return true;
        }
        bool operator()(TensorStub& tensorStub) {
            tensorStub.setAccess(valueMap.at(thisIterator));
            return true;
        }
        IteratorTransformVisitor(IteratorEvaluator& evaluator, const std::shared_ptr<Iterator>& thisIterator):
            workingSet { evaluator.workingSet },
            valueMap { evaluator.valueMap },
            thisIterator { thisIterator }
        {}
    };
}

bool Iterator::compute(IteratorEvaluator& iteratorEvaluator) {
    return std::visit(IteratorTransformVisitor { iteratorEvaluator, shared_from_this() }, parent);
}

} // namespace kas
