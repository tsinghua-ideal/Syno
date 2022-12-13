#include <cstddef>
#include <memory>
#include <string>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

Iterator::Iterator(IteratorTransform parent, std::shared_ptr<Size> size):
    parent { std::move(parent) },
    size { std::move(size) }
{}

namespace {
    class IteratorTransformVisitor {
    public:
        const BindingContext& ctx;
        std::queue<std::shared_ptr<Iterator>>& workingSet;
        IteratorValueMap& valueMap;
        const std::shared_ptr<Iterator>& thisIterator;
        bool operator()(std::unique_ptr<RepeatLikePrimitiveOp>& repeatLikeOp) {
            auto result = valueMap.find(repeatLikeOp->parent);
            if (result != valueMap.end()) {
                return true;
            }
            auto input = repeatLikeOp->value(
                valueMap.at(thisIterator), ctx
            );
            valueMap.emplace(repeatLikeOp->parent, input);
            workingSet.push(repeatLikeOp->parent);
            return true;
        }
        bool operator()(std::shared_ptr<SplitLikePrimitiveOp>& splitLikeOp) {
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
            }, ctx);
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
                valueMap.at(thisIterator), ctx
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
            ctx { evaluator.bindingContext },
            workingSet { evaluator.workingSet },
            valueMap { evaluator.valueMap },
            thisIterator { thisIterator }
        {}
    };
}

std::shared_ptr<Size> Iterator::getSize() const {
    return size;
}

bool Iterator::compute(IteratorEvaluator& iteratorEvaluator) {
    return std::visit(IteratorTransformVisitor { iteratorEvaluator, shared_from_this() }, parent);
}

std::shared_ptr<BinaryOpValueNode> IteratorValue::operator+(IteratorValue& other) {
    return std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Add, shared_from_this(), other.shared_from_this());
}
std::shared_ptr<BinaryOpValueNode> IteratorValue::operator-(IteratorValue& other) {
    return std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Sub, shared_from_this(), other.shared_from_this());
}
std::shared_ptr<BinaryOpValueNode> IteratorValue::operator*(IteratorValue& other) {
    return std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Mul, shared_from_this(), other.shared_from_this());
}
std::shared_ptr<BinaryOpValueNode> IteratorValue::operator%(IteratorValue& other) {
    return std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Mod, shared_from_this(), other.shared_from_this());
}
std::shared_ptr<BinaryOpValueNode> IteratorValue::operator/(IteratorValue& other) {
    return std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Div, shared_from_this(), other.shared_from_this());
}
std::vector<std::shared_ptr<IteratorValue>> IteratorValue::DefaultAccessForShape(const Shape& shape, BindingContext& ctx) {
    std::vector<std::shared_ptr<IteratorValue>> result;
    auto base = std::string("i_");
    for (std::size_t i = 0; i < shape.size(); ++i) {
        auto id = ctx.addIteratorVariable(base + std::to_string(i));
        result.push_back(std::make_shared<VariableValueNode>(id));
    }
    return result;
}
VariableValueNode::VariableValueNode(std::size_t variableId):
    variableId { variableId }
{}
void VariableValueNode::accept(IteratorValueVisitor& visitor) {
    visitor.visit(*this);
}
ConstValueNode::ConstValueNode(std::shared_ptr<Size> value):
    value { std::move(value) }
{}
void ConstValueNode::accept(IteratorValueVisitor& visitor) {
    visitor.visit(*this);
}
ImmediateValueNode::ImmediateValueNode(int value):
    value { value }
{}
void ImmediateValueNode::accept(IteratorValueVisitor& visitor) {
    visitor.visit(*this);
}
BinaryOpValueNode::BinaryOpValueNode(Type type, std::shared_ptr<IteratorValue> op1, std::shared_ptr<IteratorValue> op2):
    type { type },
    op1 { std::move(op1) },
    op2 { std::move(op2) }
{}
void BinaryOpValueNode::accept(IteratorValueVisitor& visitor) {
    visitor.visit(*this);
}

IteratorValuePrinter::IteratorValuePrinter(const BindingContext& ctx):
    ctx { ctx }
{}
void IteratorValuePrinter::visit(VariableValueNode& value) {
    ss << ctx.getIteratorVariableName(value.variableId);
}
void IteratorValuePrinter::visit(ConstValueNode& value) {
    ss << value.value->toString(ctx);
}
void IteratorValuePrinter::visit(ImmediateValueNode &value) {
    ss << value.value;
}
void IteratorValuePrinter::visit(BinaryOpValueNode& value) {
    ss << "(";
    value.op1->accept(*this);
    ss << ")";
    switch (value.type) {
        case BinaryOpValueNode::Type::Add:
            ss << "+";
            break;
        case BinaryOpValueNode::Type::Sub:
            ss << "-";
            break;
        case BinaryOpValueNode::Type::Mul:
            ss << "*";
            break;
        case BinaryOpValueNode::Type::Div:
            ss << "/";
            break;
        case BinaryOpValueNode::Type::Mod:
            ss << "%";
            break;
    }
    ss << "(";
    value.op2->accept(*this);
    ss << ")";
}
std::string IteratorValuePrinter::toString(IteratorValue& value) {
    value.accept(*this);
    std::string result = ss.str();
    ss.str("");
    return result;
}

} // namespace kas
