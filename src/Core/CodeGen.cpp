#include <cstddef>
#include <functional>
#include <sstream>
#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

IteratorValue IteratorValue::operator+(const IteratorValue& other) const {
    return IteratorValue(std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Add, *this, other));
}
IteratorValue IteratorValue::operator-(const IteratorValue& other) const {
    return IteratorValue(std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Sub, *this, other));
}
IteratorValue IteratorValue::operator*(const IteratorValue& other) const {
    return IteratorValue(std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Mul, *this, other));
}
IteratorValue IteratorValue::operator/(const IteratorValue& other) const {
    return IteratorValue(std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Div, *this, other));
}
IteratorValue IteratorValue::operator%(const IteratorValue& other) const {
    return IteratorValue(std::make_shared<BinaryOpValueNode>(BinaryOpValueNode::Type::Mod, *this, other));
}

ExpressionPrecedence BinaryOpValueNode::getPrecedence() const {
    switch (type) {
        case Type::Add: return ExpressionPrecedence::Add;
        case Type::Sub: return ExpressionPrecedence::Sub;
        case Type::Mul: return ExpressionPrecedence::Mul;
        case Type::Mod: return ExpressionPrecedence::Mod;
        case Type::Div: return ExpressionPrecedence::Div;
        default: KAS_UNREACHABLE();
    }
}

std::string IteratorValue::toString(const BindingContext& ctx) const {
    IteratorValuePrinter printer { ctx };
    return printer.toString(*this);
}

const IteratorValue ImmediateValueNode::Zero = ImmediateValueNode::Create(0);
const IteratorValue ImmediateValueNode::One = ImmediateValueNode::Create(1);
const IteratorValue ImmediateValueNode::Two = ImmediateValueNode::Create(2);

void IteratorValuePrinter::visit(VariableValueNode& value) {
    ss << value.name;
}
void IteratorValuePrinter::visit(ConstValueNode& value) {
    ss << value.value.toString(ctx);
}
void IteratorValuePrinter::visit(ImmediateValueNode &value) {
    ss << value.value;
}
void IteratorValuePrinter::visit(BinaryOpValueNode& value) {
    if (value.op1.getPrecedence() > value.getPrecedence())
        ss << "(";
    value.op1.accept(*this);
    if (value.op1.getPrecedence() > value.getPrecedence())
        ss << ")";
    switch (value.type) {
        case BinaryOpValueNode::Type::Add:
            ss << " + ";
            break;
        case BinaryOpValueNode::Type::Sub:
            ss << " - ";
            break;
        case BinaryOpValueNode::Type::Mul:
            ss << " * ";
            break;
        case BinaryOpValueNode::Type::Div:
            ss << " / ";
            break;
        case BinaryOpValueNode::Type::Mod:
            ss << " % ";
            break;
    }
    if (value.op2.getPrecedence() > value.getPrecedence())
        ss << "(";
    value.op2.accept(*this);
    if (value.op2.getPrecedence() > value.getPrecedence())
        ss << ")";
}
void IteratorValuePrinter::visit(IntervalBoundValueNode& value) {
    ss << "restrict(";
    value.input.accept(*this);
    ss << ", 0, ";
    ss << value.max.toString(ctx);
    ss << ")";
}
std::string IteratorValuePrinter::toString(const IteratorValue& value) {
    value.accept(*this);
    std::string result = ss.str();
    ss.str("");
    return result;
}

TensorExpression TensorExpression::operator+(const TensorExpression& other) const {
    return BinaryOpTensorExpression::Create(*this, other, BinaryOpTensorExpression::Op::Add);
}
TensorExpression& TensorExpression::operator+=(const TensorExpression& other) {
    return *this = *this + other;
}
TensorExpression TensorExpression::operator*(const TensorExpression& other) const {
    return BinaryOpTensorExpression::Create(*this, other, BinaryOpTensorExpression::Op::Mul);
}
TensorExpression& TensorExpression::operator*=(const TensorExpression& other) {
    return *this = *this * other;
}

std::string TensorExpression::toString() const {
    TensorExpressionPrinter p;
    return p.print(*this);
}

TensorExpression TensorExpression::ProductOfTensors(std::size_t numTensors) {
    KAS_ASSERT(numTensors >= 1);
    TensorExpression result = IntegerTensorExpression::Create(1);
    for (std::size_t i = 0; i < numTensors; ++i) {
        result *= TensorTensorExpression::Create(i);
    }
    return result;
}

TensorExpression TensorExpression::ProductAndPlusOneTensor(std::size_t numTensors) {
    KAS_ASSERT(numTensors >= 2);
    auto result = ProductOfTensors(numTensors - 1);
    result += TensorTensorExpression::Create(numTensors - 1);
    return result;
}

TensorExpression IntegerTensorExpression::Create(int value) {
    static auto zero = TensorExpression(std::make_shared<IntegerTensorExpression>(0));
    static auto one = TensorExpression(std::make_shared<IntegerTensorExpression>(1));
    if (value == 0) {
        return zero;
    } else if (value == 1) {
        return one;
    }
    return TensorExpression(std::make_shared<IntegerTensorExpression>(value));
}

ExpressionPrecedence BinaryOpTensorExpression::getPrecedence() const {
    switch (op) {
    case Op::Add: return ExpressionPrecedence::Add;
    case Op::Mul: return ExpressionPrecedence::Mul;
    default: KAS_UNREACHABLE();
    }
}

TensorExpression BinaryOpTensorExpression::Create(TensorExpression lhs, TensorExpression rhs, Op op) {
    switch (op) {
    case Op::Add:
        if (auto lhsInt = lhs.tryAs<IntegerTensorExpression>(); lhsInt) {
            if (lhsInt->value == 0) {
                return rhs;
            }
            if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
                return TensorExpression(std::make_shared<IntegerTensorExpression>(lhsInt->value + rhsInt->value));
            }
        } else if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
            if (rhsInt->value == 0) {
                return lhs;
            }
        }
        break;
    case Op::Mul:
        if (auto lhsInt = lhs.tryAs<IntegerTensorExpression>(); lhsInt) {
            if (lhsInt->value == 0) {
                return lhs;
            } else if (lhsInt->value == 1) {
                return rhs;
            }
            if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
                return TensorExpression(std::make_shared<IntegerTensorExpression>(lhsInt->value * rhsInt->value));
            }
        } else if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
            if (rhsInt->value == 0) {
                return rhs;
            } else if (rhsInt->value == 1) {
                return lhs;
            }
        }
        break;
    }
    return TensorExpression(std::make_shared<BinaryOpTensorExpression>(std::move(lhs), std::move(rhs), op));
}

void TensorExpressionPrinter::visit(IntegerTensorExpression& expr) {
    ss << expr.value;
}
void TensorExpressionPrinter::visit(TensorTensorExpression& expr) {
    ss << TensorExpression::PositionToString(expr.position);
}
void TensorExpressionPrinter::visit(BinaryOpTensorExpression& expr) {
    if (expr.lhs.getPrecedence() > expr.getPrecedence()) {
        ss << "(";
        expr.lhs.accept(*this);
        ss << ")";
    } else {
        expr.lhs.accept(*this);
    }
    switch (expr.op) {
    case BinaryOpTensorExpression::Op::Add:
        ss << " + ";
        break;
    case BinaryOpTensorExpression::Op::Mul:
        ss << " * ";
        break;
    }
    if (expr.rhs.getPrecedence() > expr.getPrecedence()) {
        ss << "(";
        expr.rhs.accept(*this);
        ss << ")";
    } else {
        expr.rhs.accept(*this);
    }
}
std::string TensorExpressionPrinter::print(const TensorExpression& expr) {
    expr.accept(*this);
    std::string result = ss.str();
    ss.str("");
    return result;
}

} // namespace kas
