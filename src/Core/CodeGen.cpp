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
    ss << "(";
    value.op1.accept(*this);
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
    value.op2.accept(*this);
    ss << ")";
}
void IteratorValuePrinter::visit(IntervalBoundValueNode& value) {
    ss << "restrict(";
    value.input.accept(*this);
    ss << ",";
    value.min.accept(*this);
    ss << ",";
    value.max.accept(*this);
    ss << ")";
}
std::string IteratorValuePrinter::toString(const IteratorValue& value) {
    value.accept(*this);
    std::string result = ss.str();
    ss.str("");
    return result;
}

} // namespace kas
