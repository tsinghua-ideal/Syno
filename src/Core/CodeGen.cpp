#include "KAS/Core/CodeGen.hpp"


namespace kas {

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
    std::string base("i_");
    for (std::size_t i = 0; i < shape.size(); ++i) {
        auto id = ctx.addIteratorVariable(base + std::to_string(i));
        result.emplace_back(std::make_shared<VariableValueNode>(id));
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
