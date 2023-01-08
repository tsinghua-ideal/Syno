#include <cstddef>
#include <functional>
#include <sstream>
#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

CodeGenContext::TensorMetadata::TensorMetadata(std::string_view name):
    name { name }
{}

CodeGenContext::IteratorVariableMetadata::IteratorVariableMetadata(std::string_view name):
    name { name }
{}

std::string_view CodeGenContext::getTensorName(std::size_t index) const {
    return tensorMetadata.at(index).name;
}

std::size_t CodeGenContext::addTensor(std::string_view name) {
    tensorMetadata.emplace_back(name);
    return tensorMetadata.size() - 1;
}

std::string_view CodeGenContext::getIteratorVariableName(std::size_t index) const {
    return iteratorVariableMetadata.at(index).second.name;
}

std::size_t CodeGenContext::addIteratorVariable(std::shared_ptr<Iterator> iterator, bool isOuterLoopIterator) {
    auto id = iteratorVariableMetadata.size();
    iteratorVariableMetadata.emplace_back(std::move(iterator), "i_" + std::to_string(id));
    if (isOuterLoopIterator) {
        outerLoopIterators.push_back(id);
    }
    return id;
}

std::pair<std::string, std::size_t> CodeGenContext::printOuterLoopsHeader(const BindingContext& ctx) const {
    std::stringstream ss;
    std::size_t depth = 0;
    for (auto i: outerLoopIterators) {
        const auto& [it, name] = iteratorVariableMetadata[i];
        ss << std::string(4 * depth, ' ');
        ss << "for (int " << name.name << " = 0; " << name.name << " < " << it->getSize()->toString(ctx) << "; " << name.name << "++) {\n";
        ++depth;
    }
    return { ss.str(), depth };
}

std::string CodeGenContext::printOuterLoopsTail() const {
    std::stringstream ss;
    std::size_t depth = outerLoopIterators.size();
    while (depth --> 0) {
        ss << std::string(4 * depth, ' ') << "}\n";
    }
    return ss.str();
}

std::string CodeGenContext::outerLoopIteratorsToString() const {
    return VectorToString(outerLoopIterators, std::function([&](const std::size_t& id) -> std::string {
        return std::string(getIteratorVariableName(id));
    }));
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
std::vector<std::shared_ptr<IteratorValue>> IteratorValue::DefaultAccessForShape(const std::vector<std::shared_ptr<Iterator>>& interface, CodeGenContext& ctx) {
    std::vector<std::shared_ptr<IteratorValue>> result;
    for (const auto& it: interface) {
        auto id = ctx.addIteratorVariable(it, true);
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

IteratorValuePrinter::IteratorValuePrinter(const BindingContext& ctx, const CodeGenContext& cgCtx):
    ctx { ctx },
    cgCtx { cgCtx }
{}
void IteratorValuePrinter::visit(VariableValueNode& value) {
    ss << cgCtx.getIteratorVariableName(value.variableId);
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
