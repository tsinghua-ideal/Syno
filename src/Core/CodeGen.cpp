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

std::string CodeGenContext::getTensorName(std::size_t index) const {
    return tensorMetadata.at(index).name;
}

std::size_t CodeGenContext::addTensor(std::string_view name) {
    tensorMetadata.emplace_back(name);
    return tensorMetadata.size() - 1;
}

std::string CodeGenContext::getIteratorVariableName(std::size_t index) const {
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
    return VectorToString(outerLoopIterators | std::ranges::views::transform([&](const std::size_t& id) {
        return getIteratorVariableName(id);
    }));
}

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
std::vector<IteratorValue> IteratorValue::DefaultAccessForShape(const std::vector<std::shared_ptr<Iterator>>& interface, CodeGenContext& ctx) {
    std::vector<IteratorValue> result;
    for (const auto& it: interface) {
        auto id = ctx.addIteratorVariable(it, true);
        result.emplace_back(std::make_shared<VariableValueNode>(id));
    }
    return result;
}

const IteratorValue ImmediateValueNode::Zero = ImmediateValueNode::Create(0);
const IteratorValue ImmediateValueNode::One = ImmediateValueNode::Create(1);
const IteratorValue ImmediateValueNode::Two = ImmediateValueNode::Create(2);

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
