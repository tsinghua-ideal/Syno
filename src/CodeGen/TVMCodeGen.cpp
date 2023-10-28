#include <fstream>
#include <ranges>

#include "KAS/CodeGen/TVMCodeGen.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Algorithm.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

TVMConcreteSize TVMConcreteSize::BinaryOp(Precedence precedence, const TVMConcreteSize& lhs, bool lhsParen, const TVMConcreteSize& rhs, bool rhsParen, std::string_view op) {
    TVMConcreteSize result { .precedence = precedence, .hasIterator = lhs.hasIterator || rhs.hasIterator };
    std::string& value = result.value;
    if (lhsParen)
        if (rhsParen) value = fmt::format("({}) {} ({})", lhs.value, op, rhs.value);
        else value = fmt::format("({}) {}, {}", lhs.value, op, rhs.value);
    else
        if (rhsParen) value = fmt::format("{} {} ({})", lhs.value, op, rhs.value);
        else value = fmt::format("{} {} {}", lhs.value, op, rhs.value);
    return result;
}

TVMConcreteSize TVMConcreteSize::operator+(const TVMConcreteSize& rhs) const {
    bool lhsParen = precedence < Precedence::Expr;
    bool rhsParen = rhs.precedence < Precedence::Expr;
    return BinaryOp(Precedence::Expr, *this, lhsParen, rhs, rhsParen, "+");
}

TVMConcreteSize TVMConcreteSize::operator-(const TVMConcreteSize& rhs) const {
    bool lhsParen = precedence < Precedence::Expr;
    bool rhsParen = rhs.precedence <= Precedence::Expr;
    return BinaryOp(Precedence::Expr, *this, lhsParen, rhs, rhsParen, "-");
}

TVMConcreteSize TVMConcreteSize::operator*(const TVMConcreteSize& rhs) const {
    bool lhsParen = precedence < Precedence::Term;
    bool rhsParen = rhs.precedence <= Precedence::Term;
    return BinaryOp(Precedence::Term, *this, lhsParen, rhs, rhsParen, "*");
}

TVMConcreteSize TVMConcreteSize::operator/(const TVMConcreteSize& rhs) const {
    bool lhsParen = precedence < Precedence::Term;
    bool rhsParen = rhs.precedence <= Precedence::Term;
    return BinaryOp(Precedence::Term, *this, lhsParen, rhs, rhsParen, "//");
}

TVMConcreteSize TVMConcreteSize::operator%(const TVMConcreteSize& rhs) const {
    bool lhsParen = precedence < Precedence::Remainder;
    bool rhsParen = rhs.precedence <= Precedence::Remainder;
    return BinaryOp(Precedence::Remainder, *this, lhsParen, rhs, rhsParen, "%");
}

TVMConcreteSize TVMConcreteSize::floorDiv(const TVMConcreteSize& rhs) const {
    // If we have no iterator here, just use div, because Python uses floor division by default.
    if (!hasIterator) {
        KAS_ASSERT(!rhs.hasIterator);
        return *this / rhs;
    }
    return TVMConcreteSize {
        .precedence = Precedence::Factor,
        .value = fmt::format("te.floordiv({}, {})", value, rhs.value),
    };
}

TVMConcreteSize TVMConcreteSize::floorMod(const TVMConcreteSize& rhs) const {
    // Same reason.
    if (!hasIterator) {
        KAS_ASSERT(!rhs.hasIterator);
        return *this % rhs;
    }
    return TVMConcreteSize {
        .precedence = Precedence::Factor,
        .value = fmt::format("te.floormod({}, {})", value, rhs.value),
    };
}

TVMConcreteSize TVMConcreteSize::Literal(std::size_t value) {
    return TVMConcreteSize {
        .precedence = Precedence::Factor,
        .value = std::to_string(value),
        .hasIterator = false,
    };
}
TVMConcreteSize TVMConcreteSize::Iterator(std::string name) {
    return TVMConcreteSize {
        .precedence = Precedence::Factor,
        .value = std::move(name),
        .hasIterator = true,
    };
}

TVMConcreteSize TVMSizeConcretizer::concretize(const Size& size) const {
    auto primaryMeta = ctx.getPrimaryMetadata();
    auto coefficientMeta = ctx.getCoefficientMetadata();
    std::vector<std::pair<std::string, int>> numerators, denominators;
    auto addFactor = [&](const std::string& alias, int power) {
        if (power > 0) {
            numerators.emplace_back(alias, power);
        } else if (power < 0) {
            denominators.emplace_back(alias, -power);
        }
    };
    for (std::size_t i = 0; auto p: size.getPrimary()) {
        addFactor(primaryMeta[i++].alias, p);
    }
    for (std::size_t i = 0; auto c: size.getCoefficient()) {
        addFactor(coefficientMeta[i++].alias, c);
    }
    auto prod = fmt::format("{}", fmt::join(
        numerators | std::views::transform([](const auto& p) {
            if (p.second == 1) return p.first;
            return fmt::format("{} ** {}", p.first, p.second);
        }), " * "
    ));
    if (numerators.empty()) {
        prod = "1";
    }
    auto prodDenominators = fmt::format("{}", fmt::join(
        denominators | std::views::transform([](const auto& p) {
            if (p.second == 1) return p.first;
            return fmt::format("{} ** {}", p.first, p.second);
        }), " // "
    ));
    if (!prodDenominators.empty()) {
        prod += " // ";
        prod += prodDenominators;
    }
    return {
        .precedence = denominators.empty() && numerators.size() <= 1 ? TVMConcreteSize::Precedence::Factor : TVMConcreteSize::Precedence::Term,
        .value = std::move(prod),
        .hasIterator = false,
    };
}

void TVMOpLower::assignOutput(const std::vector<Dimension>& dims) {
    for (std::size_t i = 0; const Dimension& dim: dims) {
        assign(dim, TVMConcreteSize::Iterator(IteratorNameForOutput(i++)));
    }
}
void TVMOpLower::assignReductions(const std::vector<const Reduce *>& reductions) {
    for (std::size_t i = 0; const Reduce *reduction: reductions) {
        assign(reduction, TVMConcreteSize::Iterator(IteratorNameForReduction(i++)));
    }
}

TVMConcreteSize TVMOpLower::valueOf(const Dimension& dim) {
    if (auto it = valuations.find(dim); it != valuations.end()) return it->second;
    auto op = dim.getOpBelow();
    KAS_ASSERT(op != nullptr, "Have you assigned all the output of this subgraph?");
    op->accept(*this);
    return valuations.at(dim);
}

void TVMOpLower::visit(const ExpandOp& op) { KAS_UNREACHABLE(); }
void TVMOpLower::visit(const ReduceOp& op) { KAS_UNREACHABLE(); }
void TVMOpLower::visit(const MergeOp& op) {
    auto output = valueOf(op.output);
    auto block = concretizer.concretize(op.getBlock());
    assign(op.getInputL(), output.floorDiv(block));
    assign(op.getInputR(), output.floorMod(block));
}
void TVMOpLower::visit(const ShareOp& op) {
    auto output = valueOf(op.output);
    assign(op.getInputL(), output);
    assign(op.getInputR(), output);
}
void TVMOpLower::visit(const ShiftOp& op) {
    auto output = valueOf(op.output);
    auto shift = TVMConcreteSize::Literal(op.getShift());
    assign(op.getInput(), (output + shift).floorMod(concretizer.concretize(op.output.size())));
}
void TVMOpLower::visit(const SplitOp& op) {
    auto outputLhs = valueOf(op.outputLhs);
    auto outputRhs = valueOf(op.outputRhs);
    auto block = concretizer.concretize(op.getBlock());
    assign(op.getInput(), outputLhs * block + outputRhs);
}
void TVMOpLower::visit(const StrideOp& op) {
    auto output = valueOf(op.output);
    auto stride = concretizer.concretize(op.getStride());
    assign(op.getInput(), output * stride);
}
void TVMOpLower::visit(const UnfoldOp& op) {
    auto outputLhs = valueOf(op.outputLhs);
    auto outputRhs = valueOf(op.outputRhs);
    auto window = concretizer.concretize(op.getWindow());
    auto halfWindow = window / TVMConcreteSize::Literal(2);
    auto input = outputLhs + outputRhs - halfWindow;
    assign(op.getInput(), input);
    bounds.emplace_back(input, window);
}

std::vector<TVMConcreteSize> TVMOpLower::eval(const std::vector<Dimension>& dims) {
    return ranges::to<std::vector<TVMConcreteSize>>(
        dims | std::views::transform([this](const Dimension& dim) { return valueOf(dim); })
    );
}

void TVMCodeGen::generateImports() {
    printer.writeLn("import tvm");
    printer.writeLn("from tvm import relax, te");
    printer.writeLn("from tvm.relax import BlockBuilder");
    printer.writeLn("from typing import List");
    printer.writeLn("import numpy as np");
    printer.writeLn();
}

void TVMCodeGen::generateMappingsParams() {
    auto arg = [&](const auto& meta) {
        int estimate = -1;
        if (meta.estimate.has_value()) {
            estimate = *meta.estimate;
        }
        printer.write("{}: int = {}, ", meta.alias, estimate);
    };
    for (const auto& meta: ctx.getPrimaryMetadata()) arg(meta);
    for (const auto& meta: ctx.getCoefficientMetadata()) arg(meta);
}

void TVMCodeGen::generateWeights() {
    std::size_t index = 0;
    for (const Tensor& tensor: ir.inputTensors | std::views::drop(1)) {
        auto name = VarNameForInput(++index);
        variables.try_emplace(tensor, name);
        printer.writeLn(
            "{}: relax.Constant = relax.const(np.random.normal(size=({},)).astype(\"float32\"))",
            name, fmt::join(
                tensor.output() | std::views::transform([&](const Dimension& dim) {
                    return concretizer.concretize(dim.size()).value;
                }), ", "
            )
        );
    }
}

void TVMCodeGen::generateWeightsBuilder() {
    printer.write("def weights(");
    generateMappingsParams();
    printer.writeLn(") -> List[relax.Constant]:");
    printer.indent([&] {
        generateWeights();
        printer.writeLn("return [{}]", fmt::join(
            ir.inputTensors | std::views::drop(1) | std::views::transform([&](const Tensor& tensor) -> const std::string& {
                return variables.at(tensor);
            }), ", "
        ));
    });
}

void TVMCodeGen::generateBuilderArgs() {
    printer.write("bb: BlockBuilder, ");
    variables.try_emplace(ir.inputTensors.at(0), VarNameForInput(0));
    for (std::size_t i = 0; const Tensor& inputTensor: ir.inputTensors) {
        auto name = VarNameForInput(i++);
        printer.write("{}: relax.Expr, ", variables.at(inputTensor));
    }
    generateMappingsParams();
}

void TVMCodeGen::generateAssertions() {
    printer.write("assert ");
    bool comma = false;
    auto positive = [&](const auto& meta) {
        if (comma) printer.write(" and ");
        comma = true;
        printer.write("{} > 0", meta.alias);
    };
    for (const auto& meta: ctx.getPrimaryMetadata()) positive(meta);
    for (const auto& meta: ctx.getCoefficientMetadata()) positive(meta);
    printer.writeLn();
    KAS_ASSERT(comma);
}

void TVMCodeGen::generateSubgraph(const Tensor& tensor) {
    if (variables.contains(tensor)) return;
    std::size_t subgraphId = variables.size();
    auto myName = fmt::format("subgraph_{}", subgraphId);
    variables.try_emplace(tensor, myName);
    for (const Tensor& inputTensor: tensor.inputs()) {
        generateSubgraph(inputTensor);
    }

    auto opLower = TVMOpLower { concretizer };
    opLower.assignOutput(tensor.output());
    opLower.assignReductions(tensor.reductions());

    std::vector<std::vector<TVMConcreteSize>> inputs;
    for (const Tensor& inputTensor: tensor.inputs()) {
        inputs.emplace_back(opLower.eval(inputTensor.output()));
    }

    auto indices = [&](const std::vector<TVMConcreteSize>& is) {
        printer.write("[{}]", fmt::join(
            is | std::views::transform([](const TVMConcreteSize& i) { return i.value; }), ", "
        ));
    };
    auto expression = [&] {
        for (std::size_t i = 0; const auto& input: inputs) {
            if (i != 0) printer.write(" * ");
            printer.write("{}", VarNameForInput(i));
            indices(input);
            ++i;
        }
        if (divBy) {
            auto divByFactor = concretizer.concretize(*divBy);
            if (divByFactor.precedence <= TVMConcreteSize::Precedence::Term) {
                printer.write(" / ({})", divByFactor.value);
            } else {
                printer.write(" / {}", divByFactor.value);
            }
            divBy = std::nullopt;
        }
        printer.writeLn(",");
    };
    auto guardCond = [&] {
        printer.writeLn(
            "te.all({}),",
            fmt::join(
                opLower.bounds | std::views::transform([](const auto& bound) {
                    return fmt::format("{0} >= 0, {0} < {1}", bound.first.value, bound.second.value);
                }), ", "
            )
        );
    };
    auto guardedExpression = [&] {
        if (opLower.bounds.empty()) {
            expression();
        } else {
            printer.write("te.if_then_else");
            printer.parens<true>([&] {
                guardCond();
                expression();
                printer.writeLn("0.0,");
            });
        }
    };
    auto summedExpression = [&] {
        if (tensor.reductions().empty()) {
            guardedExpression();
        } else {
            printer.write("te.sum");
            printer.parens<true>([&] {
                guardedExpression();
                printer.writeLn("axis=[{}],", fmt::join(
                    std::views::iota(0_uz, tensor.reductions().size())
                    | std::views::transform([](std::size_t i) {
                        return TVMOpLower::IteratorNameForReduction(i);
                    }), ", "
                ));
            });
        }
    };

    printer.writeLn(
        "def build_{}({}) -> te.Tensor:",
        myName,
        fmt::join(
            std::views::iota(0_uz, tensor.inputs().size())
            | std::views::transform([](std::size_t i) {
                return VarNameForInput(i) + ": te.Tensor";
            }), ", "
        )
    );
    printer.indent([&] {
        for (std::size_t i = 0; const Reduce *reduction: tensor.reductions()) {
            printer.writeLn(
                "{0} = te.reduce_axis((0, {1}), \"{0}\")",
                TVMOpLower::IteratorNameForReduction(i++),
                concretizer.concretize(reduction->size()).value
            );
        }
        printer.write("return te.compute");
        printer.parens([&] {
            printer.writeLn("({},),", fmt::join(
                tensor.output() | std::views::transform([&](const Dimension& dim) {
                    return concretizer.concretize(dim.size()).value;
                }), ", "
            ));
            printer.writeLn("lambda {}:", fmt::join(
                std::views::iota(0_uz, tensor.output().size())
                | std::views::transform([](std::size_t i) {
                    return TVMOpLower::IteratorNameForOutput(i);
                }), ", "
            ));
            printer.indent([&] {
                summedExpression();
            });
            printer.writeLn("name=\"{}\",", myName);
        });
    });
}

void TVMCodeGen::generateCalls() {
    ir.topBottomForEach([&](const Tensor& tensor) {
        if (!tensor.isInputTensor()) {
            printer.writeLn("{0} = bb.emit_te(build_{0}, {1})", variables.at(tensor), fmt::join(
                tensor.inputs() | std::views::transform([&](const Tensor& input) {
                    return variables.at(input);
                }), ", "
            ));
        }
    });
}

TVMCodeGen::TVMCodeGen(const BindingContext& ctx, const IR& ir):
    ctx { ctx }, concretizer { ctx }, ir { ir }, graph { ir.buildGraph() }, printer { code, 0 }
{
    divBy = ranges::fold_left_first(
        graph.getReduceIterators()
        | std::views::filter([](const Reduce *r) {
            return r->getReduce() == Reduce::ReduceType::Mean;
        })
        | std::views::transform([](const Reduce *r) -> const Size& {
            return r->getBase().getDomain();
        }),
        std::multiplies<>{}
    );
    generateImports();
    generateWeightsBuilder();
    printer.writeLn();
    printer.write("def build(");
    generateBuilderArgs();
    printer.writeLn(") -> relax.Var:");
    printer.indent([this] {
        generateAssertions();
        generateSubgraph(this->ir.outputTensor);
        generateCalls();
        printer.writeLn("return {}", variables.at(this->ir.outputTensor));
    });
}

void TVMCodeGen::generate(std::ostream& outputStream) const {
    outputStream << code.str();
}

void TVMCodeGen::generate(const std::filesystem::path& path) const {
    std::ofstream file { path };
    generate(file);
}

} // namespace kas
