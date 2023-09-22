#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

#include <fmt/format.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Transforms.hpp"


namespace kas {

namespace detail {

void GraphvizCode::generate(const std::filesystem::path& outputPath, std::string_view funcName) const {
    std::filesystem::create_directories(outputPath.parent_path());
    std::ofstream file { outputPath };
    file << "digraph " << funcName << " {\n";
    file << code;
    file << "}\n";
    file.close();
}

std::string GraphvizCode::print(std::string_view funcName) const {
    return fmt::format("digraph {} {{\n{}}}\n", funcName, code);
}

} // namespace detail

namespace {

struct OpCanvas: public OpVisitor {
    const BindingContext& ctx;
    const Graph& graph;
    std::stringstream& ss;
    struct Attribute {
        std::optional<Order> input, output;
    };
    std::map<Dimension, Attribute, Dimension::AddressLessThan> attributes;

    OpCanvas(const BindingContext& ctx, const Graph& graph, std::stringstream& ss):
        ctx { ctx },
        graph { graph },
        ss { ss }
    {}

    auto SSIt() { return std::ostreambuf_iterator<char>(ss); }
    static std::string Name(const PrimitiveOp& op) {
        return fmt::format("op_{}", fmt::ptr(&op));
    }
    static std::string Name(const Reduce& op) {
        return fmt::format("reduce_{}", fmt::ptr(&op));
    }

    void visits(const RepeatLikeOp& op) {
        fmt::format_to(SSIt(), "{} [label=\"{}\"];\n", Name(op), op.getType());
    }
    void visits(const SplitLikeOp& op) {
        fmt::format_to(SSIt(), "{} [label=\"{}\"];\n", Name(op), op.getType());
        attributes[op.outputLhs].input = Order::Left;
        attributes[op.outputRhs].input = Order::Right;
    }
    void visits(const MergeLikeOp& op) {
        fmt::format_to(SSIt(), "{} [label=\"{}\"];\n", Name(op), op.getType());
        attributes[op.getInputL()].output = Order::Left;
        attributes[op.getInputR()].output = Order::Right;
    }
    void visit(const ExpandOp& op) override {
        fmt::format_to(SSIt(), "{} [label=\"{}\"];\n", Name(op), op.getType());
    }
    void visit(const ReduceOp& op) override {
        // The is left for other procedures to draw.
    }
    void visit(const MergeOp& op) override { visits(op); }
    void visit(const ShareOp& op) override { visits(op); }
    void visit(const ShiftOp& op) override { visits(op); }
    void visit(const SplitOp& op) override { visits(op); }
    void visit(const StrideOp& op) override { visits(op); }
    void visit(const UnfoldOp& op) override { visits(op); }

    std::string attributedLabelForDim(const Dimension& dim) {
        auto [input, output] = attributes[dim];
        std::string label = dim.size().toString(ctx);
        if (!input && !output) {
            return label;
        } else {
            return fmt::format("{} ({}->{})", label, OrderToLR(input).value_or(""), OrderToLR(output).value_or(""));
        }
    }

    void draw() {
        // Draw the Ops.
        for (const PrimitiveOp *op: graph.getOps()) {
            op->accept(*this);
        }

        // Draw the reductions.
        for (const Reduce *reduction: graph.getReduceIterators()) {
            fmt::format_to(SSIt(), "{} [label=\"{}\", shape=box];\n", Name(*reduction), reduction->whatReduce());
        }

        // Draw the outputs.
        ss << "subgraph cluster_out {\n";
        ss << "label = \"Output\";\n";
        for (const Iterator *output: graph.getOutputIterators()) {
            fmt::format_to(SSIt(), "out_{} [label=\"{}\", shape=none];\n", output->getIndex(), output->size().toString(ctx));
        }
        ss << "}\n";

        // Align the reductions and outputs.
        ss << "{ rank = same;\n";
        for (const Reduce *reduction: graph.getReduceIterators()) {
            fmt::format_to(SSIt(), "{};\n", Name(*reduction));
        }
        for (const Iterator *output: graph.getOutputIterators()) {
            fmt::format_to(SSIt(), "out_{};\n", output->getIndex());
        }
        ss << "}\n";
    }
};

struct DimCanvas: public DimVisitor {
    const Graph& graph;
    OpCanvas& opCanvas;
    std::stringstream& ss;
    std::string from;
    DimCanvas(const Graph& graph, OpCanvas& opCanvas, std::stringstream& ss):
        graph { graph },
        opCanvas { opCanvas },
        ss { ss }
    {}
    auto SSIt() { return std::ostreambuf_iterator<char>(ss); }
    void visit(const Iterator& dim) override {
        fmt::format_to(SSIt(), "{} -> out_{} [label=\"{}\"];\n", from, dim.getIndex(), opCanvas.attributedLabelForDim(&dim));
    }
    void visit(const Reduce& dim) override {
        fmt::format_to(SSIt(), "{} -> {} [label=\"{}\"];\n", from, OpCanvas::Name(dim), opCanvas.attributedLabelForDim(&dim));
    }
    void visit(const RepeatLikeOp::Input& dim) override {
        auto op = dim.getOp();
        fmt::format_to(SSIt(), "{} -> {} [label=\"{}\"];\n", from, OpCanvas::Name(*op), opCanvas.attributedLabelForDim(&dim));
    }
    void visit(const SplitLikeOp::Input& dim) override {
        auto op = dim.getOp();
        fmt::format_to(SSIt(), "{} -> {} [label=\"{}\"];\n", from, OpCanvas::Name(*op), opCanvas.attributedLabelForDim(&dim));
    }
    void visit(const MergeLikeOp::Input& dim) override {
        auto op = dim.getOp();
        fmt::format_to(SSIt(), "{} -> {} [label=\"{}\"];\n", from, OpCanvas::Name(*op), opCanvas.attributedLabelForDim(&dim));
    }
    void drawInput(const Dimension& dim, std::string from) {
        this->from = std::move(from);
        dim.accept(*this);
    }
    void drawOthers() {
        for (const Dimension& dim: graph.getDimensions()) {
            const PrimitiveOp *opAbove = graph.getOpAbove(dim);
            if (opAbove) {
                // This is not input.
                from = OpCanvas::Name(*opAbove);
                dim.accept(*this);
            } else {
                // This is input, which will be handled elsewhere.
            }
        }
    }
};

template<TopmostRange R>
std::string draw(const BindingContext& ctx, R&& tensors) {
    std::stringstream ss;
    auto SSIt = [&]() { return std::ostreambuf_iterator<char>(ss); };

    Graph graph =
        Graph::Builder()
        .addTopmosts(tensors)
        .build();

    ss << "newrank = true;\n"; // To allow alignment for subgraphs.

    // First draw all the Ops.
    OpCanvas opCanvas { ctx, graph, ss };
    opCanvas.draw();

    // Input labels.
    for (std::size_t j = 0; auto&& tensor: tensors) {
        fmt::format_to(SSIt(), "subgraph cluster_in_{} {{\n", j);
        fmt::format_to(SSIt(), "label = \"Input {}\";\n", j);
        for (std::size_t i = 0; auto&& dim: tensor.getDimensions()) {
            fmt::format_to(SSIt(), "in_{}_{} [label=\"{}\", shape=none];\n", j, i, dim.size().toString(ctx));
            ++i;
        }
        ss << "}\n";
        ++j;
    }

    // Align the input labels.
    ss << "{ rank = same;\n";
    for (std::size_t j = 0; auto&& tensor: tensors) {
        for (std::size_t i = 0; i < tensor.getDimensions().size(); ++i) {
            fmt::format_to(SSIt(), "in_{}_{};\n", j, i);
        }
        ++j;
    }
    ss << "}\n";

    // Draw the inputs.
    DimCanvas dimCanvas { graph, opCanvas, ss };
    for (std::size_t j = 0; auto&& tensor: tensors) {
        for (std::size_t i = 0; auto&& dim: tensor.getDimensions()) {
            dimCanvas.drawInput(dim, fmt::format("in_{}_{}", j, i));
            ++i;
        }
        ++j;
    }
    dimCanvas.drawOthers();
    return ss.str();
}

} // namespace

GraphvizGen::GraphvizGen(const Topmost& inputs, const BindingContext& ctx) {
    auto oneTensor = { inputs };
    code = draw(ctx, oneTensor);
}

GraphvizGen::GraphvizGen(const std::vector<Topmost>& tensors, const BindingContext& ctx) {
    code = draw(ctx, tensors);
}

GraphvizGen::GraphvizGen(const TensorView& tensorView, const BindingContext& ctx) {
    code = draw(ctx, tensorView.getUnderlyingTensors() | std::views::transform(&PureTensor::getContent));
}

std::string GraphvizDFGGen::InterfaceName(const Dimension& dim, std::size_t subgraphIndex, Direction type) {
    return fmt::format("interface_{}_{}_{}", subgraphIndex, type == Direction::Up ? "in" : "out", fmt::ptr(dim.getInnerPointer()));
}
std::string GraphvizDFGGen::Name(const PrimitiveOp& op) {
    return fmt::format("op_{}", fmt::ptr(&op));
}
std::string GraphvizDFGGen::Name(const Reduce& op) {
    return fmt::format("reduce_{}", fmt::ptr(&op));
}

void GraphvizDFGGen::drawDFGEdge(const Tensor& from, std::size_t to) {
    for (const Dimension& dim: from.output()) {
        printer.writeLn("{} -> {};", InterfaceName(dim, subgraphIndex.at(from), Direction::Down), InterfaceName(dim, to, Direction::Up));
    }
}

void GraphvizDFGGen::drawTensor(const Tensor& tensor) {
    auto [it, inserted] = subgraphIndex.try_emplace(tensor, subgraphIndex.size());
    if (!inserted) return;
    auto index = it->second;

    if (tensor.isInputTensor()) {
        // Only need to draw the box.
        printer.writeLn("// Input tensor.");
        drawTensorBox(
            fmt::format("subgraph_{}", index), fmt::format("Input {}", index),
            index, Direction::Down, tensor.output()
        );
    } else {
        printer.writeLn("// Stage tensor.");
        auto subgraph = ConstrainedGraph::Builder(graph)
            .addTop(tensor.inputs() | std::views::transform(&Tensor::output) | std::views::join)
            .addBottom(tensor.output())
            .addBottom(tensor.reductions())
            .build();

        // TODO: if this tensor needs to be stored, indicate in the label.
        auto _ = printer.scope("subgraph cluster_subgraph_{0}", index);
        printer.writeLn("label = \"Subgraph {0}\";", index);

        // First draw the reductions.
        printer.writeLn("// Reductions.");
        for (const Reduce *reduction: tensor.reductions()) {
            printer.writeLn("{} [label=\"{}\", shape=box];", Name(*reduction), reduction->whatReduce());
        }

        // Then draw the output.
        printer.writeLn("// Output.");
        drawTensorBox(
            fmt::format("subgraph_{}_out", index), "",
            index, Direction::Down, tensor.output()
        );

        // Align the output with reductions.
        {
            auto _ = printer.scope();
            printer.writeLn("rank = same;");
            for (const Dimension& dim: tensor.output()) {
                printer.writeLn("{};", Name(dim.as<Reduce>()));
            }
            for (const Dimension& dim: tensor.output()) {
                printer.writeLn("{};", InterfaceName(dim, index, Direction::Down));
            }
        }

        // Draw the inputs.
        for (std::size_t i = 0; const Tensor& input: tensor.inputs()) {
            printer.writeLn("// Input {}.", i);
            drawTensorBox(
                fmt::format("subgraph_{}_in_{}", index, i), "",
                index, Direction::Up, input.output()
            );
            ++i;
        }
        // Align all the inputs.
        {
            auto _ = printer.scope();
            printer.writeLn("rank = same;");
            for (const Tensor& input: tensor.inputs()) {
                for (const Dimension& dim: input.output()) {
                    printer.writeLn("{};", InterfaceName(dim, index, Direction::Up));
                }
            }
        }

        // Finally draw the Op's.
        printer.writeLn("// Op's.");
        for (const PrimitiveOp *op: subgraph.getOps()) {
            printer.writeLn("{} [label=\"{}\"];", Name(*op), op->getType());
        }

        // And the dimensions.
        printer.writeLn("// Dimension's.");
        auto matcher = Match {
            [](GeneralizedVertex auto&& vertex, auto) {
                return Name(vertex.op);
            },
            [index](Direction type, const Dimension& dim) {
                return InterfaceName(dim, index, type);
            },
        };
        for (const Dimension& dim: subgraph.getDimensions()) {
            std::string fro = subgraph.visitAlong(dim, Direction::Up).match(matcher);
            std::string to = subgraph.visitAlong(dim, Direction::Down).match(matcher);
            printer.writeLn("{} -> {} [label=\"{}\"];", fro, to, dim.size().toString(ctx));
        }
    }
    printer.writeLn();

    // DFS.
    for (const Tensor& input: tensor.inputs()) {
        drawTensor(input);
        drawDFGEdge(input, index);
        printer.writeLn();
    }
}

GraphvizDFGGen::GraphvizDFGGen(const IR& subgraphs, const BindingContext& ctx):
    ctx { ctx },
    graph { subgraphs.buildGraph() }
{
    printer.writeLn("newrank = true;"); // To allow alignment for subgraphs.
    printer.writeLn();

    // DFS.
    drawTensor(subgraphs.outputTensor);

    // Align input tensors.
    {
        auto _ = printer.scope();
        printer.writeLn("rank = same;");
        for (const Tensor& input: subgraphs.inputTensors) {
            std::size_t index = subgraphIndex.at(input);
            for (const Dimension& dim: input.output()) {
                printer.writeLn("{};", InterfaceName(dim, index, Direction::Down));
            }
        }
    }

    // Draw output box.
    std::size_t outputIndex = subgraphIndex.size();
    drawTensorBox("subgraph_output", "Output", outputIndex, Direction::Up, subgraphs.outputTensor.output());

    // And link the output box.
    drawDFGEdge(subgraphs.outputTensor, outputIndex);

    printer.writeLn();

    code = ss.str();
}

} // namespace kas
