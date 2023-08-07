#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

#include <fmt/format.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Transforms.hpp"


namespace kas {

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
    void visit(const MapReduceOp& op) override {
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
        for (const MapReduce *reduction: graph.getMapReduceIterators()) {
            fmt::format_to(SSIt(), "reduce_{} [label=\"{}\", shape=box];\n", reduction->getPriority(), reduction->whatReduce());
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
        for (const MapReduce *reduction: graph.getMapReduceIterators()) {
            fmt::format_to(SSIt(), "reduce_{};\n", reduction->getPriority());
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
    void visit(const MapReduce& dim) override {
        fmt::format_to(SSIt(), "{} -> reduce_{} [label=\"{}\"];\n", from, dim.getPriority(), opCanvas.attributedLabelForDim(&dim));
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
    void drawExpand(const Expand *op) {
        this->from = OpCanvas::Name(dynamic_cast<const ExpandOp&>(*op));
        op->output.accept(*this);
    }
    void drawOthers() {
        for (const Dimension& dim: graph.getDimensions()) {
            const PrimitiveOp *opAbove = graph.getOpAbove(dim);
            if (opAbove) {
                // This is not input.
                from = OpCanvas::Name(*opAbove);
                dim.accept(*this);
            } else {
                // This is input or expand, which will be handled elsewhere.
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
    // Draw the expands.
    for (const Expand *op: graph.getTopmost().getExpansions()) {
        dimCanvas.drawExpand(op);
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

void GraphvizGen::generate(const std::filesystem::path& outputPath, std::string_view funcName) const {
    std::filesystem::create_directories(outputPath.parent_path());
    std::ofstream file { outputPath };
    file << "digraph " << funcName << " {\n";
    file << code;
    file << "}\n";
    file.close();
}

std::string GraphvizGen::print(std::string_view funcName) const {
    return fmt::format("digraph {} {{\n{}}}\n", funcName, code);
}

} // namespace kas
