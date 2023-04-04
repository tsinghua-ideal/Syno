#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

#include <fmt/format.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

namespace {
    // DFS the dimensions.
    class DFS final: public DimVisitor {
        const BindingContext& ctx;
        std::stringstream& ss;
        std::vector<const Iterator *>& outputs;
        std::vector<const MapReduceOp *>& reductions;
        std::set<Dimension, Dimension::AddressLessThan> drawn;
        std::set<const RepeatLikeOp *> drawnRepeatLikes;
        std::set<const SplitLikeOp *> drawnSplitLikes;
        std::set<const MergeLikeOp *> drawnMergeLikes;
        std::string from;
        auto SSIt() { return std::ostreambuf_iterator<char>(ss); }
        void visit(const Iterator& dim) override {
            outputs.emplace_back(&dim);
            fmt::format_to(SSIt(), "{} -> out_{} [label=\"{}\"];\n", from, dim.getIndex(), dim.size().toString(ctx));
        }
        void visit(const MapReduceOp& dim) override {
            reductions.emplace_back(&dim);
            fmt::format_to(SSIt(), "{} -> reduce_{} [label=\"{}\"];\n", from, dim.getPriority(), dim.size().toString(ctx));
        }
        void visit(const RepeatLikeOp::Input& dim) override {
            auto op = dim.getOp();
            auto to = fmt::format("repeat_like_{}", fmt::ptr(op));
            auto [_, toDraw] = drawnRepeatLikes.insert(op);
            if (toDraw) {
                fmt::format_to(SSIt(), "{} [label=\"{}\"];\n", to, DimensionTypeDescription(op->getType()));
            }
            fmt::format_to(SSIt(), "{} -> {} [label=\"{}\"];\n", from, to, dim.size().toString(ctx));
            from = std::move(to);
            operator()(dim.getOp()->output);
        }
        void visit(const SplitLikeOp::Input& dim) override {
            auto op = dim.getOp();
            auto to = fmt::format("split_like_{}", fmt::ptr(op));
            auto [_, toDraw] = drawnSplitLikes.insert(op);
            if (toDraw) {
                fmt::format_to(SSIt(), "{} [label=\"{}\"];\n", to, DimensionTypeDescription(op->getType()));
            }
            fmt::format_to(SSIt(), "{} -> {} [label=\"{}\"];\n", from, to, dim.size().toString(ctx));
            from = to;
            operator()(dim.getOp()->outputLhs);
            from = std::move(to);
            operator()(dim.getOp()->outputRhs);
        }
        void visit(const MergeLikeOp::Input& dim) override {
            auto op = dim.getOp();
            auto to = fmt::format("merge_like_{}", fmt::ptr(op));
            auto [_, toDraw] = drawnMergeLikes.insert(op);
            if (toDraw) {
                fmt::format_to(SSIt(), "{} [label=\"{}\"];\n", to, DimensionTypeDescription(op->getType()));
            }
            fmt::format_to(SSIt(), "{} -> {} [label=\"{}\"];\n", from, to, dim.size().toString(ctx));
            from = std::move(to);
            operator()(dim.getOp()->output);
        }
        using DimVisitor::visit;
        void operator()(const Dimension& edge) {
            auto [_, toDraw] = drawn.insert(edge);
            if (toDraw) {
                visit(edge);
            }
        }
    public:
        DFS(const BindingContext& ctx, std::stringstream& ss, std::vector<const Iterator *>& outputs, std::vector<const MapReduceOp *>& reductions):
            ctx { ctx },
            ss { ss },
            outputs { outputs },
            reductions { reductions }
        {}
        void operator()(std::string_view from, const Dimension& edge) {
            this->from = std::string { from };
            operator()(edge);
        }
        void done() {
            std::ranges::sort(reductions, {}, [](const MapReduceOp * i) { return i->getPriority(); });
            for (std::size_t i = 0; auto&& reduction: reductions) {
                fmt::format_to(SSIt(), "reduce_{} [label=\"{}\", shape=box];\n", i, reduction->whatReduce());
                ++i;
            }

            std::ranges::sort(outputs, {}, [](const Iterator * i) { return i->getIndex(); });
            ss << "subgraph cluster_out {\n";
            ss << "label = \"Output\";\n";
            for (std::size_t i = 0; auto&& output: outputs) {
                fmt::format_to(SSIt(), "out_{} [label=\"{}\", shape=none];\n", i, output->size().toString(ctx));
                ++i;
            }
            ss << "}\n";
        }
    };
}

GraphvizGen::GraphvizGen(const Interface& inputs, const BindingContext& ctx) {
    std::stringstream ss;
    auto SSIt = [&]() { return std::ostreambuf_iterator<char>(ss); };
    std::vector<const Iterator *> outputs;
    std::vector<const MapReduceOp *> reductions;
    auto dfs = DFS { ctx, ss, outputs, reductions };

    ss << "subgraph cluster_in {\n";
    ss << "label = \"Input\";\n";
    for (std::size_t i = 0; auto&& inputDim: inputs) {
        fmt::format_to(SSIt(), "in_{} [label=\"{}\", shape=none];\n", i, inputDim.size().toString(ctx));
        ++i;
    }
    ss << "}\n";
    for (std::size_t i = 0; auto&& inputDim: inputs) {
        dfs(fmt::format("in_{}", i), inputDim);
        ++i;
    }
    dfs.done();
    code = ss.str();
}

GraphvizGen::GraphvizGen(const TensorView& tensorView, const BindingContext& ctx) {
    std::stringstream ss;
    auto SSIt = [&]() { return std::ostreambuf_iterator<char>(ss); };
    std::vector<const Iterator *> outputs;
    std::vector<const MapReduceOp *> reductions;
    auto dfs = DFS { ctx, ss, outputs, reductions };

    for (std::size_t j = 0; auto&& tensor: tensorView.getUnderlyingTensors()) {
        fmt::format_to(SSIt(), "subgraph cluster_in_{} {{\n", j);
        fmt::format_to(SSIt(), "label = \"Input {}\";\n", j);
        for (std::size_t i = 0; auto&& dim: tensor.dims) {
            fmt::format_to(SSIt(), "in_{}_{} [label=\"{}\", shape=none];\n", j, i, dim.size().toString(ctx));
            ++i;
        }
        ss << "}\n";
        ++j;
    }
    for (std::size_t j = 0; auto&& tensor: tensorView.getUnderlyingTensors()) {
        for (std::size_t i = 0; auto&& dim: tensor.dims) {
            dfs(fmt::format("in_{}_{}", j, i), dim);
            ++i;
        }
        ++j;
    }
    dfs.done();
    code = ss.str();
}

void GraphvizGen::generate(std::filesystem::path outputDirectory, std::string_view funcName) {
    std::filesystem::create_directories(outputDirectory);
    std::ofstream file { outputDirectory / fmt::format("{}.dot", funcName) };
    file << "digraph " << funcName << " {\n";
    file << code;
    file << "}\n";
    file.close();
}

} // namespace kas
