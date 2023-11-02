#pragma once

#include <filesystem>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/TensorView.hpp"


namespace kas {

namespace detail {
    class GraphvizCode {
    protected:
        std::string code;
    public:
        void generate(const std::filesystem::path& outputPath, std::string_view funcName) const;
        std::string print(std::string_view funcName) const;
    };
};

class GraphvizCodePrinter {
    std::ostringstream& oss;
    bool isNewLine = true;
    std::size_t indentLevel = 0;
    void writeIndent() {
        for (std::size_t i = 0; i < indentLevel; ++i) oss << "    ";
    }
    struct ScopeGuard {
        GraphvizCodePrinter& printer;
        template<typename... Args>
        ScopeGuard(GraphvizCodePrinter& printer, fmt::format_string<Args...> format, Args&&... args):
            printer { printer }
        {
            printer.write(format, std::forward<Args>(args)...);
            printer.writeLn(" {{");
            ++printer.indentLevel;
        }
        ScopeGuard(GraphvizCodePrinter& printer):
            printer { printer }
        {
            printer.writeLn("{{");
            ++printer.indentLevel;
        }
        ~ScopeGuard() {
            --printer.indentLevel;
            printer.writeLn("}}");
        }
    };
public:
    GraphvizCodePrinter(std::ostringstream& oss): oss { oss } {}
    template<typename... Args>
    void write(fmt::format_string<Args...> format, Args&&... args) {
        if (isNewLine) {
            writeIndent();
            isNewLine = false;
        }
        fmt::format_to(std::ostreambuf_iterator(oss), format, std::forward<Args>(args)...);
    }
    void writeLn() {
        oss << "\n";
        isNewLine = true;
    }
    template<typename... Args>
    void writeLn(fmt::format_string<Args...> format, Args&&... args) {
        write(format, std::forward<Args>(args)...);
        writeLn();
    }
    template<typename... Args>
    [[nodiscard]] ScopeGuard scope(fmt::format_string<Args...> format, Args&&... args) {
        return ScopeGuard(*this, format, std::forward<Args>(args)...);
    }
    [[nodiscard]] ScopeGuard scope() {
        return ScopeGuard(*this);
    }
};

class GraphvizGen: public detail::GraphvizCode {
public:
    GraphvizGen(const Topmost& inputs, const BindingContext& ctx);
    GraphvizGen(const std::vector<Topmost>& tensors, const BindingContext& ctx);
    GraphvizGen(const TensorView& tensorView, const BindingContext& ctx);
    static std::string Print(const TensorView& tensorView, const BindingContext& ctx);
    static std::string Print(const Topmost& inputs, const BindingContext& ctx);
    // Functions to emphasize some Dimension's. TODO
};

class GraphvizDFGGen: public detail::GraphvizCode {
    const BindingContext& ctx;
    Graph graph;
    std::map<Tensor, std::size_t> inputIndex;
    std::map<Tensor, std::size_t> subgraphIndex;

    std::ostringstream ss;
    GraphvizCodePrinter printer { ss };

    static std::string InterfaceName(const Dimension& dim, std::size_t subgraphIndex, Direction type);
    static std::string Name(const PrimitiveOp& op);
    static std::string Name(const Reduce& op);
    void drawDFGEdge(const Tensor& from, std::size_t to);
    template<DimensionRange R>
    void drawTensorBox(std::string_view name, std::string_view label, std::size_t subgraphIndex, Direction type, R&& dims) {
        auto _ = printer.scope("subgraph cluster_{}", name);
        printer.writeLn("label = \"{}\";", label);
        for (const Dimension& dim: dims) {
            printer.writeLn("{} [label=\"{}\", shape=none];", InterfaceName(dim, subgraphIndex, type), dim.size().toString(ctx));
        }
    }
    // Returns subgraph index.
    void drawTensor(const Tensor& tensor);

public:
    GraphvizDFGGen(const IR& subgraphs, const BindingContext& ctx);
    static std::string Print(const IR& subgraphs, const BindingContext& ctx);
};

} // namespace kas
