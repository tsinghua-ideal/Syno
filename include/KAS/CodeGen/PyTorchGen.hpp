#pragma once

#include <queue>

#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/TensorView.hpp"


namespace kas {

template<typename F>
requires std::invocable<F, const Dimension&>
class ShareBlockDiscoverer {
    const Graph& graph;
    Dimension bottommost;
    F f;
public:
    void operator()(const RepeatLikeVertex&, auto) {}
    void operator()(const SplitLikeVertex&, auto) {}
    void operator()(const MergeLikeVertex& vertex, MergeLikeOp::Branch) {
        if (vertex.op.getType() != DimensionType::Share) {
            return;
        }
        // This is a ShareOp. First collect.
        f(vertex[MergeLikeOp::Branch::InputLhs]);
        f(vertex[MergeLikeOp::Branch::InputRhs]);
        // Then propagate.
        propagateTo(vertex.visitAdjacent(MergeLikeOp::Branch::InputLhs));
        propagateTo(vertex.visitAdjacent(MergeLikeOp::Branch::InputRhs));
    }
    void operator()(const ExpandVertex&, auto) {}
    void propagateTo(VisitedVertex vertex) {
        vertex.match(*this);
    }
    ShareBlockDiscoverer(const Graph& graph, Dimension dim, const Graph::CutSet& boundary, F&& f):
        graph { graph },
        bottommost { dim }, // temporary.
        f { std::forward<F>(f) }
    {
        // First find the bottom-most Share dimension.
        while (dim.type() == DimensionType::Share && !boundary.contains(dim)) {
            dim = dim.as<MergeLikeOp::Input>().getOp()->output;
        }
        bottommost = dim;
    }
    Dimension getBottommost() const { return bottommost; }
    Dimension traverse() {
        f(bottommost);
        propagateTo(graph.visitAlong(bottommost, Direction::Up));
        return bottommost;
    }
};

class EinsumTensorContractor: protected DependentCutSetDiscoverer {
    std::set<const ShareOp *>& remainingShares;
public:
    EinsumTensorContractor(const Graph& graph, std::set<const ShareOp *>& remainingShares):
        DependentCutSetDiscoverer(graph), remainingShares(remainingShares) {}
    template<DimensionRange R>
    EinsumTensorContractor& with(R&& dims) {
        includeUnchecked(std::forward<R>(dims));
        return *this;
    }
    EinsumTensorContractor& contract();
    Graph::CutSet dump() { return std::move(cutSet); }
};

// For PyTorch codegen, further split Tensor's apart so that contractions are apparent, that is, ShareOp's are above any other type of Op's in each Tensor.
class PerformViewsIRPass {
    Graph graph;
    IR& ir;
public:
    class ViewPerformer: protected DependentCutSetDiscoverer {
        Tensor tensor;
        ConstrainedGraph subgraph;

        std::set<const ShareOp *> remainingShares;

        enum class State: bool {
            Disabled, // Share descendants.
            Collected, // Reachable by views.
        };
        Graph::DimensionMap<State> visited;

        int warn = 0;

        Graph::DimensionSet einsumContract();
        template<State Marker, bool StopAtShare>
        void mark(const Dimension& dim);
        void disable(const Dimension& dim);
        void collect(const Dimension& dim);
    public:
        ViewPerformer(const Graph& graph, Tensor& tensor);
        ViewPerformer& shouldWarn(int level) { warn = level + 1; return *this; }
        void apply();
    };
    PerformViewsIRPass(IR& ir);
    void apply();
};

class PythonCodePrinter {
    std::ostringstream& oss;
    bool isNewLine = true;
    std::size_t indentLevel;
    void writeIndent() {
        for (std::size_t i = 0; i < indentLevel; ++i) oss << '\t';
    }
public:
    template<typename F>
    void indent(F&& f) {
        const std::size_t oldIndentLevel = indentLevel++;
        std::invoke(f);
        indentLevel = oldIndentLevel;
    };
    PythonCodePrinter(std::ostringstream& oss, std::size_t indentLevel):
        oss { oss },  indentLevel { indentLevel } {}
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
};

class PyTorchGen {
    const BindingContext& ctx;
    IR ir;
    Graph graph;

    std::vector<Tensor> topologicallyOrderedTensors;

    std::map<Tensor, std::string> tensorNames;
    const std::string& use(const Tensor& tensor) const {
        auto it = tensorNames.find(tensor);
        KAS_ASSERT(it != tensorNames.end());
        return it->second;
    }
    bool declared(const Tensor& tensor) {
        return tensorNames.find(tensor) != tensorNames.end();
    }
    void declare(const Tensor& tensor, std::string_view name) {
        auto [_, inserted] = tensorNames.emplace(tensor, name);
        KAS_ASSERT(inserted);
    }
    void declare(const Tensor& tensor) {
        declare(tensor, fmt::format("t_{}", tensorNames.size()));
    }

public:
    class SubgraphGen {
        const BindingContext& ctx;
        const Graph& graph;
        const std::map<Tensor, std::string>& tensorNames;

        PythonCodePrinter& printer;

        const Tensor& tensor;
        Graph::CutSet bottommost;
        std::set<const Reduce *> remainingReductions;

        // To perform contraction using einsum, we need to track the subscripts.
        std::size_t existingSubscripts = 0;
        std::map<Dimension, std::size_t, Dimension::AddressLessThan> subscripts;
        std::size_t newSubscript() { return existingSubscripts++; }
        void assignSubscript(Dimension dim, std::size_t subscript) {
            auto [_, inserted] = subscripts.emplace(dim, subscript);
            KAS_ASSERT(inserted);
        }
        // Returns the bottom-most dimension of the chain of ShareOp.
        std::pair<Dimension, std::size_t> assignShare(Dimension dim);
        std::vector<Dimension> performContraction(std::string_view name);

        struct OpLower final: public OpVisitor {
            const BindingContext& ctx;
            const Graph& graph;
            PythonCodePrinter& printer;
            const ConcreteConsts& consts;
            std::size_t concretize(const Size& size) const { return size.eval<std::size_t>(consts); }
            std::vector<Dimension>& interface;
            const Tensor& tensor;
            std::set<const Reduce *>& remainingReductions;
            const std::string& name;
            std::map<Dimension, std::size_t, Dimension::AddressLessThan> outputSet;
            OpLower(const BindingContext& ctx, const Graph& graph, PythonCodePrinter& printer, const ConcreteConsts& consts, std::vector<Dimension>& interface, const Tensor& tensor, std::set<const Reduce *>& remainingReductions, const std::string& name);

            bool successfulVisit = false;
            bool operator()(const auto& vertex, auto) {
                const auto& op = vertex.op;
                printer.writeLn("# {}", op.description(ctx));
                successfulVisit = false;
                vertex.op.accept(*this);
                bool result = successfulVisit;
                successfulVisit = false;
                printer.writeLn();
                return result;
            }
            void lower();

            template<PrimitiveOpImpl Op>
            std::pair<Dimension, std::size_t> getSingleInput(const Op& op) {
                Dimension input = op.getInput();
                std::size_t inputIndex = std::distance(interface.begin(), std::ranges::find(interface, input));
                KAS_ASSERT(inputIndex < interface.size());
                successfulVisit = true;
                return { std::move(input), inputIndex };
            }
            // Based on the shape of interface, reshape the PyTorch tensor to this.
            void reshapeToInterface();
            // Reshape the PyTorch tensor to NCHW, making the specified dimension as height. Note that all other dimensions sizes are computed from interface, except for the size of the specified dimension.
            std::array<std::size_t, 4> reshapeToNCHW(std::size_t heightIndexInInterface, std::size_t heightSize);

            void visit(const ExpandOp& op) override { KAS_CRITICAL("Cannot lower Expand to PyTorch as an Op."); }
            void visit(const ReduceOp& op) override { KAS_CRITICAL("Cannot lower Reduce to PyTorch as an Op."); }
            void visit(const MergeOp& op) override;
            void visit(const ShareOp& op) override;
            void visit(const ShiftOp& op) override;
            void visit(const SplitOp& op) override;
            void visit(const StrideOp& op) override;
            void visit(const UnfoldOp& op) override;
        };

    public:
        SubgraphGen(const BindingContext& ctx, const Graph& graph, const std::map<Tensor, std::string>& tensorNames, PythonCodePrinter& printer, const Tensor& tensor);

        template<std::ranges::input_range R>
        requires std::same_as<std::ranges::range_value_t<R>, std::size_t>
        static std::string ToEinsteinNotation(R&& subscripts) {
            std::string result;
            for (std::size_t subscript: subscripts) {
                result += 'i' + static_cast<char>(subscript);
            }
            return result;
        }

        void generate(const ConcreteConsts& consts);
    };

    std::vector<std::size_t> concretize(const std::vector<Dimension>& interface, const ConcreteConsts& consts) const;

    PyTorchGen(const BindingContext& ctx, const IR& subgraphs);
    PyTorchGen(const BindingContext& ctx, const TensorView& tensorView):
        PyTorchGen { ctx, tensorView.getSubgraphs() } {}
    void loadWeights(PythonCodePrinter& printer) const;
    void padInputTensor(PythonCodePrinter& printer, const PaddedConsts& consts) const;
    void cropOutputTensor(PythonCodePrinter& printer, const PaddedConsts& consts) const;
    void applyDivision(PythonCodePrinter& printer, const ConcreteConsts& consts, const AbstractAccess& forwardAccess) const;
    void generatePrelude(std::ostream& outputStream) const;
    void generate(std::ostream& outputStream, std::string_view className, const AbstractAccess& forwardAccess, const PaddedConsts& consts) const;
    void generateSingle(const std::filesystem::path& outputPath, std::string_view className, const TensorView& tensorView, const std::map<std::string, std::size_t>& mappings) const;
};

} // namespace kas
