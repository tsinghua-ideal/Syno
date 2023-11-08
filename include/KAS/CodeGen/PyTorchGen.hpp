#pragma once

#include "KAS/CodeGen/Python.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/IR.hpp"
#include "KAS/Core/Pass.hpp"
#include "KAS/Core/TensorView.hpp"


namespace kas {

// For PerformViewIRPass.
class EinsumDiscoverer final: protected DependentCutSetDiscoverer {
    const ConstrainedGraph& subgraph;
    std::set<const ShareOp *>& remainingShares;
public:
    EinsumDiscoverer(const ConstrainedGraph& subgraph, std::set<const ShareOp *>& remainingShares);
    // Return collected Share blocks, representing them with bottommost dimensions.
    Graph::DimensionSet contract();
};

// For SubgraphGen.
class EinsumContractor final: protected DependentCutSetDiscoverer {
    const ConstrainedGraph& subgraph;
    std::set<const Reduce *>& remainingReductions;

    // To perform contraction using einsum, we need to track the subscripts.
    std::size_t existingSubscripts = 0;
    Graph::DimensionMap<std::size_t> subscripts;
    std::size_t newSubscript() { return existingSubscripts++; }
    void assignSubscript(const Dimension& dim, std::size_t subscript) {
        auto [_, inserted] = subscripts.try_emplace(dim, subscript);
        KAS_ASSERT(inserted);
    }

public:
    EinsumContractor(const ConstrainedGraph& subgraph, std::set<const Reduce *>& remainingReductions);
    EinsumContractor& contract();
    std::vector<Dimension> build(const std::vector<Tensor>& inputs) const;
    void beforeExclusionHook(const PrimitiveOp *op) override;
    const Graph::DimensionMap<std::size_t>& getSubscripts() const { return subscripts; }
};

// For PyTorch codegen, further split Tensor's apart so that contractions are apparent, that is, ShareOp's are above any other type of Op's in each Tensor.
class PerformViewsIRPass {
    const Graph& graph;
public:
    class ViewPerformer {
        Tensor tensor;
        ConstrainedGraph subgraph;

        std::set<const ShareOp *> remainingShares;

        enum class State: bool {
            Disabled, // Share descendants.
            Collected, // Reachable by views.
        };
        Graph::DimensionMap<State> visited;

        int warn = 0;

        std::vector<Dimension> einsumContract();
        template<State Marker, bool StopAtShare>
        void mark(const Dimension& dim);
        void disable(const Dimension& dim);
        template<DimensionRange R>
        void disable(R&& dims) {
            for (const Dimension& dim: dims) disable(dim);
        }
        void collect(const Dimension& dim);
        template<DimensionRange R>
        void collect(R&& dims) {
            for (const Dimension& dim: dims) collect(dim);
        }
        // From `visited`, find `ShareOp::Input`s. They are crucial for einsum contraction.
        std::vector<Dimension> buildStage(const std::vector<Dimension>& cutSet) const;
    public:
        ViewPerformer(const Graph& graph, Tensor& tensor);
        ViewPerformer& shouldWarn(int level) { warn = level + 1; return *this; }
        void apply();
    };
    PerformViewsIRPass(const Graph& graph);
    void operator()(IR& ir) const;
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
        const std::map<Tensor, std::string>& tensorNames;

        PythonCodePrinter& printer;

        const Tensor& tensor;
        ConstrainedGraph subgraph;
        std::set<const Reduce *> remainingReductions;

        struct OpLower final: public OpVisitor, private DependentCutSetDiscoverer {
            const BindingContext& ctx;
            const ConstrainedGraph& subgraph;
            const Graph& graph;
            PythonCodePrinter& printer;
            const ConcreteConsts& consts;
            std::size_t concretize(const Size& size) const { return size.eval<std::size_t>(consts); }
            std::vector<Dimension>& interface;
            const Tensor& tensor;
            const std::string& name;

            std::map<Dimension, const ExpandOp *, Dimension::AddressLessThan> undoneExpansions;

            OpLower(const BindingContext& ctx, const ConstrainedGraph& subgraph, PythonCodePrinter& printer, const ConcreteConsts& consts, std::vector<Dimension>& interface, const Tensor& tensor, const std::string& name);

            template<typename Op>
            std::pair<Dimension, std::size_t> getSingleInput(const Op& op) {
                Dimension input = op.getInput();
                std::size_t inputIndex = std::distance(interface.begin(), std::ranges::find(interface, input));
                KAS_ASSERT(inputIndex < interface.size());
                return { std::move(input), inputIndex };
            }
            // Based on the shape of interface, reshape the PyTorch tensor to this.
            void reshapeToInterface();
            // Reshape the PyTorch tensor to NCHW, making the specified dimension as height. Note that all other dimensions sizes are computed from interface, except for the size of the specified dimension.
            std::array<std::size_t, 4> reshapeToNCHW(std::size_t heightIndexInInterface, std::size_t heightSize);

            void visit(const ExpandOp& op) override;
            void visit(const ReduceOp& op) override { KAS_CRITICAL("Cannot lower Reduce to PyTorch as an Op."); }
            void visit(const Reduce& reduction);
            void visit(const MergeOp& op) override;
            void visit(const ShareOp& op) override { KAS_CRITICAL("Cannot lower Share to PyTorch as an Op."); }
            void visit(const ShiftOp& op) override;
            void visit(const SplitOp& op) override;
            void visit(const StrideOp& op) override;
            void visit(const UnfoldOp& op) override;
            // Repeat == Expand + Merge.
            void visitRepeat(const ExpandOp& expandOp, const MergeOp& mergeOp);

            void afterExclusionHook(const PrimitiveOp *op) override;
            using DependentCutSetDiscoverer::include;

            void checkDone() const;
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

    PyTorchGen(const BindingContext& ctx, const TensorView& tensorView);
    void loadWeights(PythonCodePrinter& printer) const;
    void padInputTensor(PythonCodePrinter& printer, const PaddedConsts& consts) const;
    void cropOutputTensor(PythonCodePrinter& printer, const PaddedConsts& consts) const;
    void applyDivision(PythonCodePrinter& printer, const ConcreteConsts& consts, const AbstractAccess& forwardAccess) const;
    void generatePrelude(std::ostream& outputStream) const;
    void generate(std::ostream& outputStream, std::string_view className, const AbstractAccess& forwardAccess, const PaddedConsts& consts) const;
    void generateSingle(const std::filesystem::path& outputPath, std::string_view className, const TensorView& tensorView, const std::map<std::string, std::size_t>& mappings) const;
};

} // namespace kas
