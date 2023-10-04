#pragma once

#include "KAS/CodeGen/Python.hpp"
#include "KAS/Core/IR.hpp"
#include "KAS/Transforms/Transforms.hpp"


namespace kas {

struct TVMConcreteSize {
    enum class Precedence: int {
        Expr,
        Remainder,
        Term,
        Factor,
    };
    Precedence precedence;
    std::string value;
    bool hasIterator;
    static TVMConcreteSize BinaryOp(Precedence precedence, const TVMConcreteSize& lhs, bool lhsParen, const TVMConcreteSize& rhs, bool rhsParen, std::string_view op);
    TVMConcreteSize operator+(const TVMConcreteSize& rhs) const;
    TVMConcreteSize operator-(const TVMConcreteSize& rhs) const;
    TVMConcreteSize operator*(const TVMConcreteSize& rhs) const;
    TVMConcreteSize operator/(const TVMConcreteSize& rhs) const;
    TVMConcreteSize operator%(const TVMConcreteSize& rhs) const;
    TVMConcreteSize floorDiv(const TVMConcreteSize& rhs) const;
    TVMConcreteSize floorMod(const TVMConcreteSize& rhs) const;
    static TVMConcreteSize Literal(std::size_t value);
    static TVMConcreteSize Iterator(std::string name);
};

struct TVMSizeConcretizer {
    const BindingContext& ctx;
    TVMConcreteSize concretize(const Size& size) const;
};

struct TVMOpLower final: OpVisitor {
    const TVMSizeConcretizer& concretizer;
    Graph::DimensionMap<TVMConcreteSize> valuations;
    // For UnfoldOp.
    std::vector<std::pair<TVMConcreteSize, TVMConcreteSize>> bounds;
    TVMOpLower(const TVMSizeConcretizer& concretizer): concretizer { concretizer } {}
    template<typename... Args>
    void assign(const Dimension& dim, Args&&... args) {
        auto [it, inserted] = valuations.try_emplace(dim, std::forward<Args>(args)...);
        KAS_ASSERT(inserted);
    }
    static std::string IteratorNameForOutput(std::size_t index) { return "i_" + std::to_string(index); }
    static std::string IteratorNameForReduction(std::size_t index) { return "ri_" + std::to_string(index); }
    void assignOutput(const std::vector<Dimension>& dims);
    void assignReductions(const std::vector<const Reduce *>& reductions);
    TVMConcreteSize valueOf(const Dimension& dim);
    void visit(const ExpandOp& op);
    void visit(const ReduceOp& op);
    void visit(const MergeOp& op);
    void visit(const ShareOp& op);
    void visit(const ShiftOp& op);
    void visit(const SplitOp& op);
    void visit(const StrideOp& op);
    void visit(const UnfoldOp& op);
    std::vector<TVMConcreteSize> eval(const std::vector<Dimension>& dims);
};

class TVMCodeGen {
    const BindingContext& ctx;
    TVMSizeConcretizer concretizer;
    const IR& ir;
    Graph graph;
    std::ostringstream code;
    PythonCodePrinter printer;

    std::optional<Size> divBy;

    static std::string VarNameForInput(std::size_t index) { return "in_" + std::to_string(index); }
    std::map<Tensor, std::string> variables;

    void generateImports();
    void generateMappingsParams();
    void generateWeights();
    void generateWeightsBuilder();
    void generateBuilderArgs();
    void generateAssertions();
    // DFS. Generates `build_subgraph_` functions for each subgraph.
    void generateSubgraph(const Tensor& tensor);
    void generateCalls();
public:
    TVMCodeGen(const BindingContext& ctx, const IR& ir);
    void generate(std::ostream& outputStream) const;
    void generate(const std::filesystem::path& path) const;
};

} // namespace kas
