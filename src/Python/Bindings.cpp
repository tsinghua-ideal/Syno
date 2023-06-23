#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Statistics.hpp"
#include "KAS/Transforms/Forward.hpp"


PYBIND11_MODULE(kas_cpp_bindings, m) {
    m.doc() = "Python/C++ API bindings for KAS.";

    using namespace kas;

    pybind11::class_<SampleOptions>(m, "SampleOptions")
        .def(
            pybind11::init([](SampleOptions::Seed seed, std::size_t depth, std::size_t dimLowerBound, std::size_t dimUpperBound, std::size_t maximumTensors, std::size_t maximumReductions, float maxFLOPs, std::size_t maximumVariablesInSize, std::size_t maximumVariablesPowersInSize, std::string expressionOneTensor, std::string expressionTwoTensors, std::string expressionThreeTensors, std::string expressionFourTensors, std::size_t maximumFinalizations, bool allowWeightPermutation, std::size_t maxStridedDimSize, std::size_t maxUnfoldKernelSize, float minimumUnfoldRatio, float minimumMergeRatio, bool disallowDiscontinuousView, bool canonicalizeUnfoldOrder, bool disallowSplitRAboveUnfold, bool disallowUnfoldLAboveSplit, bool disallowMergeWithLargeBlockAboveUnfold, bool disallowUnfoldLAboveMergeR, bool disallowSplitRAboveStride, bool disallowStrideAboveSplit, bool disallowMergeWithLargeBlockAboveStride, bool disallowStrideAboveMergeR, bool disallowUnfoldLAboveShift, bool disallowShiftAboveUnfold, int maximumMerges, int maximumSplits, int maximumShifts, int maximumStrides, int maximumUnfolds, int maximumShares) {
                return SampleOptions {
                    .seed = seed,
                    .depth = depth,
                    .dimLowerBound = dimLowerBound,
                    .dimUpperBound = dimUpperBound,
                    .maximumTensors = maximumTensors,
                    .maximumReductions = maximumReductions,
                    .maxFLOPs = static_cast<std::size_t>(maxFLOPs),
                    .maximumVariablesInSize = maximumVariablesInSize,
                    .maximumVariablesPowersInSize = maximumVariablesPowersInSize,
                    .expressionOneTensor = expressionOneTensor,
                    .expressionTwoTensors = expressionTwoTensors,
                    .expressionThreeTensors = expressionThreeTensors,
                    .expressionFourTensors = expressionFourTensors,
                    .maximumFinalizations = maximumFinalizations,
                    .allowWeightPermutation = allowWeightPermutation,
                    .maxStridedDimSize = maxStridedDimSize,
                    .maxUnfoldKernelSize = maxUnfoldKernelSize,
                    .minimumUnfoldRatio = minimumUnfoldRatio,
                    .minimumMergeRatio = minimumMergeRatio,
                    .disallowDiscontinuousView = disallowDiscontinuousView,
                    .canonicalizeUnfoldOrder = canonicalizeUnfoldOrder,
                    .disallowSplitRAboveUnfold = disallowSplitRAboveUnfold,
                    .disallowUnfoldLAboveSplit = disallowUnfoldLAboveSplit,
                    .disallowMergeWithLargeBlockAboveUnfold = disallowMergeWithLargeBlockAboveUnfold,
                    .disallowUnfoldLAboveMergeR = disallowUnfoldLAboveMergeR,
                    .disallowSplitRAboveStride = disallowSplitRAboveStride,
                    .disallowStrideAboveSplit = disallowStrideAboveSplit,
                    .disallowMergeWithLargeBlockAboveStride = disallowMergeWithLargeBlockAboveStride,
                    .disallowStrideAboveMergeR = disallowStrideAboveMergeR,
                    .disallowUnfoldLAboveShift = disallowUnfoldLAboveShift,
                    .disallowShiftAboveUnfold = disallowShiftAboveUnfold,
                    .maximumMerges = maximumMerges,
                    .maximumSplits = maximumSplits,
                    .maximumShifts = maximumShifts,
                    .maximumStrides = maximumStrides,
                    .maximumUnfolds = maximumUnfolds,
                    .maximumShares = maximumShares,
                };
            }),
            pybind11::arg("seed") = DefaultSampleOptions.seed,
            pybind11::arg("depth") = DefaultSampleOptions.depth,
            pybind11::arg("dim_lower") = DefaultSampleOptions.dimLowerBound,
            pybind11::arg("dim_upper") = DefaultSampleOptions.dimUpperBound,
            pybind11::arg("maximum_tensors") = DefaultSampleOptions.maximumTensors,
            pybind11::arg("maximum_reductions") = DefaultSampleOptions.maximumReductions,
            pybind11::arg("max_flops") = static_cast<float>(DefaultSampleOptions.maxFLOPs),
            pybind11::arg("maximum_variables_in_size") = DefaultSampleOptions.maximumVariablesInSize,
            pybind11::arg("maximum_variables_powers_in_size") = DefaultSampleOptions.maximumVariablesPowersInSize,
            pybind11::arg("expression_one_tensor") = DefaultSampleOptions.expressionOneTensor,
            pybind11::arg("expression_two_tensors") = DefaultSampleOptions.expressionTwoTensors,
            pybind11::arg("expression_three_tensors") = DefaultSampleOptions.expressionThreeTensors,
            pybind11::arg("expression_four_tensors") = DefaultSampleOptions.expressionFourTensors,
            pybind11::arg("maximum_finalizations") = DefaultSampleOptions.maximumFinalizations,
            pybind11::arg("allow_weight_permutation") = DefaultSampleOptions.allowWeightPermutation,
            pybind11::arg("max_strided_dim_size") = DefaultSampleOptions.maxStridedDimSize,
            pybind11::arg("max_unfold_kernel_size") = DefaultSampleOptions.maxUnfoldKernelSize,
            pybind11::arg("minimum_unfold_ratio") = DefaultSampleOptions.minimumUnfoldRatio,
            pybind11::arg("minimum_merge_ratio") = DefaultSampleOptions.minimumMergeRatio,
            pybind11::arg("disallow_discontinuous_view") = DefaultSampleOptions.disallowDiscontinuousView,
            pybind11::arg("canonicalize_unfold_order") = DefaultSampleOptions.canonicalizeUnfoldOrder,
            pybind11::arg("disallow_split_r_above_unfold") = DefaultSampleOptions.disallowSplitRAboveUnfold,
            pybind11::arg("disallow_unfold_l_above_split") = DefaultSampleOptions.disallowUnfoldLAboveSplit,
            pybind11::arg("disallow_merge_with_large_block_above_unfold") = DefaultSampleOptions.disallowMergeWithLargeBlockAboveUnfold,
            pybind11::arg("disallow_unfold_l_above_merge_r") = DefaultSampleOptions.disallowUnfoldLAboveMergeR,
            pybind11::arg("disallow_split_r_above_stride") = DefaultSampleOptions.disallowSplitRAboveStride,
            pybind11::arg("disallow_stride_above_split") = DefaultSampleOptions.disallowStrideAboveSplit,
            pybind11::arg("disallow_merge_with_large_block_above_stride") = DefaultSampleOptions.disallowMergeWithLargeBlockAboveStride,
            pybind11::arg("disallow_stride_above_merge_r") = DefaultSampleOptions.disallowStrideAboveMergeR,
            pybind11::arg("disallow_unfold_l_above_shift") = DefaultSampleOptions.disallowUnfoldLAboveShift,
            pybind11::arg("disallow_shift_above_unfold") = DefaultSampleOptions.disallowShiftAboveUnfold,
            pybind11::arg("maximum_merges") = DefaultSampleOptions.maximumMerges,
            pybind11::arg("maximum_splits") = DefaultSampleOptions.maximumSplits,
            pybind11::arg("maximum_shifts") = DefaultSampleOptions.maximumShifts,
            pybind11::arg("maximum_strides") = DefaultSampleOptions.maximumStrides,
            pybind11::arg("maximum_unfolds") = DefaultSampleOptions.maximumUnfolds,
            pybind11::arg("maximum_shares") = DefaultSampleOptions.maximumShares
        )
        .def_readonly("seed", &SampleOptions::seed)
        .def_readonly("depth", &SampleOptions::depth)
        .def_readonly("dim_lower", &SampleOptions::dimLowerBound)
        .def_readonly("dim_upper", &SampleOptions::dimUpperBound)
        .def_readonly("maximum_tensors", &SampleOptions::maximumTensors);

    pybind11::class_<HalideGen::Options> cgOpts(m, "CodeGenOptions");
    pybind11::enum_<HalideGen::Options::AutoScheduler>(cgOpts, "AutoScheduler")
        .value("ComputeRoot", HalideGen::Options::AutoScheduler::ComputeRoot)
        .value("Mullapudi2016", HalideGen::Options::AutoScheduler::Mullapudi2016)
        .value("Li2018", HalideGen::Options::AutoScheduler::Li2018)
        .value("Adams2019", HalideGen::Options::AutoScheduler::Adams2019)
        .value("Anderson2021", HalideGen::Options::AutoScheduler::Anderson2021)
        .export_values();
    cgOpts.def(
        pybind11::init<bool, HalideGen::Options::AutoScheduler, std::size_t, float>(),
        pybind11::arg("use_gpu") = false,
        pybind11::arg("auto_scheduler") = HalideGen::Options::AutoScheduler::Li2018,
        pybind11::arg("rfactor_threshold") = 32,
        pybind11::arg("in_bounds_likely_threshold") = 0.3f
    );

    pybind11::class_<Next> next(m, "Next");
    pybind11::enum_<Next::Type>(next, "Type")
        .value("MapReduce", Next::Type::MapReduce)
        .value("Shift", Next::Type::Shift)
        .value("Stride", Next::Type::Stride)
        .value("Split", Next::Type::Split)
        .value("Unfold", Next::Type::Unfold)
        .value("Merge", Next::Type::Merge)
        .value("Share", Next::Type::Share)
        .value("Finalize", Next::Type::Finalize)
        .export_values();
    next
        .def(
            pybind11::init<Next::Type, std::size_t>(),
            pybind11::arg("type"), pybind11::arg("key")
        )
        .def_readonly_static("NumTypes", &Next::NumTypes)
        .def_readwrite("type", &Next::type)
        .def_readwrite("key", &Next::key)
        .def("__eq__", &Next::operator==)
        .def("__hash__", &Next::hash)
        .def("__repr__", &Next::toString);

    pybind11::class_<Kernel>(m, "Kernel")
        .def(
            "generate_operator", &Kernel::generateOperator,
            pybind11::arg("path"), pybind11::arg("name")
        )
        .def(
            "generate_graphviz", &Kernel::generateGraphviz,
            pybind11::arg("path"), pybind11::arg("name")
        )
        .def(
            "get_consts", &Kernel::getConsts,
            pybind11::arg("index")
        )
        .def(
            "get_flops", &Kernel::getFLOPs,
            pybind11::arg("index")
        )
        .def("get_total_flops", &Kernel::getTotalFLOPs)
        .def("get_count_inputs", &Kernel::getCountInputs)
        .def(
            "get_inputs_shapes", &Kernel::getInputsShapes,
            pybind11::arg("padded"), pybind11::arg("index")
        )
        .def(
            "get_output_shape", &Kernel::getOutputShape,
            pybind11::arg("padded"), pybind11::arg("index")
        )
        .def("__repr__", &Kernel::toNestedLoops);

    pybind11::class_<Arc>(m, "Arc")
        .def("__eq__", &Arc::operator==)
        .def("__hash__", &Arc::hash)
        .def("to_next", &Arc::toNext)
        .def("__repr__", &Arc::toString);

    pybind11::class_<Node>(m, "Node")
        .def("__eq__", &Node::operator==)
        .def("__hash__", &Node::hash)
        .def("children_count", &Node::countChildren)
        .def("get_children_handles", &Node::getChildrenHandles)
        .def("get_children_arcs", &Node::getChildrenArcs)
        .def(
            "get_arc_from_handle", &Node::getArcFromHandle,
            pybind11::arg("next")
        )
        .def(
            "get_child", &Node::getChild,
            pybind11::arg("next")
        )
        .def(
            "get_child_from_arc", &Node::getChildFromArc,
            pybind11::arg("arc")
        )
        .def("get_possible_path", &Node::getPossiblePath)
        .def("get_composing_arcs", &Node::getComposingArcs)
        .def(
            "get_child_description", &Node::getChildDescription,
            pybind11::arg("next")
        )
        .def("is_final", &Node::isFinal)
        .def("is_dead_end", &Node::isDeadEnd)
        .def("discovered_final_descendant", &Node::discoveredFinalDescendant)
        .def(
            "realize_as_final", &Node::realizeAsFinal,
            pybind11::arg("all_mappings"), pybind11::arg("halide_options")
        )
        .def("estimate_total_flops_as_final", &Node::estimateTotalFLOPsAsFinal)
        .def("generate_graphviz", &Node::generateGraphviz)
        .def("generate_graphviz_as_final", &Node::generateGraphvizAsFinal)
        .def("get_nested_loops_as_final", &Node::getNestedLoopsAsFinal)
        .def("__repr__", &Node::toString);

    pybind11::class_<Size>(m, "Size")
        .def("__eq__", &Size::operator==)
        .def("__mul__", &Size::operator*)
        .def("__truediv__", &Size::operator/);

    pybind11::class_<Forward::Dimension>(m, "ForwardDimension")
        .def(
            "output", &Forward::Dimension::output,
            pybind11::arg("index")
        )
        .def(
            "sum", [](Forward::Dimension& self, std::size_t priority) {
                self.reduce(priority, MapReduce::MapType::Identity, MapReduce::ReduceType::Sum);
            },
            pybind11::arg("priority")
        )
        .def(
            "mean", [](Forward::Dimension& self, std::size_t priority) {
                self.reduce(priority, MapReduce::MapType::Identity, MapReduce::ReduceType::Mean);
            },
            pybind11::arg("priority")
        )
        .def("size", &Forward::Dimension::sizeToString);

    pybind11::class_<Forward::Factory>(m, "Assembler")
        .def(
            "get_sizes",
            [](const Forward::Factory& self, const std::vector<std::string>& names) {
                return self.getSizes(names);
            },
            pybind11::arg("names")
        )
        .def(
            "make_dims_of_sizes",
            [](Forward::Factory& self, const std::vector<Size>& sizes) {
                return self.makeDimsOfShape(sizes);
            },
            pybind11::arg("sizes")
        )
        .def_static("create_merge", &Forward::MergeOp::Create)
        .def_static("create_share", &Forward::ShareOp::Create)
        .def_static("create_shift", &Forward::ShiftOp::Create)
        .def_static("create_split", &Forward::SplitOp::Create)
        .def_static("create_stride", &Forward::StrideOp::Create)
        .def_static("create_unfold", &Forward::UnfoldOp::Create)
        .def(
            "convert_assembled_to_path", [](Forward::Factory& self, const std::vector<std::vector<Forward::Dimension>>& tensors, const Sampler& sampler) -> std::vector<Next> {
                auto backTensors = Forward::Factory::ForwardDimsToBackwardDims(tensors);
                return sampler.convertTensorsToPath(backTensors);
            }
        )
        .def(
            "build", [](Forward::Factory& self, const std::vector<std::vector<Forward::Dimension>>& tensors, const std::string& blending, const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options) {
                TensorView& tensorView = self.buildTensorView(tensors, Parser(blending).parseTensorExpression());
                return std::make_unique<Kernel>(tensorView, self.getBindingContext(), allMappings, std::move(options));
            }
        );

    pybind11::class_<Sampler>(m, "Sampler")
        .def(
            pybind11::init<std::string, std::string, std::vector<std::string>, std::vector<std::string>, std::vector<std::map<std::string, std::size_t>>, std::vector<std::pair<std::size_t, std::size_t>>, SampleOptions>(),
            pybind11::arg("input_shape"), pybind11::arg("output_shape"), pybind11::arg("primary_specs"), pybind11::arg("coefficient_specs"), pybind11::arg("all_mappings"), pybind11::arg("fixed_io_pairs"), pybind11::arg("options"))
        .def(
            "visit", &Sampler::visit,
            pybind11::arg("path")
        )
        .def(
            "random_node_with_prefix", &Sampler::randomNodeWithPrefix,
            pybind11::arg("prefix")
        )
        .def(
            "create_assembler", [](Sampler& self) {
                return std::make_unique<Forward::Factory>(self.getBindingContext());
            }
        )
        .def(
            "bind_debug_context", [](Sampler& self) {
                BindingContext::DebugPublicCtx = &self.getBindingContext();
            }
        );
    
    pybind11::class_<StatisticsCollector>(m, "StatisticsCollector")
        .def_static(
            "PrintSummary", []() -> std::string {
                std::ostringstream ss;
                StatisticsCollector::PrintSummary(ss);
                return ss.str();
            }
        );

#ifdef VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
