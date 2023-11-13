#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#ifdef KAS_USE_HALIDE
#include "KAS/CodeGen/HalideGen.hpp"
#endif
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
            pybind11::init([](SampleOptions::Seed seed, std::size_t depth, std::size_t maxChainLength, std::size_t maximumTensors, std::size_t maximumReductions, float maxFLOPs, float minFLOPs, std::size_t maxRDomSizeMultiplier, bool enableFLOPsBasedPruning, std::size_t maximumEnumerationsPerVar, std::size_t maximumVariablesInSize, std::size_t maximumVariablesPowersInSize, bool requiresExactDivision, bool requiresOddKernelSizeInUnfold, bool countCoefficientsInWeightsAsAllowanceUsage, std::string expressionOneTensor, std::string expressionTwoTensors, std::string expressionThreeTensors, std::string expressionFourTensors, std::size_t maximumFinalizations, bool allowWeightPermutation, std::size_t minSingleWeightParams, std::size_t maxStridedDimSize, std::size_t maxUnfoldKernelSize, float minimumUnfoldRatio, float maximumValidReshapeShiftPattern, bool disallowMergeInputAndWeight, bool disallowTile, bool disallowShareWeights, std::size_t maxExpansionRepeatMultiplier, std::size_t maxExpansionMergeMultiplier, std::size_t maxExpansionWeightsSharingDimSize, std::size_t minExpansionWeightsSharingDimSize, bool canonicalizeUnfoldOrder, bool disallowSplitLAboveUnfold, bool disallowSplitRAboveUnfold, bool disallowUnfoldLAboveSplit, bool disallowMergeWithLargeBlockAboveUnfold, bool disallowUnfoldLAboveMergeR, bool disallowSplitRAboveStride, bool disallowStrideAboveSplit, bool disallowMergeWithLargeBlockAboveStride, bool disallowStrideAboveMergeR, bool disallowUnfoldLAboveShift, bool disallowShiftAboveUnfold, int maximumExpands, int maximumMerges, int maximumSplits, int maximumShifts, int maximumStrides, int maximumUnfolds, int maximumShares) {
                return SampleOptions {
                    .seed = seed,
                    .depth = depth,
                    .maxChainLength = maxChainLength,
                    .maximumTensors = maximumTensors,
                    .maximumReductions = maximumReductions,
                    .maxFLOPs = static_cast<std::size_t>(maxFLOPs),
                    .minFLOPs = static_cast<std::size_t>(minFLOPs),
                    .maxRDomSizeMultiplier = maxRDomSizeMultiplier,
                    .enableFLOPsBasedPruning = enableFLOPsBasedPruning,
                    .maximumEnumerationsPerVar = maximumEnumerationsPerVar,
                    .maximumVariablesInSize = maximumVariablesInSize,
                    .maximumVariablesPowersInSize = maximumVariablesPowersInSize,
                    .requiresExactDivision = requiresExactDivision,
                    .requiresOddKernelSizeInUnfold = requiresOddKernelSizeInUnfold,
                    .countCoefficientsInWeightsAsAllowanceUsage = countCoefficientsInWeightsAsAllowanceUsage,
                    .expressionOneTensor = expressionOneTensor,
                    .expressionTwoTensors = expressionTwoTensors,
                    .expressionThreeTensors = expressionThreeTensors,
                    .expressionFourTensors = expressionFourTensors,
                    .maximumFinalizations = maximumFinalizations,
                    .allowWeightPermutation = allowWeightPermutation,
                    .minSingleWeightParams = minSingleWeightParams,
                    .maxStridedDimSize = maxStridedDimSize,
                    .maxUnfoldKernelSize = maxUnfoldKernelSize,
                    .minimumUnfoldRatio = minimumUnfoldRatio,
                    .maximumValidReshapeShiftPattern = maximumValidReshapeShiftPattern,
                    .disallowMergeInputAndWeight = disallowMergeInputAndWeight,
                    .disallowTile = disallowTile,
                    .disallowShareWeights = disallowShareWeights,
                    .maxExpansionRepeatMultiplier = maxExpansionRepeatMultiplier,
                    .maxExpansionMergeMultiplier = maxExpansionMergeMultiplier,
                    .maxExpansionWeightsSharingDimSize = maxExpansionWeightsSharingDimSize,
                    .minExpansionWeightsSharingDimSize = minExpansionWeightsSharingDimSize,
                    .canonicalizeUnfoldOrder = canonicalizeUnfoldOrder,
                    .disallowSplitLAboveUnfold = disallowSplitLAboveUnfold,
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
                    .maximumExpands = maximumExpands,
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
            pybind11::arg("max_chain_length") = DefaultSampleOptions.maxChainLength,
            pybind11::arg("maximum_tensors") = DefaultSampleOptions.maximumTensors,
            pybind11::arg("maximum_reductions") = DefaultSampleOptions.maximumReductions,
            pybind11::arg("max_flops") = static_cast<float>(DefaultSampleOptions.maxFLOPs),
            pybind11::arg("min_flops") = static_cast<float>(DefaultSampleOptions.minFLOPs),
            pybind11::arg("max_rdom_size_multiplier") = DefaultSampleOptions.maxRDomSizeMultiplier,
            pybind11::arg("enable_flops_based_pruning") = DefaultSampleOptions.enableFLOPsBasedPruning,
            pybind11::arg("maximum_enumerations_per_var") = DefaultSampleOptions.maximumEnumerationsPerVar,
            pybind11::arg("maximum_variables_in_size") = DefaultSampleOptions.maximumVariablesInSize,
            pybind11::arg("maximum_variables_powers_in_size") = DefaultSampleOptions.maximumVariablesPowersInSize,
            pybind11::arg("requires_exact_division") = DefaultSampleOptions.requiresExactDivision,
            pybind11::arg("requires_odd_kernel_size_in_unfold") = DefaultSampleOptions.requiresOddKernelSizeInUnfold,
            pybind11::arg("count_coefficients_in_weights_as_allowance_usage") = DefaultSampleOptions.countCoefficientsInWeightsAsAllowanceUsage,
            pybind11::arg("expression_one_tensor") = DefaultSampleOptions.expressionOneTensor,
            pybind11::arg("expression_two_tensors") = DefaultSampleOptions.expressionTwoTensors,
            pybind11::arg("expression_three_tensors") = DefaultSampleOptions.expressionThreeTensors,
            pybind11::arg("expression_four_tensors") = DefaultSampleOptions.expressionFourTensors,
            pybind11::arg("maximum_finalizations") = DefaultSampleOptions.maximumFinalizations,
            pybind11::arg("allow_weight_permutation") = DefaultSampleOptions.allowWeightPermutation,
            pybind11::arg("min_single_weight_params") = DefaultSampleOptions.minSingleWeightParams,
            pybind11::arg("max_strided_dim_size") = DefaultSampleOptions.maxStridedDimSize,
            pybind11::arg("max_unfold_kernel_size") = DefaultSampleOptions.maxUnfoldKernelSize,
            pybind11::arg("minimum_unfold_ratio") = DefaultSampleOptions.minimumUnfoldRatio,
            pybind11::arg("maximum_valid_reshape_shift_pattern") = DefaultSampleOptions.maximumValidReshapeShiftPattern,
            pybind11::arg("disallow_merge_input_and_weight") = DefaultSampleOptions.disallowMergeInputAndWeight,
            pybind11::arg("disallow_tile") = DefaultSampleOptions.disallowTile,
            pybind11::arg("disallow_share_weights") = DefaultSampleOptions.disallowShareWeights,
            pybind11::arg("max_expansion_repeat_multiplier") = DefaultSampleOptions.maxExpansionRepeatMultiplier,
            pybind11::arg("max_expansion_merge_multiplier") = DefaultSampleOptions.maxExpansionMergeMultiplier,
            pybind11::arg("max_expansion_weights_sharing_dim_size") = DefaultSampleOptions.maxExpansionWeightsSharingDimSize,
            pybind11::arg("min_expansion_weights_sharing_dim_size") = DefaultSampleOptions.minExpansionWeightsSharingDimSize,
            pybind11::arg("canonicalize_unfold_order") = DefaultSampleOptions.canonicalizeUnfoldOrder,
            pybind11::arg("disallow_split_l_above_unfold") = DefaultSampleOptions.disallowSplitLAboveUnfold,
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
            pybind11::arg("maximum_expands") = DefaultSampleOptions.maximumExpands,
            pybind11::arg("maximum_merges") = DefaultSampleOptions.maximumMerges,
            pybind11::arg("maximum_splits") = DefaultSampleOptions.maximumSplits,
            pybind11::arg("maximum_shifts") = DefaultSampleOptions.maximumShifts,
            pybind11::arg("maximum_strides") = DefaultSampleOptions.maximumStrides,
            pybind11::arg("maximum_unfolds") = DefaultSampleOptions.maximumUnfolds,
            pybind11::arg("maximum_shares") = DefaultSampleOptions.maximumShares
        )
        .def_readonly("seed", &SampleOptions::seed)
        .def_readonly("depth", &SampleOptions::depth)
        .def_readonly("maximum_tensors", &SampleOptions::maximumTensors);

    pybind11::class_<CodeGenOptions> cgOpts(m, "CodeGenOptions");
    pybind11::enum_<CodeGenOptions::AutoScheduler>(cgOpts, "AutoScheduler")
        .value("ComputeRoot", CodeGenOptions::AutoScheduler::ComputeRoot)
        .value("Mullapudi2016", CodeGenOptions::AutoScheduler::Mullapudi2016)
        .value("Li2018", CodeGenOptions::AutoScheduler::Li2018)
        .value("Adams2019", CodeGenOptions::AutoScheduler::Adams2019)
        .value("Anderson2021", CodeGenOptions::AutoScheduler::Anderson2021)
        .export_values();
    cgOpts.def(
        pybind11::init<bool, bool, CodeGenOptions::AutoScheduler, std::map<std::string, std::string>, std::size_t, float>(),
        pybind11::arg("halide") = false,
        pybind11::arg("use_gpu") = false,
        pybind11::arg("auto_scheduler") = CodeGenOptions::AutoScheduler::Li2018,
        pybind11::arg("extra_options") = std::map<std::string, std::string>(),
        pybind11::arg("rfactor_threshold") = 32,
        pybind11::arg("in_bounds_likely_threshold") = 0.3f
    );

    pybind11::class_<Next> next(m, "Next");
    pybind11::enum_<Next::Type>(next, "Type")
        .value("Reduce", Next::Type::Reduce)
        .value("Expand", Next::Type::Expand)
        .value("Shift", Next::Type::Shift)
        .value("Stride", Next::Type::Stride)
        .value("Split", Next::Type::Split)
        .value("Unfold", Next::Type::Unfold)
        .value("Merge", Next::Type::Merge)
        .value("Contraction", Next::Type::Contraction)
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
        .def("to_next", [](const Next& self) -> Next { return self; })
        .def("__repr__", &Next::toString);

    pybind11::class_<LoaderParameters>(m, "LoaderParameters")
        .def_readonly("path", &LoaderParameters::path)
        .def_readonly("symbol", &LoaderParameters::symbol)
        .def_readonly("cuda", &LoaderParameters::cuda)
        .def_readonly("count_inputs", &LoaderParameters::countInputs)
        .def_readonly("count_kernels", &LoaderParameters::countKernels)
        .def_readonly("valid_placeholder_indices", &LoaderParameters::validPlaceholdersIndices);

    pybind11::class_<Kernel>(m, "Kernel")
        .def(pybind11::init<std::string>()) // From directory.
        .def("get_name", &Kernel::getName)
        .def("get_directory", &Kernel::getDirectory)
        .def("halide", &Kernel::halide)
        .def("use_cuda", &Kernel::cuda)
        .def("get_count_placeholders", &Kernel::countPlaceholders)
        .def("get_count_valid_kernels", &Kernel::countKernels)
        .def("get_valid_placeholder_index", &Kernel::getValidPlaceholderIndex)
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
        .def("get_loader_args", &Kernel::getLoaderArgs)
        .def("__repr__", &Kernel::getNestedLoops);

    pybind11::class_<Arc>(m, "Arc")
        .def("__eq__", &Arc::operator==)
        .def("__hash__", &Arc::hash)
        .def("to_next", &Arc::toNext)
        .def("__repr__", &Arc::toString);

    pybind11::class_<ShapeDistance>(m, "ShapeDistance")
        .def_readwrite("steps", &ShapeDistance::steps)
        .def_readwrite("flops", &ShapeDistance::flops)
        .def("__repr__", &ShapeDistance::toString);

    pybind11::class_<Node>(m, "Node")
        .def("__eq__", &Node::operator==)
        .def("__hash__", &Node::hash)
        .def("arbitrary_parent", &Node::arbitraryParent)
        .def("recompute_shape_distance", &Node::recomputeShapeDistance)
        .def("get_shape_distance", &Node::getShapeDistance)
        .def("depth", &Node::depth)
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
            "get_children", &Node::getChildren,
            pybind11::arg("nexts")
        )
        .def(
            "get_child_from_arc", &Node::getChildFromArc,
            pybind11::arg("arc")
        )
        .def(
            "get_children_from_arcs", &Node::getChildrenFromArcs,
            pybind11::arg("arcs")
        )
        .def("get_possible_path", &Node::getPossiblePath)
        .def("get_composing_arcs", &Node::getComposingArcs)
        .def(
            "expand", &Node::expandSync,
            pybind11::arg("layers")
        )
        .def(
            "expand_to", &Node::expandToSync,
            pybind11::arg("target")
        )
        .def(
            "expand_async", &Node::expand,
            pybind11::arg("layers")
        )
        .def(
            "get_child_description", &Node::getChildDescription,
            pybind11::arg("next")
        )
        .def("is_final", &Node::isFinal)
        .def("is_dead_end", &Node::isDeadEnd)
        .def("discovered_final_descendant", &Node::discoveredFinalDescendant)
        .def(
            "realize_as_final", &Node::realizeAsFinal,
            pybind11::arg("all_mappings"), pybind11::arg("halide_options"), pybind11::arg("directory"), pybind11::arg("name")
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
            "sum", [](Forward::Dimension& self) {
                self.reduce(Reduce::ReduceType::Sum);
            }
        )
        .def(
            "mean", [](Forward::Dimension& self) {
                self.reduce(Reduce::ReduceType::Mean);
            }
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
        .def(
            "create_expand",
            [](Forward::Factory& self, const Size& size) {
                return Forward::ExpandOp::Create(self, size);
            },
            pybind11::arg("size")
        )
        .def_static("create_merge", &Forward::MergeOp::Create)
        .def_static("create_share", &Forward::ShareOp::Create)
        .def_static("create_shift", &Forward::ShiftOp::Create)
        .def_static("create_split", &Forward::SplitOp::Create)
        .def_static("create_stride", &Forward::StrideOp::Create)
        .def_static("create_unfold", &Forward::UnfoldOp::Create)
        .def(
            "inputs", &Forward::Factory::inputs,
            pybind11::arg("tensors")
        )
        .def(
            "convert_assembled_to_path", [](Forward::Factory& self, const Sampler& sampler) -> std::vector<Next> {
                auto backTensors = self.getInputs();
                sampler.convertTensorsToSearchableForm(backTensors);
                return Sampler::ConvertSearchableTensorsToPath(backTensors, self.getStore());
            }
        )
        .def(
            "build", [](Forward::Factory& self, const std::string& blending, const std::vector<std::map<std::string, std::size_t>>& allMappings, CodeGenOptions options, const std::filesystem::path& dir, const std::string& name) {
                TensorView& tensorView = self.buildTensorView(Parser(blending).parseTensorExpression());
                return std::make_unique<Kernel>(self.getBindingContext(), tensorView, allMappings, std::move(options), dir, name);
            }
        );

    pybind11::class_<Sampler>(m, "Sampler")
        .def(
            pybind11::init<std::string, std::string, std::vector<std::string>, std::vector<std::string>, std::vector<std::map<std::string, std::size_t>>, std::vector<std::pair<std::size_t, std::size_t>>, SampleOptions, std::size_t>(),
            pybind11::arg("input_shape"), pybind11::arg("output_shape"), pybind11::arg("primary_specs"), pybind11::arg("coefficient_specs"), pybind11::arg("all_mappings"), pybind11::arg("fixed_io_pairs"), pybind11::arg("options"), pybind11::arg("num_worker_threads") = 1)
        .def("get_all_stats", &Sampler::statsToString)
        .def(
            "visit", &Sampler::visit,
            pybind11::arg("path")
        )
        .def(
            "random_node_with_prefix", &Sampler::randomNodeWithPrefix,
            pybind11::arg("prefix")
        )
        .def(
            "random_final_nodes_with_prefix", &Sampler::randomFinalNodesWithPrefix,
            pybind11::arg("prefix"), pybind11::arg("count"), pybind11::arg("type"), pybind11::arg("steps")
        )
        .def(
            "create_assembler", [](Sampler& self) {
                return std::make_unique<Forward::Factory>(self.getBindingContext());
            }
        )
        .def(
            "bind_debug_context", [](Sampler& self) {
                self.getBindingContext().debug();
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
