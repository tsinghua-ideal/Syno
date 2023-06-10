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
            pybind11::init([](SampleOptions::Seed seed, std::size_t depth, std::size_t dimLowerBound, std::size_t dimUpperBound, std::size_t maximumTensors, std::size_t maximumReductions, float maxFLOPs, std::size_t maxStridedDimSize, std::size_t maxUnfoldKernelSize, float minimumUnfoldRatio, float minimumMergeRatio, bool disallowDiscontinuousView, bool canonicalizeUnfoldOrder, bool disallowSplitRAboveUnfold, bool disallowUnfoldLAboveSplit, bool disallowMergeWithLargeBlockAboveUnfold, bool disallowUnfoldLAboveMergeR, bool disallowSplitRAboveStride, bool disallowStrideAboveSplit, bool disallowMergeWithLargeBlockAboveStride, bool disallowStrideAboveMergeR, bool disallowUnfoldLAboveShift, bool disallowShiftAboveUnfold) {
                return SampleOptions {
                    .seed = seed,
                    .depth = depth,
                    .dimLowerBound = dimLowerBound,
                    .dimUpperBound = dimUpperBound,
                    .maximumTensors = maximumTensors,
                    .maximumReductions = maximumReductions,
                    .maxFLOPs = static_cast<std::size_t>(maxFLOPs),
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
                };
            }),
            pybind11::arg("seed") = 42,
            pybind11::arg("depth") = 4,
            pybind11::arg("dim_lower") = 1,
            pybind11::arg("dim_upper") = 8,
            pybind11::arg("maximum_tensors") = 2,
            pybind11::arg("maximum_reductions") = 2,
            pybind11::arg("max_flops") = std::numeric_limits<std::size_t>::max(),
            pybind11::arg("max_strided_dim_size") = 30,
            pybind11::arg("max_unfold_kernel_size") = 30,
            pybind11::arg("minimum_unfold_ratio") = 2.0f,
            pybind11::arg("minimum_merge_ratio") = 2.0f,
            pybind11::arg("disallow_discontinuous_view") = true,
            pybind11::arg("canonicalize_unfold_order") = true,
            pybind11::arg("disallow_split_r_above_unfold") = true,
            pybind11::arg("disallow_unfold_l_above_split") = false,
            pybind11::arg("disallow_merge_with_large_block_above_unfold") = false,
            pybind11::arg("disallow_unfold_l_above_merge_r") = false,
            pybind11::arg("disallow_split_r_above_stride") = true,
            pybind11::arg("disallow_stride_above_split") = false,
            pybind11::arg("disallow_merge_with_large_block_above_stride") = false,
            pybind11::arg("disallow_stride_above_merge_r") = false,
            pybind11::arg("disallow_unfold_l_above_shift") = true,
            pybind11::arg("disallow_shift_above_unfold") = false
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
        pybind11::init<bool, HalideGen::Options::AutoScheduler>(),
        pybind11::arg("use_gpu") = false,
        pybind11::arg("auto_scheduler") = HalideGen::Options::AutoScheduler::Li2018
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
        .def_readwrite("type", &Next::type)
        .def_readwrite("key", &Next::key)
        .def("__eq__", &Next::operator==)
        .def("__hash__", &Next::hash)
        .def(
            "description", &Next::description,
            pybind11::arg("based_on_node")
        )
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

    pybind11::class_<Node>(m, "Node")
        .def("__eq__", &Node::operator==)
        .def("__hash__", &Node::hash)
        .def("children_count", &Node::countChildren)
        .def("get_children_handles", &Node::getChildrenHandles)
        .def(
            "get_child", &Node::getChild,
            pybind11::arg("next")
        )
        .def("is_final", &Node::isFinal)
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
                self.reduce(priority, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
            },
            pybind11::arg("priority")
        )
        .def(
            "mean", [](Forward::Dimension& self, std::size_t priority) {
                self.reduce(priority, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Mean);
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
                return sampler.convertTensorViewToPath(backTensors);
            }
        )
        .def(
            "build", [](Forward::Factory& self, const std::vector<std::vector<Forward::Dimension>>& tensors, const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options) {
                TensorView& tensorView = self.buildTensorView(tensors);
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
