#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/Sample.hpp"


PYBIND11_MODULE(kas_cpp_bindings, m) {
    m.doc() = "Python/C++ API bindings for KAS.";

    using namespace kas;

    pybind11::class_<SampleOptions>(m, "SampleOptions")
        .def(
            pybind11::init([](SampleOptions::Seed seed, std::size_t depth, std::size_t dimLowerBound, std::size_t dimUpperBound, std::size_t maximumTensors) {
                return SampleOptions {
                    .seed = seed,
                    .depth = depth,
                    .dimLowerBound = dimLowerBound,
                    .dimUpperBound = dimUpperBound,
                    .maximumTensors = maximumTensors,
                };
            }),
            pybind11::arg("seed") = 42,
            pybind11::arg("depth") = 4,
            pybind11::arg("dim_lower") = 1,
            pybind11::arg("dim_upper") = 8,
            pybind11::arg("maximum_tensors") = 2
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
            "generate", &Kernel::generate,
            pybind11::arg("path"), pybind11::arg("name"), pybind11::arg("mappings")
        )
        .def(
            "get_inputs_shapes", &Kernel::getInputsShapes,
            pybind11::arg("mappings")
        )
        .def(
            "get_output_shape", &Kernel::getOutputShape,
            pybind11::arg("mappings")
        )
        .def("__repr__", &Kernel::toNestedLoops);

    pybind11::class_<Node>(m, "Node")
        .def("children_count", &Node::countChildren)
        .def("get_children_handles", &Node::getChildrenHandles)
        .def(
            "get_child", &Node::getChild,
            pybind11::arg("next")
        )
        .def("is_final", &Node::isFinal)
        .def(
            "realize_as_final",
            [](Node& self, HalideGen::Options options) -> std::unique_ptr<Kernel> {
                auto kernel = self.asKernel();
                if (kernel == nullptr) {
                    return nullptr;
                }
                return std::make_unique<Kernel>(*kernel, self.getSampler()->getBindingContext(), std::move(options));
            },
            pybind11::arg("halide_options")
        )
        .def("__repr__", &Node::toString);

    pybind11::class_<Sampler>(m, "Sampler")
        .def(
            pybind11::init<std::string, std::string, std::vector<std::string>, std::vector<std::string>, std::vector<std::map<std::string, std::size_t>>, SampleOptions>(),
            pybind11::arg("input_shape"), pybind11::arg("output_shape"), pybind11::arg("primary_specs"), pybind11::arg("coefficient_specs"), pybind11::arg("all_mappings"), pybind11::arg("options"))
        .def(
            "visit", &Sampler::visit,
            pybind11::arg("path")
        )
        .def(
            "random_node_with_prefix", &Sampler::randomNodeWithPrefix,
            pybind11::arg("prefix")
        );

#ifdef VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
