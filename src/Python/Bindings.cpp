#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/Search/Sample.hpp"


PYBIND11_MODULE(kas_cpp_bindings, m) {
    m.doc() = "Python/C++ API bindings for KAS.";

    using namespace kas;

    pybind11::class_<SampleOptions>(m, "SampleOptions")
        .def(pybind11::init([](SampleOptions::Seed seed, std::size_t depth, std::size_t dimLowerBound, std::size_t dimUpperBound) {
            return SampleOptions {
                .seed = seed,
                .depth = depth,
                .dimLowerBound = dimLowerBound,
                .dimUpperBound = dimUpperBound
            };
        }),
        pybind11::arg("seed") = 42,
        pybind11::arg("depth") = 4,
        pybind11::arg("dim_lower") = 1,
        pybind11::arg("dim_upper") = 8)
        .def_readonly("seed", &SampleOptions::seed)
        .def_readonly("depth", &SampleOptions::depth)
        .def_readonly("dim_lower", &SampleOptions::dimLowerBound)
        .def_readonly("dim_upper", &SampleOptions::dimUpperBound);

    pybind11::class_<HalideGen::Options> cgOpts(m, "CodeGenOptions");
    pybind11::enum_<HalideGen::Options::AutoScheduler>(cgOpts, "AutoScheduler")
        .value("ComputeRoot", HalideGen::Options::AutoScheduler::ComputeRoot)
        .value("Mullapudi2016", HalideGen::Options::AutoScheduler::Mullapudi2016)
        .value("Li2018", HalideGen::Options::AutoScheduler::Li2018)
        .value("Adams2019", HalideGen::Options::AutoScheduler::Adams2019);
    cgOpts.def(pybind11::init<bool, HalideGen::Options::AutoScheduler>());

    pybind11::class_<Kernel>(m, "Kernel")
        .def("__repr__", &Kernel::toNestedLoops)
        .def("generate", &Kernel::generate)
        .def("get_arguments", &Kernel::getArguments)
        .def("get_inputs_shapes", &Kernel::getInputsShapes)
        .def("get_output_shape", &Kernel::getOutputShape);

    pybind11::class_<Sampler>(m, "Sampler")
        .def(pybind11::init<std::string, std::string, std::vector<std::string>, std::vector<std::string>, SampleOptions>())
        .def("random_path_with_prefix", &Sampler::randomPathWithPrefix)
        .def("is_final", &Sampler::isFinal)
        .def("children_count", &Sampler::childrenCount)
        .def("children_types", &Sampler::childrenTypes)
        .def("node_str", &Sampler::nodeString)
        .def("op_str", &Sampler::opString)
        .def("op_type", &Sampler::opType)
        .def("realize", [](Sampler& self, const std::vector<std::size_t>& path) -> std::unique_ptr<Kernel> {
            auto [tensorView, cgCtx] = self.realize(path);
            return std::make_unique<Kernel>(std::move(tensorView), self.getBindingContext(), std::move(cgCtx));
        });

#ifdef VERSION_INFO
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
