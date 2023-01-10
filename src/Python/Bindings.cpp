#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {
    // This is for convenience. As a python interface, we need easy access to related methods of TensorView.
    class Kernel {
    protected:
        TensorView tensorView;
        BindingContext& ctx;
        std::shared_ptr<CodeGenContext> cgCtx;
        HalideGen gen;
    public:
        template<typename T>
        Kernel(T&& tensorView, BindingContext& ctx, std::shared_ptr<CodeGenContext> cgCtx):
            tensorView { std::forward<T>(tensorView) },
            ctx { ctx },
            cgCtx { std::move(cgCtx) },
            gen { ctx, this->tensorView }
        {}
        std::string toNestedLoops() const {
            return tensorView.printNestedLoops(ctx);
        }
        void generate(const std::string& path, const std::string& name, HalideGen::Options options) {
            gen.generate(path, name, options);
        }
    };
} // namespace kas

PYBIND11_MODULE(kas_cpp_bindings, m) {
    m.doc() = "Python/C++ API bindings for KAS.";

    using namespace kas;

    pybind11::class_<SampleOptions>(m, "SampleOptions")
        .def(pybind11::init<>())
        .def_readwrite("countPrimaryVariables", &SampleOptions::countPrimaryVariables)
        .def_readwrite("countCoefficientVariables", &SampleOptions::countCoefficientVariables)
        .def_readwrite("depth", &SampleOptions::depth)
        .def_readwrite("dimLowerBound", &SampleOptions::dimLowerBound)
        .def_readwrite("dimUpperBound", &SampleOptions::dimUpperBound);

    pybind11::class_<HalideGen::Options> cgOpts(m, "CodeGenOptions");
    pybind11::enum_<HalideGen::Options::AutoScheduler>(cgOpts, "AutoScheduler")
        .value("Mullapudi2016", HalideGen::Options::AutoScheduler::Mullapudi2016)
        .value("Li2018", HalideGen::Options::AutoScheduler::Li2018)
        .value("Adams2019", HalideGen::Options::AutoScheduler::Adams2019);
    cgOpts.def(pybind11::init<bool, HalideGen::Options::AutoScheduler>());

    pybind11::class_<Kernel>(m, "Kernel")
        .def("__repr__", &Kernel::toNestedLoops)
        .def("generate", &Kernel::generate);

    pybind11::class_<Sampler>(m, "Sampler")
        .def(pybind11::init<std::string, std::string, SampleOptions>())
        .def("isFinal", &Sampler::isFinal)
        .def("countChildren", &Sampler::countChildren)
        .def("realize", [](Sampler& self, std::vector<std::size_t> path) -> std::unique_ptr<Kernel> {
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
