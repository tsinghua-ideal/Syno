#include <filesystem>
#include <fstream>
#include <future>

#include <fmt/format.h>
#include <fmt/std.h>

#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/CodeGen/PyTorchGen.hpp"
#include "KAS/CodeGen/TVMCodeGen.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

std::size_t KernelMetadata::countPlaceholders() const {
    return validPlaceholdersIndices.size();
}
std::size_t KernelMetadata::countKernels() const {
    return validPlaceholders.size();
}
const KernelMetadata::PlaceholderMetadata& KernelMetadata::getPlaceholder(std::size_t index) const {
    return validPlaceholders.at(validPlaceholdersIndices.at(index));
}

void Kernel::loadMetadataAndNestedLoops() {
    {
        std::ifstream metadataFile { dir / "metadata.json" };
        nlohmann::json metadataJson;
        metadataFile >> metadataJson;
        metadata = metadataJson.template get<KernelMetadata>();
    }
    {
        std::ifstream nestedLoopsFile { dir / "nested_loops.h" };
        nestedLoops = std::string { std::istreambuf_iterator<char>(nestedLoopsFile), std::istreambuf_iterator<char>() };
    }
}

Kernel::Kernel(const BindingContext& ctx, const TensorView& tensorView, const PyTorchSpecializedIR& pyTorchSpecializedIR, const std::vector<std::map<std::string, std::size_t>>& allMappings, CodeGenOptions options, const std::filesystem::path& dir, const std::string& name):
    dir { dir }
{
    if (std::filesystem::exists(dir / "metadata.json")) {
        loadMetadataAndNestedLoops();
        if (nestedLoops == tensorView.printNestedLoopsForAll(ctx)) {
            // The kernel is already generated.
            return;
        }
        KAS_WARNING("Overwriting existing generated kernel files: {}, due to collision!", dir);
    }
    const Graph graph = tensorView.buildGraph();
    std::vector<std::size_t> validPlaceholdersIndices;
    std::vector<KernelMetadata::PlaceholderMetadata> validPlaceholders;
    std::map<ConcreteConsts, std::size_t> constsToValidPlaceholderIndex;
    for (const auto& mappings: allMappings) {
        validPlaceholders.emplace_back();
        auto& [
            consts,
            constsDescription,
            unpaddedInputsShapes,
            paddedInputsShapes,
            unpaddedOutputShape,
            paddedOutputShape,
            flops
        ] = validPlaceholders.back();

        consts.unpadded = ctx.realizeConsts(mappings);
        auto [it, inserted] = constsToValidPlaceholderIndex.try_emplace(consts.unpadded, constsToValidPlaceholderIndex.size());
        if (!inserted) {
            // We have duplicate kernels.
            validPlaceholdersIndices.emplace_back(it->second);
            validPlaceholders.pop_back();
            continue;
        } else {
            // This is a new kernel.
            validPlaceholdersIndices.emplace_back(validPlaceholders.size() - 1);
        }
        consts.padded = tensorView.computePadding(ctx, graph, consts.unpadded);
        constsDescription = consts.toString(ctx);

        auto concretizeInputTensors = [&](const ConcreteConsts& consts) {
            return ranges::to<std::vector<std::vector<std::size_t>>>(
                tensorView.getUnderlyingTensors()
                | std::views::transform([&](const PureTensor& t) {
                    return t.getShape().eval<std::size_t>(consts);
                })
            );
        };
        unpaddedInputsShapes = concretizeInputTensors(consts.unpadded);
        paddedInputsShapes = concretizeInputTensors(consts.padded);

        auto concretizeOutputTensor = [&](const ConcreteConsts& consts) {
            return tensorView.getInterfaceShape().eval<std::size_t>(consts);
        };
        unpaddedOutputShape = concretizeOutputTensor(consts.unpadded);
        paddedOutputShape = concretizeOutputTensor(consts.padded);

        // Because we actually use the padded consts.
        flops = tensorView.getFLOPs(ctx, consts.padded);
    }

    std::filesystem::create_directories(dir);
    {
        std::ofstream nestedLoopsFile { dir / "nested_loops.h" };
        nestedLoops = tensorView.printNestedLoopsForAll(ctx);
        nestedLoopsFile << nestedLoops;
    }
    {
        GraphvizGen gen { tensorView, ctx };
        gen.generate(dir / "kernel_graph.dot", name);
    }
    {
        GraphvizDFGGen gen { tensorView.getSubgraphs(), ctx };
        gen.generate(dir / "kernel_dfg.dot", name);
    }
#ifdef KAS_USE_HALIDE
    if (options.halide) {
        std::vector<std::future<std::string>> schedules;
        const auto gen = HalideGen { ctx, tensorView, options };
        auto compileSingleKernel = [&dir, &name, &gen](std::size_t i, const ConcreteConsts& consts) -> std::string {
            std::ostringstream os;
            // Pass padded consts to HalideGen.
            const std::string filename = fmt::format("kernel_{}", i);
            gen.generate(
                dir / fmt::format("kernel_{}.o", i),
                dir / fmt::format("kernel_{}_grad.o", i),
                fmt::format("{}_{}", name, i),
                fmt::format("{}_{}_grad", name, i),
                consts,
                &os
            );
            return os.str();
        };
        for (std::size_t i = 0; auto&& placeholder: validPlaceholders) {
            schedules.emplace_back(std::async(std::launch::async, compileSingleKernel, i, placeholder.consts.padded));
            ++i;
        }
        std::ofstream halideScheduleFile { dir / "halide_schedule.h" };
        for (std::size_t i = 0; i < validPlaceholders.size(); ++i) {
            halideScheduleFile << "// Schedule for placeholder " << i << "\n";
            halideScheduleFile << schedules[i].get() << std::endl;
        }
        // Invoke the linker through command line and produce a shared library
        constexpr const char *soName = "kernels.so";
        std::vector<std::string> objects;
        auto getObjectPath = [](std::size_t i, bool grad) {
            return fmt::format("kernel_{}{}.o", i, grad ? "_grad" : "");
        };
        for (std::size_t i = 0; i < validPlaceholders.size(); ++i) {
            objects.emplace_back(getObjectPath(i, false));
            objects.emplace_back(getObjectPath(i, true));
        }
        int err = LinkObjects(dir, soName, objects);
        KAS_ASSERT(err == 0, "Failed to invoke linker, error code = {}", err);
    }
#else
    KAS_ASSERT(!options.halide, "Halide is not enabled!");
#endif
    {
        auto gen = PyTorchGen { ctx, pyTorchSpecializedIR };
        std::ofstream pytorchFile { dir / "kernels.py" };
        gen.generatePrelude(pytorchFile);
        for (std::size_t i = 0; auto&& placeholder: validPlaceholders) {
            gen.generate(pytorchFile, fmt::format("{}_{}", name, i), tensorView.getForwardAccess(), placeholder.consts);
            ++i;
        }
    }
    {
        auto gen = TVMCodeGen { ctx, tensorView.getSubgraphs() };
        std::ofstream tvmFile { dir / "kernels_tvm.py" };
        gen.generate(tvmFile);
    }
    {
        std::ofstream metadataFile { dir / "metadata.json" };
        metadata = {
            .name = name,
            .inputsShapes = tensorView.getUnderlyingTensors()
                | std::views::transform([&ctx](const PureTensor& t) { return t.getShape().toString(ctx); })
                | ranges::to<std::vector<std::string>>(),
            .outputShape = tensorView.getInterfaceShape().toString(ctx),
            .vram = PyTorchGen::EstimateVRAMUsage(ctx, pyTorchSpecializedIR),
            .halide = options.halide,
            .cuda = options.useGPU,
            .countInputs = tensorView.getUnderlyingTensors().size(),
            .validPlaceholdersIndices = std::move(validPlaceholdersIndices),
            .validPlaceholders = std::move(validPlaceholders),
        };
        nlohmann::json metadataJson = metadata;
        metadataFile << metadataJson.dump(4) << std::endl;
    }
}

Kernel::Kernel(const std::filesystem::path& dir):
    dir { dir }
{
    loadMetadataAndNestedLoops();
}

const std::string& Kernel::getName() const {
    return metadata.name;
}
const std::filesystem::path& Kernel::getDirectory() const {
    return dir;
}

const std::string& Kernel::getNestedLoops() const {
    return nestedLoops;
}

bool Kernel::halide() const {
    return metadata.halide;
}

bool Kernel::cuda() const {
    return metadata.cuda;
}
std::size_t Kernel::countPlaceholders() const {
    return metadata.countPlaceholders();
}
std::size_t Kernel::countKernels() const {
    return metadata.countKernels();
}

std::size_t Kernel::getValidPlaceholderIndex(std::size_t index) const {
    return metadata.validPlaceholdersIndices.at(index);
}

const std::string& Kernel::getConsts(std::size_t index) const {
    return metadata.getPlaceholder(index).constsDescription;
}

std::size_t Kernel::getFLOPs(std::size_t index) const {
    return metadata.getPlaceholder(index).flops;
}
std::size_t Kernel::getTotalFLOPs() const {
    std::size_t result = 0;
    for (std::size_t i = 0; i < metadata.countPlaceholders(); ++i) {
        result += metadata.getPlaceholder(i).flops;
    }
    return result;
}

std::size_t Kernel::getVRAMUsage() const {
    return metadata.vram;
}

std::size_t Kernel::getCountInputs() const {
    return metadata.countInputs;
}

const std::vector<std::vector<std::size_t>>& Kernel::getInputsShapes(bool padded, std::size_t index) const {
    const auto& placeholder = metadata.getPlaceholder(index);
    return padded ? placeholder.paddedInputsShapes : placeholder.unpaddedInputsShapes;
}

const std::vector<std::size_t>& Kernel::getOutputShape(bool padded, std::size_t index) const {
    const auto& placeholder = metadata.getPlaceholder(index);
    return padded ? placeholder.paddedOutputShape : placeholder.unpaddedOutputShape;
}

LoaderParameters Kernel::getLoaderArgs() const {
    return {
        .path = dir / "kernels.so",
        .symbol = metadata.name,
        .cuda = metadata.cuda,
        .countInputs = metadata.countInputs,
        .countKernels = metadata.countKernels(),
        .validPlaceholdersIndices = metadata.validPlaceholdersIndices,
    };
}

} // namespace kas
