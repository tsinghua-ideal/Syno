#include <filesystem>
#include <fstream>

#include <fmt/format.h>
#include <future>

#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"
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

Kernel::Kernel(const BindingContext& ctx, const TensorView& tensorView, const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options, const std::filesystem::path& dir, const std::string& name):
    dir { dir }
{
    if (std::filesystem::exists(dir / "metadata.json")) {
        loadMetadataAndNestedLoops();
        if (nestedLoops == tensorView.printNestedLoopsForAll(ctx)) {
            // The kernel is already generated.
            return;
        }
    }
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
        consts.padded = tensorView.computePadding(ctx, consts.unpadded);
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
        flops = tensorView.getFLOPs(consts.padded);
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
    {
        std::ofstream metadataFile { dir / "metadata.json" };
        metadata = {
            .name = name,
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

const std::string& Kernel::getNestedLoops() const {
    return nestedLoops;
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
