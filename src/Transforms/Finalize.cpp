#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/Finalize.hpp"
#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string FinalizeShapeOp::Epilogue::toDebugString(const BindingContext& ctx, const Shape& outputShape, const Shape& desiredShape) const {
    std::stringstream ss;

    ss << "Desired input shape are mapped to the following groups:" << std::endl;
    for (std::size_t i = 0; i < desiredInputToGroupId.size(); ++i) {
        ss << "  " << desiredShape[i]->toString(ctx) << " \t-> " << desiredInputToGroupId[i] << std::endl;
    }
    std::map<std::size_t, std::size_t> outputToGroupId;
    for (std::size_t i = 0; i < outputGroups.size(); ++i) {
        for (std::size_t j: outputGroups[i]) {
            outputToGroupId[j] = i;
        }
    }
    ss << "Output shape is mapped to the following groups:" << std::endl;
    for (std::size_t i = 0; i < outputShape.size(); ++i) {
        auto gid = outputToGroupId.find(i);
        ss << "  " << outputShape[i]->toString(ctx) << " \t-> " << (gid != outputToGroupId.end() ? std::to_string(gid->second) : "unmapped") << std::endl;
    }

    return ss.str();
}

Shape FinalizeShapeOp::transformShapeInverse(const Shape& incomingOutputShape) const {
    KAS_ASSERT(desired.size() == epilogue.desiredInputToGroupId.size());
    KAS_ASSERT(outputShape == incomingOutputShape);
    KAS_ASSERT(weightRemainderInputToGroupId.empty(), "FinalizeShapeOp is supposed to fill up remainders, but it already contains {} sizes.", weightRemainderInputToGroupId.size());
    // First, the desired input constitute the frontmost dimensions.
    std::vector<std::shared_ptr<Size>> inputShape(desired.getSizes());
    // Next, we would like to compute if the groups has excessive size.
    std::vector<std::shared_ptr<Size>> remainders;
    // Compute the size of the whole group.
    for (const auto& group: epilogue.outputGroups) {
        std::vector<std::shared_ptr<Size>> groupSizes;
        for (std::size_t i: group) {
            groupSizes.emplace_back(outputShape[i]);
        }
        remainders.emplace_back(Size::Product(groupSizes));
    }
    // Remove the size of the desired input.
    for (std::size_t i = 0; i < epilogue.desiredInputToGroupId.size(); ++i) {
        remainders[epilogue.desiredInputToGroupId[i]] = *remainders[epilogue.desiredInputToGroupId[i]] / *desired[i];
    }
    // Check if there is remainder
    for (std::size_t i = 0; i < remainders.size(); ++i) {
        if (!remainders[i]->is1()) {
            inputShape.emplace_back(remainders[i]);
            weightRemainderInputToGroupId.emplace_back(i);
        }
    }
    return Shape { inputShape };
}

void FinalizeShapeOp::transformTensor(TensorView& tensor) const {
    std::vector<std::shared_ptr<Iterator>> newInterface(outputShape.size(), nullptr);
    std::vector<std::shared_ptr<Iterator>> groups(epilogue.outputGroups.size(), nullptr);
    // Add the remainder iterators.
    for (std::size_t i = 0; i < weightRemainderInputToGroupId.size(); ++i) {
        auto offset = desired.size() + i;
        groups[weightRemainderInputToGroupId[i]] = tensor.interface.at(offset);
    }
    // Merge the iterators into groups.
    for (std::size_t i = 0; i < desired.size(); ++i) {
        auto gid = epilogue.desiredInputToGroupId[i];
        if (groups[gid] == nullptr) {
            groups[gid] = tensor.interface.at(i);
        } else {
            auto current = groups[gid];
            auto next = tensor.interface.at(i);
            auto op = std::make_unique<MergeOp>(current, next);
            groups[gid] = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, *current->getSize() * *next->getSize());
        }
    }
    // Split the iterators into output.
    std::set<std::size_t> sanityCounter;
    for (std::size_t gid = 0; gid < epilogue.outputGroups.size(); ++gid) {
        const auto& groupOutputs = epilogue.outputGroups[gid];
        auto current = groups[gid];
        for (std::size_t i = groupOutputs.size() - 1; i > 0 ; --i) {
            auto outputIndex= groupOutputs[i];
            auto block = outputShape[outputIndex];
            auto op = std::make_shared<SplitOp>(current, std::weak_ptr<Iterator>(), std::weak_ptr<Iterator>());
            current = std::make_shared<Iterator>(IteratorTransform { op }, *current->getSize() / *block);
            auto extracted = std::make_shared<Iterator>(IteratorTransform { op }, block);
            op->childLhs = current;
            op->childRhs = extracted;
            newInterface[outputIndex] = extracted;
            sanityCounter.insert(outputIndex);
        }
        newInterface[groupOutputs[0]] = current;
        sanityCounter.insert(groupOutputs[0]);
    }
    KAS_ASSERT(sanityCounter.size() == outputShape.size());
    tensor.interface = std::move(newInterface);
}

std::string FinalizeShapeOp::description() const {
    std::vector<std::vector<std::size_t>> inputsOfGroups(epilogue.outputGroups.size());
    {
        std::size_t inputId = 0;
        while (inputId < desired.size()) {
            inputsOfGroups[epilogue.desiredInputToGroupId[inputId]].emplace_back(inputId);
            ++inputId;
        }
        std::size_t inputShapeSize = desired.size() + weightRemainderInputToGroupId.size();
        while (inputId < inputShapeSize) {
            inputsOfGroups[weightRemainderInputToGroupId[inputId - desired.size()]].emplace_back(inputId);
            ++inputId;
        }
    }

    std::stringstream ss;
    ss << "Finalize ";
    bool firstGroup = true;
    for (std::size_t gid = 0; gid < epilogue.outputGroups.size(); ++gid) {
        const auto& inputGroup = inputsOfGroups[gid];
        const auto& outputGroup = epilogue.outputGroups[gid];
        if (firstGroup) {
            firstGroup = false;
        } else {
            ss << ", ";
        }
        ss << "{ ";
        bool first = true;
        for (std::size_t i: inputGroup) {
            if (first) {
                first = false;
            } else {
                ss << ", ";
            }
            ss << static_cast<int>(i);
        }
        ss << " -> ";
        first = true;
        for (std::size_t i: outputGroup) {
            if (first) {
                first = false;
            } else {
                ss << ", ";
            }
            ss << static_cast<int>(i);
        }
        ss << " }";
    }
    return ss.str();
}

bool FinalizeShapeOp::isFinalizeOp() const {
    return true;
}

const FinalizeShapeOp::Epilogue& FinalizeShapeOp::getEpilogue() const {
    return epilogue;
}

// Input the mappings from desired dimensions to output dimensions, and output the Epilogue.
std::optional<FinalizeShapeOp::Epilogue> FinalizeShapeOp::solveWithMappings(const Shape& outputShape, const Shape& desiredShape, const std::vector<std::size_t>& mappings) {
    const auto& outputShapeSizes = outputShape.getSizes();

    std::vector<GroupedDim> desiredGroups;
    std::vector<GroupedDim> coefficientDimsGroups;
    std::vector<GroupedDim> generalDimsGroups;

    std::map<std::size_t, std::set<std::size_t>> desiredIndices;
    for (std::size_t i = 0; i < mappings.size(); ++i) {
        desiredIndices[mappings[i]].insert(i);
    }
    for (std::size_t i = 0; i < outputShapeSizes.size(); ++i) {
        LabeledSize dim(*outputShapeSizes[i]);
        auto it = desiredIndices.find(i);
        if (it != desiredIndices.end()) {
            GroupedDim g { dim, { i } };
            for (auto j: it->second) {
                g.dividedBy(*desiredShape[j]);
            }
            desiredGroups.emplace_back(std::move(g));
        } else {
            switch (dim.trait) {
            case Size::Trait::One:
            case Size::Trait::IllegalCoefficient:
                KAS_CRITICAL("Unexpected dimension size: one or illegal coefficient.");
                break;
            case Size::Trait::Coefficient:
                coefficientDimsGroups.emplace_back(std::move(dim), std::set<std::size_t> { i });
                break;
            case Size::Trait::General:
                generalDimsGroups.emplace_back(std::move(dim), std::set<std::size_t> { i });
                break;
            }
        }
    }

    const GroupedDim groupIdentity = desiredGroups[0].identity();

    // Now that we have extracted desired dimensions, try to merge the groups.
    // Only desiredGroups may contain illegal dimensions, i.e., IllegalCoefficient.
    // In this case, we can merge it with any General dimension to make it legal. But we should prioritize merging with coefficient dimensions, especially the coefficient dimensions in desired dimensions.
    for (std::size_t current = 0; current < desiredGroups.size();) {
        auto& cdg = desiredGroups[current].size;

        // If current is legal, just jump.
        if (!cdg.isIllegalCoefficient()) {
            ++current;
            continue;
        }
        // Otherwise, we should merge groups.

        // The first step is to try with our best to merge with coefficient dimensions.
        std::set<std::size_t> mergesDesired { current };
        auto merged = cdg;

        // First look for coefficient dimensions within desired dimensions.
        for (std::size_t i = 0; i < desiredGroups.size();) {
            auto& dg = desiredGroups[i].size;
            if (!mergesDesired.contains(i) && dg.isIndeterminedCoefficient()) {
                // Try to absorb.
                auto absorb = merged.absorbCoefficientNumeratorToDenominator(dg);
                if (absorb.has_value()) {
                    merged = std::move(absorb.value());
                    mergesDesired.insert(i);
                    if (!merged.isIllegalCoefficient()) {
                        // If the merged dimension is legal, we can stop here.
                        break;
                    }
                    // Otherwise retry merging from start.
                    i = 0;
                    continue;
                }
            }
            ++i;
        }

        // If this is still not enough, look for sizes in coefficient dimensions.
        std::set<std::size_t> mergesCoefficient;
        if (merged.isIllegalCoefficient()) {
            for (std::size_t i = 0; i < coefficientDimsGroups.size();) {
                if (mergesCoefficient.contains(i)) {
                    ++i;
                    continue;
                }
                auto& cg = coefficientDimsGroups[i].size;
                auto absorb = merged.absorbCoefficientNumeratorToDenominator(cg);
                if (absorb.has_value()) {
                    merged = std::move(absorb.value());
                    mergesCoefficient.insert(i);
                    if (!merged.isIllegalCoefficient()) {
                        // If the merged dimension is legal, we can stop here.
                        break;
                    }
                    // Otherwise retry merging from start.
                    i = 0;
                    continue;
                }
                ++i;
            }
        }

        // If these just merging coefficient dimensions suffice, really merge them.
        if (!merged.isIllegalCoefficient()) {
            auto realMerged1 = std::accumulate(
                mergesDesired.begin(), mergesDesired.end(),
                groupIdentity,
                [&desiredGroups](GroupedDim&& a, std::size_t b) -> GroupedDim {
                    a.addGroup(std::move(desiredGroups[b]));
                    return std::move(a);
                }
            );
            auto realMerged = std::accumulate(
                mergesCoefficient.begin(), mergesCoefficient.end(),
                std::move(realMerged1),
                [&coefficientDimsGroups](GroupedDim&& a, std::size_t b) -> GroupedDim {
                    a.addGroup(std::move(coefficientDimsGroups[b]));
                    return std::move(a);
                }
            );

            std::vector<GroupedDim> newDesiredGroups;
            std::size_t newDesiredGroupsSize = desiredGroups.size() - mergesDesired.size() + 1;
            newDesiredGroups.reserve(newDesiredGroupsSize);
            auto itD = mergesDesired.begin();
            std::size_t mergedTo = *(itD++);
            std::size_t backward = 0;
            std::size_t sanityCounter = 0;
            for (std::size_t i = 0; i < desiredGroups.size(); ++i) {
                if (i == mergedTo) {
                    newDesiredGroups.emplace_back(std::move(realMerged));
                    ++sanityCounter;
                } else if (itD != mergesDesired.end() && i == *itD) {
                    if (*itD <= current) {
                        ++backward;
                    }
                    ++itD;
                } else {
                    newDesiredGroups.emplace_back(std::move(desiredGroups[i]));
                    ++sanityCounter;
                }
            }
            KAS_ASSERT(sanityCounter == newDesiredGroupsSize);
            desiredGroups = std::move(newDesiredGroups);

            std::vector<GroupedDim> newCoefficientDimsGroups;
            std::size_t newCoefficientDimsGroupsSize = coefficientDimsGroups.size() - mergesCoefficient.size();
            newCoefficientDimsGroups.reserve(newCoefficientDimsGroupsSize);
            auto itC = mergesCoefficient.begin();
            sanityCounter = 0;
            for (std::size_t i = 0; i < coefficientDimsGroups.size(); ++i) {
                if (itC != mergesCoefficient.end() && i == *itC) {
                    ++itC;
                } else {
                    newCoefficientDimsGroups.emplace_back(std::move(coefficientDimsGroups[i]));
                    ++sanityCounter;
                }
            }
            KAS_ASSERT(sanityCounter == newCoefficientDimsGroupsSize);
            coefficientDimsGroups = std::move(newCoefficientDimsGroups);

            current = current - backward + 1;
            continue;
        }

        // If merging coefficient dimensions does not suffice, we pick one general dimension.
        std::optional<std::pair<std::size_t, int>> bestIndexDesired = std::nullopt;
        for (std::size_t i = 0; i < desiredGroups.size(); ++i) {
            auto& dg = desiredGroups[i].size;
            if (i != current && dg.isGeneral()) {
                int score = cdg.scoreOfGeneralDimension(dg);
                if (!bestIndexDesired.has_value() || bestIndexDesired.value().second <= score) {
                    bestIndexDesired = std::make_pair(i, score);
                }
            }
        }
        std::optional<std::pair<std::size_t, int>> bestIndexGeneral = std::nullopt;
        for (std::size_t i = 0; i < generalDimsGroups.size(); ++i) {
            auto& gg = generalDimsGroups[i].size;
            int score = cdg.scoreOfGeneralDimension(gg);
            if (!bestIndexGeneral.has_value() || bestIndexGeneral.value().second <= score) {
                bestIndexGeneral = std::make_pair(i, score);
            }
        }
        // If we cannot find any, then this is impossible.
        if (!bestIndexDesired.has_value() && !bestIndexGeneral.has_value()) {
            return std::nullopt;
        }
        // Otherwise choose the better one.
        const auto invalidPair = std::make_pair(std::numeric_limits<std::size_t>::max(), -1);
        if (bestIndexDesired.value_or(invalidPair).second > bestIndexGeneral.value_or(invalidPair).second) {
            std::size_t other = bestIndexDesired.value().first;
            if (other < current) {
                --current; // Go backward one step.
            }
            auto [to, fro] = std::minmax(current, other);
            desiredGroups[to].addGroup(std::move(desiredGroups[fro]));
            desiredGroups.erase(desiredGroups.begin() + fro);
        } else {
            std::size_t other = bestIndexGeneral.value().first;
            desiredGroups[current].addGroup(std::move(generalDimsGroups[other]));
            generalDimsGroups.erase(generalDimsGroups.begin() + other);
        }
        // Continue.
        ++current;
    }

    // Convert the results to Epilogue.
    std::vector<std::vector<std::size_t>> groups;
    auto put = [&groups](const std::set<std::size_t>& g) {
        groups.emplace_back(std::vector<std::size_t>(g.begin(), g.end()));
    };
    for (auto& g: desiredGroups) put(g.indices);
    for (auto& g: coefficientDimsGroups) put(g.indices);
    for (auto& g: generalDimsGroups) put(g.indices);
    std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
        return a.at(0) < b.at(0);
    });
    std::vector<std::size_t> desiredDimToGroupId;
    desiredDimToGroupId.reserve(mappings.size());
    for (std::size_t i = 0; i < mappings.size(); ++i) {
        std::size_t target = mappings[i];
        for (std::size_t j = 0; j < groups.size(); ++j) {
            if (std::binary_search(groups[j].begin(), groups[j].end(), target)) {
                desiredDimToGroupId.emplace_back(j);
                break;
            }
        }
    }
    KAS_ASSERT(desiredDimToGroupId.size() == mappings.size());
    return Epilogue {
        std::move(desiredDimToGroupId),
        std::move(groups),
    };
}

std::vector<std::unique_ptr<FinalizeShapeOp>> FinalizeShapeOp::generate(const Shape& outputShape, GenerateOptions options) {
    const auto& desired = options.desired;
    std::set<std::size_t> hashes;
    std::vector<std::unique_ptr<FinalizeShapeOp>> result;
    const auto determine = [&](const auto& self, const std::vector<Size>& sizes, const std::vector<std::size_t>& mappings) -> void {
        if (mappings.size() == desired.size()) {
            auto epilogue = solveWithMappings(outputShape, desired, mappings);
            if (epilogue.has_value()) {
                std::size_t hash = std::hash<Epilogue>()(epilogue.value());
                if (hashes.find(hash) == hashes.end()) {
                    hashes.emplace(hash);
                    result.emplace_back(std::make_unique<FinalizeShapeOp>(
                        outputShape,
                        desired,
                        std::move(epilogue.value())
                    ));
                }
            }
        } else {
            std::size_t next = mappings.size();
            for (std::size_t i = 0; i < sizes.size(); ++i) {
                Size newSize = sizes[i];
                if (newSize.testDividedBy(*desired[next]).has_value()) {
                    std::vector<Size> newSizes(sizes);
                    newSizes[i] = newSize;
                    std::vector<std::size_t> newMappings(mappings);
                    newMappings.emplace_back(i);
                    self(self, newSizes, newMappings);
                }
            }
        }
    };
    std::vector<Size> outputSizes;
    for (const auto& s: outputShape.getSizes()) {
        outputSizes.emplace_back(*s);
    }
    determine(determine, outputSizes, {});
    return result;
}

} // namespace kas
