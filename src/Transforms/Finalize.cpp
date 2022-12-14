#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "KAS/Core/Iterator.hpp"
#include "KAS/Transforms/Finalize.hpp"
#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Shape FinalizeShapeOp::transformShapeInverse(const Shape& incomingOutputShape) const {
    KAS_ASSERT(desired.size() == epilogue.desiredInputToGroupId.size());
    outputShape = incomingOutputShape;
    // First, the desired input constitute the frontmost dimensions.
    std::vector<std::shared_ptr<Size>> inputShape(desired.getSizes());
    // Then, directly mapped weight dimensions.
    for (std::size_t i = 0; i < epilogue.weightDirectInputToOutput.size(); ++i) {
        inputShape.push_back(outputShape[epilogue.weightDirectInputToOutput[i]]);
    }
    // Next, we would like to compute if the groups has excessive size.
    std::vector<std::shared_ptr<Size>> remainders;
    // Compute the size of the whole group.
    for (const auto& group: epilogue.outputGroups) {
        std::vector<std::shared_ptr<Size>> groupSizes;
        for (std::size_t i: group) {
            groupSizes.push_back(outputShape[i]);
        }
        remainders.push_back(Size::Product(groupSizes));
    }
    // Remove the size of the desired input.
    for (std::size_t i = 0; i < epilogue.desiredInputToGroupId.size(); ++i) {
        remainders[epilogue.desiredInputToGroupId[i]] = *remainders[epilogue.desiredInputToGroupId[i]] / *desired[i];
    }
    // Check if there is remainder
    for (std::size_t i = 0; i < remainders.size(); ++i) {
        if (!remainders[i]->is1()) {
            inputShape.push_back(remainders[i]);
            weightRemainderInputToGroupId.push_back(i);
        }
    }
    return Shape { inputShape };
}

void FinalizeShapeOp::transformTensor(TensorView& tensor) const {
    std::vector<std::shared_ptr<Iterator>> newInterface(outputShape.size(), nullptr);
    int outputCounter = 0;
    // First handle the directly mapped iterators.
    for (std::size_t i = 0; i < epilogue.weightDirectInputToOutput.size(); ++i) {
        newInterface[epilogue.weightDirectInputToOutput[i]] = tensor.interface.at(desired.size() + i);
        ++outputCounter;
    }
    std::vector<std::shared_ptr<Iterator>> groups(epilogue.outputGroups.size(), nullptr);
    // Add the remainder iterators.
    for (std::size_t i = 0; i < weightRemainderInputToGroupId.size(); ++i) {
        auto offset = desired.size() + epilogue.weightDirectInputToOutput.size() + i;
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
        }
        newInterface[0] = current;
        outputCounter += groupOutputs.size();
    }
    KAS_ASSERT(outputCounter == outputShape.size());
    tensor.interface.swap(newInterface);
}

std::vector<std::unique_ptr<FinalizeShapeOp>> FinalizeShapeOp::generate(const Shape& outputShape, GenerateOptions options) {
    // TODO
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < outputShape.size(); ++i) {
        indices.push_back(i);
    }
    std::unique_ptr<FinalizeShapeOp> op { new FinalizeShapeOp {
        options.desired,
        Epilogue {
            std::vector<std::size_t>(options.desired.size(), 0),
            {},
            { indices }
        }
    }};
    std::vector<std::unique_ptr<FinalizeShapeOp>> res;
    res.push_back(std::move(op));
    return res;
}

} // namespace kas
