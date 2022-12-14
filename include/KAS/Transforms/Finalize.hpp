#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

// [...desired_input, ...weight_direct, ...weight_group_remainder] -> [direct output and groups interleaved]
class FinalizeShapeOp final: public PrimitiveShapeOp {
public:
    struct Epilogue {
        std::vector<std::size_t> desiredInputToGroupId;
        std::vector<std::size_t> weightDirectInputToOutput;
        std::vector<std::vector<std::size_t>> outputGroups;
    };
protected:
    const Shape& desired;
    Epilogue epilogue;
    // If the size of an outputGroup is excessive, it is automatically turned into weight dimension.
    mutable std::vector<std::size_t> weightRemainderInputToGroupId;
    mutable Shape outputShape;
public:
    template<typename T>
    FinalizeShapeOp(const Shape& desired, T&& epilogue):
        desired { desired },
        epilogue { std::forward<T>(epilogue) }
    {}
    Shape transformShapeInverse(const Shape& outputShape) const override;
    // Here, we can assert that the interface are all TensorStub.
    void transformTensor(TensorView& tensor) const override;

    struct GenerateOptions {
        const Shape& desired;
    };
    static std::vector<std::unique_ptr<FinalizeShapeOp>> generate(const Shape& outputShape, GenerateOptions options);
};

} // namespace kas
