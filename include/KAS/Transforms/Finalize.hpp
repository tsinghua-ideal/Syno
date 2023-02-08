#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

// [...desired_input, ...weight_group_remainder] -> [direct output and groups interleaved]
class FinalizeShapeOp final: public PrimitiveShapeOp {
public:
    // Describes how to remap the desired input shape to the generated final level shape. This is done by grouping the dimensions, and splitting the desired size from the groups.
    struct Epilogue {
        // Maps the desired input shape to groups
        std::vector<std::size_t> desiredInputToGroupId;
        // The groups.
        std::vector<std::vector<std::size_t>> outputGroups;

        std::string toDebugString(const BindingContext& ctx, const Shape& outputShape, const Shape& desiredShape) const;
    };
protected:
    // Represents a group of dimensions.
    struct GroupedDim {
        LabeledSize size;
        std::set<std::size_t> indices;
        inline void dividedBy(const Size& other) {
            bool success = size.testDividedBy(other);
            KAS_ASSERT(success);
        }
        inline void addGroup(const GroupedDim& other) {
            indices.insert(other.indices.begin(), other.indices.end());
            size *= other.size;
        }
        inline GroupedDim identity() const {
            return { size.identity(), {} };
        }
    };

    const Shape& outputShape;
    const Shape& desired;
    Epilogue epilogue;
    // If the size of an outputGroup is excessive, it is automatically turned into weight dimension.
    mutable std::vector<std::size_t> weightRemainderInputToGroupId;
public:
    template<typename T>
    FinalizeShapeOp(const Shape& outputShape, const Shape& desired, T&& epilogue):
        outputShape { outputShape },
        desired { desired },
        epilogue { std::forward<T>(epilogue) }
    {}
    Shape transformShapeInverse(const Shape& outputShape) const override;
    // Here, we can assert that the interface are all TensorStub.
    void transformTensor(TensorView& tensor) const override;
    inline std::string type() const override { return "Finalize"; }
    std::string description() const override;
    bool isFinalizeOp() const override;

    const Epilogue& getEpilogue() const;

    struct GenerateOptions {
        const Shape& desired;
    };
    static std::optional<FinalizeShapeOp::Epilogue> solveWithMappings(const Shape& outputShape, const Shape& desiredShape, const std::vector<std::size_t>& mappings);
    static std::vector<std::unique_ptr<FinalizeShapeOp>> generate(const Shape& outputShape, GenerateOptions options);
};

} // namespace kas

template<typename T>
inline std::size_t hash_combine(std::size_t seed, const T& v) {
    std::hash<T> hasher;
    return seed ^ (hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template<typename T>
struct std::hash<std::vector<T>> {
    std::size_t operator()(const std::vector<T>& vec) const {
        return std::accumulate(vec.begin(), vec.end(), vec.size(), hash_combine<T>);
    }
};

template<>
struct std::hash<kas::FinalizeShapeOp::Epilogue> {
    using Epilogue = kas::FinalizeShapeOp::Epilogue;
    std::size_t operator()(const Epilogue& e) const noexcept
    {
        std::size_t h1 = std::hash<std::vector<std::size_t>>()(e.desiredInputToGroupId);
        std::size_t h2 = hash_combine(h1, e.outputGroups);
        return h2;
    }
};
