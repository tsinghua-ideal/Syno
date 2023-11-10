#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

enum class ContractionType: bool {
    Outer, // Basically a tensor product.
    Inner, // Basically a contraction.
};

class ContractionOpStore;

class ContractionOp {
    struct Dimwise {
        Dimension dim;
        ContractionType type;
        bool operator==(const Dimwise& other) const noexcept = default;
        std::weak_ordering operator<=>(const Dimwise& other) const noexcept {
            auto hash = dim.hash() <=> other.dim.hash();
            if (hash != 0) {
                return hash;
            }
            return type <=> other.type;
        }
    };
public:
    bool operator==(const ContractionOp& other) const noexcept;
    std::size_t opHash() const noexcept;
    GraphHandle applyToInterface(const GraphHandle& interface) const;
    std::string description(const BindingContext& ctx) const;
    std::string descendantsDescription(const BindingContext& ctx) const;
    static std::vector<const ContractionOp *> Generate(ContractionOpStore& store);
};

static_assert(GeneralizedOp<ContractionOp>);

} // namespace kas
