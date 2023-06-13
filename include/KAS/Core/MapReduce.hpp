#pragma once

#include <memory>
#include <unordered_set>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

class ReductionStore;

class MapReduceOp final: public DimensionImpl {
public:
    enum class MapType {
        Absolute,
        ArcTan,
        Exp,
        Log,
        Identity,
        Inverse,
        Negative,
        ReLU,
        Sigmoid,
        Sign,
        MapTypeCount,
    };
    static std::string what(MapType);
    enum class ReduceType {
        Sum,
        Max,
        Mean,
        Min,
        Product,
        ReduceTypeCount
    };
    static std::string what(ReduceType);

protected:
    // This decides the order in which MapReduce is applied.
    std::size_t priority;
    Size domain;

    MapType mapType;
    ReduceType reduceType;

public:
    MapReduceOp(std::size_t priority, auto&& domain, MapType mapType, ReduceType reduceType):
        priority { priority },
        domain { std::forward<decltype(domain)>(domain) },
        mapType { mapType },
        reduceType { reduceType }
    {}
    const Size& size() const noexcept override { return domain; }
    std::size_t hash() const noexcept override {
        std::size_t h = std::hash<DimensionType>{}(DimensionType::MapReduce);
        HashCombine(h, mapType);
        HashCombine(h, reduceType);
        HashCombine(h, priority);
        HashCombine(h, domain);
        return h;
    }
    constexpr DimensionType type() const noexcept override { return DimensionType::MapReduce; }
    void accept(DimVisitor& visitor) const final override;

    bool operator==(const MapReduceOp& other) const noexcept {
        return mapType == other.mapType && reduceType == other.reduceType && priority == other.priority && domain == other.domain;
    }

    MapType getMap() const { return mapType; }
    ReduceType getReduce() const { return reduceType; }

    std::size_t getPriority() const { return priority; }
    std::string getName() const {
        return "ri_" + std::to_string(priority);
    }
    std::string whatMap() const;
    std::string whatReduce() const;
    std::string what() const;

    std::string description(const BindingContext& ctx) const;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
        Size outputSize;
        std::size_t maxFLOPs;
    };
    static std::vector<const MapReduceOp *> Generate(ReductionStore& store, const std::vector<const MapReduceOp *>& current, const GenerateOptions& options);
};

class ReductionStore {
    struct Hash {
        std::size_t operator()(const MapReduceOp *op) const {
            return op->hash();
        }
    };
    struct Equal {
        bool operator()(const MapReduceOp *lhs, const MapReduceOp *rhs) const {
            return *lhs == *rhs;
        }
    };
    std::unordered_set<MapReduceOp *, Hash, Equal> store;
public:
    ReductionStore() = default;
    ReductionStore(const ReductionStore&) = delete;
    ReductionStore(ReductionStore&&) = delete;
    const MapReduceOp *get(auto&&... args) {
        auto op = std::make_unique<MapReduceOp>(std::forward<decltype(args)>(args)...);
        auto it = store.find(op.get());
        if (it != store.end()) {
            return *it;
        }
        auto ptr = op.release();
        store.insert(ptr);
        return ptr;
    }
    ~ReductionStore() {
        for (auto *op: store) {
            delete op;
        }
    }
};

} // namespace kas
