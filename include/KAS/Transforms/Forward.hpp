#pragma once

#include <memory>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

namespace Forward {

using BackwardDimension = ::kas::Dimension;
using BackwardDimensionImpl = ::kas::DimensionImpl;

class DimensionImpl {
protected:
    DimensionStore& store;
    Size size;
    const BackwardDimensionImpl *value = nullptr;
    virtual void notifyParent() const = 0;
    DimensionImpl(DimensionStore& store, auto&& size):
        store { store },
        size { std::forward<decltype(size)>(size) }
    {}
public:
    inline DimensionStore& getStore() const { return store; }
    inline const Size& getSize() const { return size; }
    inline bool evaluated() const { return value != nullptr; }
    // Each time a dimension is evluated, its parent Op gets notified. Once all of the children of the Op are evaluated, the Op can be evaluated, propagating to top-most dimensions.
    inline void set(const BackwardDimension& value) {
        this->value = value.getInnerPointer();
        notifyParent();
    }
    inline BackwardDimension get() const {
        KAS_ASSERT(evaluated(), "Dimension is not evaluated yet");
        return value;
    }
};

class Dimension {
    std::shared_ptr<DimensionImpl> inner;
public:
    inline Dimension(std::shared_ptr<DimensionImpl> inner): inner { std::move(inner) } {}
    inline DimensionStore& getStore() const { return inner->getStore(); }
    inline const Size& getSize() const { return inner->getSize(); }
    inline bool evaluated() const { return inner->evaluated(); }
    inline void set(const BackwardDimension& value) { inner->set(value); }
    inline BackwardDimension get() const { return inner->get(); }
    inline operator BackwardDimension() const { return get(); }
    [[nodiscard]] std::unique_ptr<Iterator> output(std::size_t index);
    [[nodiscard]] std::unique_ptr<MapReduceOp> reduce(std::size_t priority, MapReduceOp::MapType mapType, MapReduceOp::ReduceType reduceType);
};

class Pure final: public DimensionImpl {
protected:
    inline void notifyParent() const override {}
public:
    Pure(DimensionStore& store, auto&& size):
        DimensionImpl { store, std::forward<decltype(size)>(size) }
    {}
};

class Factory {
    DimensionStore store;
public:
    [[nodiscard]] Dimension makeSize(auto&& size) {
        return Dimension(std::make_shared<Pure>(store, std::forward<decltype(size)>(size)));
    }
    template<typename... Sizes>
    [[nodiscard]] auto makeSizes(Sizes&&... sizes) -> std::array<Dimension, sizeof...(Sizes)> {
        return { makeSize(std::forward<decltype(sizes)>(sizes))... };
    }
    template<typename Storage, auto Mapping>
    [[nodiscard]] std::vector<Dimension> makeShape(const AbstractShape<Storage, Mapping>& shape) {
        std::vector<Dimension> result;
        for (auto&& size: shape) {
            result.emplace_back(makeSize(size));
        }
        return result;
    }
};

class Op {
public:
    virtual void onNotification(DimensionStore& store) = 0;
};

class RepeatLikeOp: public Op {
public:
    class Output final: public DimensionImpl {
        std::unique_ptr<RepeatLikeOp> op;
        inline void notifyParent() const override {
            op->onNotification(store);
        }
        Output(DimensionStore& store, auto&& size, std::unique_ptr<RepeatLikeOp> op):
            DimensionImpl { store, std::forward<decltype(size)>(size) },
            op { std::move(op) }
        {}
    public:
        static std::shared_ptr<Output> Create(DimensionStore& store, auto&& size, std::unique_ptr<RepeatLikeOp> op) {
            auto ptr = std::shared_ptr<Output>(new Output { store, std::forward<decltype(size)>(size), std::move(op) });
            ptr->op->setOutput(ptr);
            return ptr;
        }
    };
protected:
    Dimension input;
    std::weak_ptr<Output> output;
    inline void setOutput(std::shared_ptr<Output> output) {
        this->output = std::move(output);
    }
    inline RepeatLikeOp(const Dimension& input): input { input } {}
};

class SplitLikeOp: public Op {
public:
    class Output final: public DimensionImpl {
        std::shared_ptr<SplitLikeOp> op;
        inline void notifyParent() const override {
            op->onNotification(store);
        }
        Output(DimensionStore& store, auto&& size, std::shared_ptr<SplitLikeOp> op, Order order):
            DimensionImpl { store, std::forward<decltype(size)>(size) },
            op { std::move(op) }
        {}
    public:
        static std::shared_ptr<Output> Create(DimensionStore& store, auto&& size, std::shared_ptr<SplitLikeOp> op, Order order) {
            auto ptr = std::shared_ptr<Output>(new Output { store, std::forward<decltype(size)>(size), std::move(op), order });
            ptr->op->setOutput(ptr, order);
            return ptr;
        }
    };
protected:
    Dimension input;
    std::weak_ptr<Output> outputLhs, outputRhs;
    inline void setOutput(std::shared_ptr<Output> output, Order order) {
        switch (order) {
        case Order::Left:
            this->outputLhs = std::move(output);
            break;
        case Order::Right:
            this->outputRhs = std::move(output);
            break;
        }
    }
    inline SplitLikeOp(const Dimension& input) : input { input } {}
};

class MergeLikeOp: public Op {
public:
    class Output final: public DimensionImpl {
        std::unique_ptr<MergeLikeOp> op;
        inline void notifyParent() const override {
            op->onNotification(store);
        }
        Output(DimensionStore& store, auto&& size, std::unique_ptr<MergeLikeOp> op):
            DimensionImpl { store, std::forward<decltype(size)>(size) },
            op { std::move(op) }
        {}
    public:
        static std::shared_ptr<Output> Create(DimensionStore& store, auto&& size, std::unique_ptr<MergeLikeOp> op) {
            auto ptr = std::shared_ptr<Output>(new Output { store, std::forward<decltype(size)>(size), std::move(op) });
            ptr->op->setOutput(ptr);
            return ptr;
        }
    };
protected:
    Dimension inputLhs, inputRhs;
    std::weak_ptr<Output> output;
    inline void setOutput(std::shared_ptr<Output> output) {
        this->output = std::move(output);
    }
    inline MergeLikeOp(const Dimension& lhs, const Dimension& rhs) : inputLhs { lhs }, inputRhs { rhs } {}
};

class MergeOp final: public MergeLikeOp {
protected:
    using MergeLikeOp::MergeLikeOp;
public:
    void onNotification(DimensionStore& store) override;
    static Dimension Create(const Dimension& lhs, const Dimension& rhs);
};

class ShareOp final: public MergeLikeOp {
protected:
    using MergeLikeOp::MergeLikeOp;
public:
    void onNotification(DimensionStore& store) override;
    static Dimension Create(const Dimension& lhs, const Dimension& rhs);
};

class ShiftOp final: public RepeatLikeOp {
    int shift;
protected:
    inline ShiftOp(const Dimension& input, int shift):
        RepeatLikeOp { input },
        shift { shift }
    {}
public:
    void onNotification(DimensionStore& store) override;
    static Dimension Create(const Dimension& input, int shift);
};

class SplitOp final: public SplitLikeOp {
protected:
    using SplitLikeOp::SplitLikeOp;
public:
    void onNotification(DimensionStore& store) override;
    static std::pair<Dimension, Dimension> Create(const Dimension& input, const Size& block);
};

class StrideOp final: public RepeatLikeOp {
    Size stride;
protected:
    StrideOp(const Dimension& input, auto&& stride):
        RepeatLikeOp { input },
        stride { std::forward<decltype(stride)>(stride) }
    {}
public:
    void onNotification(DimensionStore& store) override;
    static Dimension Create(const Dimension& input, const Size& stride);
};

class UnfoldOp final: public SplitLikeOp {
protected:
    UnfoldOp(const Dimension& input):
        SplitLikeOp { input }
    {}
public:
    void onNotification(DimensionStore& store) override;
    static std::pair<Dimension, Dimension> Create(const Dimension& input, const Size& window);
};

} // namespace Forward

} // namespace kas
