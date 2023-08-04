#pragma once

#include <memory>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

class TensorView;

namespace Forward {

using BackwardDimension = ::kas::Dimension;
using BackwardDimensionImpl = ::kas::DimensionImpl;

class Factory;

class DimensionImpl {
protected:
    Factory& factory;
    Size size;
    const BackwardDimensionImpl *value = nullptr;
    virtual void notifyParent() const = 0;
    DimensionImpl(Factory& factory, auto&& size):
        factory { factory },
        size { std::forward<decltype(size)>(size) }
    {}
public:
    inline Factory& getFactory() const { return factory; }
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
    inline Factory& getFactory() const { return inner->getFactory(); }
    inline const Size& getSize() const { return inner->getSize(); }
    std::string sizeToString() const;
    inline bool evaluated() const { return inner->evaluated(); }
    inline void set(const BackwardDimension& value) { inner->set(value); }
    inline BackwardDimension get() const { return inner->get(); }
    inline operator BackwardDimension() const { return get(); }
    void output(std::size_t index);
    void reduce(std::size_t priority, MapReduce::MapType mapType, MapReduce::ReduceType reduceType);
};

class Pure final: public DimensionImpl {
protected:
    inline void notifyParent() const override {}
public:
    Pure(Factory& factory, auto&& size):
        DimensionImpl { factory, std::forward<decltype(size)>(size) }
    {}
};

class Factory {
    const BindingContext& ctx;
    PrimitiveOpStore store;
    std::vector<std::unique_ptr<Iterator>> iterators;
    std::vector<std::unique_ptr<MapReduceOp>> mapReduces;
    std::unique_ptr<TensorView> result;

public:
    inline Factory(const BindingContext& ctx): ctx { ctx } {}
    template<typename Arg>
    inline Size getSize(Arg&& arg) const {
        return ctx.getSize(std::forward<decltype(arg)>(arg));
    }
    template<typename... Args>
    requires std::conjunction_v<std::is_convertible<Args, std::string>...>
    auto getSizes(Args&&... args) const -> std::array<Size, sizeof...(Args)> {
        return ctx.getSizes(std::forward<decltype(args)>(args)...);
    }
    inline std::vector<Size> getSizes(const std::vector<std::string>& names) const {
        return ctx.getSizes(names);
    }
    [[nodiscard]] Dimension makeDimOfSize(auto&& size) {
        return Dimension(std::make_shared<Pure>(*this, std::forward<decltype(size)>(size)));
    }
    template<typename... Sizes>
    requires std::conjunction_v<std::is_convertible<Sizes, Size>...>
    [[nodiscard]] auto makeDimsOfSizes(Sizes&&... sizes) -> std::array<Dimension, sizeof...(Sizes)> {
        return { makeDimOfSize(std::forward<decltype(sizes)>(sizes))... };
    }
    template<SizeRange R>
    [[nodiscard]] std::vector<Dimension> makeDimsOfShape(R&& shape) {
        std::vector<Dimension> result;
        for (auto&& size: shape) {
            result.emplace_back(makeDimOfSize(std::forward<decltype(size)>(size)));
        }
        return result;
    }

    const BindingContext& getBindingContext() const { return ctx; }
    PrimitiveOpStore& getStore() { return store; }
    void storeIterator(std::unique_ptr<Iterator> iterator);
    void storeMapReduce(std::unique_ptr<MapReduceOp> mapReduce);

    // TODO!!! Expand needs to be wrapped in ForwardDimension!
    static std::vector<Topmost> ForwardDimsToBackwardDims(const std::vector<std::vector<Dimension>>& tensors);
    TensorView& buildTensorView(const std::vector<std::vector<Dimension>>& tensors, TensorExpression blending);
};

class Op {
public:
    virtual void onNotification(Factory& store) = 0;
};

class RepeatLikeOp: public Op {
public:
    class Output final: public DimensionImpl {
        std::unique_ptr<RepeatLikeOp> op;
        inline void notifyParent() const override {
            op->onNotification(factory);
        }
        Output(Factory& factory, auto&& size, std::unique_ptr<RepeatLikeOp> op):
            DimensionImpl { factory, std::forward<decltype(size)>(size) },
            op { std::move(op) }
        {}
    public:
        static std::shared_ptr<Output> Create(Factory& factory, auto&& size, std::unique_ptr<RepeatLikeOp> op) {
            auto ptr = std::shared_ptr<Output>(new Output { factory, std::forward<decltype(size)>(size), std::move(op) });
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
            op->onNotification(factory);
        }
        Output(Factory& factory, auto&& size, std::shared_ptr<SplitLikeOp> op, Order order):
            DimensionImpl { factory, std::forward<decltype(size)>(size) },
            op { std::move(op) }
        {}
    public:
        static std::shared_ptr<Output> Create(Factory& factory, auto&& size, std::shared_ptr<SplitLikeOp> op, Order order) {
            auto ptr = std::shared_ptr<Output>(new Output { factory, std::forward<decltype(size)>(size), std::move(op), order });
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
            op->onNotification(factory);
        }
        Output(Factory& factory, auto&& size, std::unique_ptr<MergeLikeOp> op):
            DimensionImpl { factory, std::forward<decltype(size)>(size) },
            op { std::move(op) }
        {}
    public:
        static std::shared_ptr<Output> Create(Factory& factory, auto&& size, std::unique_ptr<MergeLikeOp> op) {
            auto ptr = std::shared_ptr<Output>(new Output { factory, std::forward<decltype(size)>(size), std::move(op) });
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
    void onNotification(Factory& factory) override;
    static Dimension Create(const Dimension& lhs, const Dimension& rhs);
};

class ShareOp final: public MergeLikeOp {
protected:
    using MergeLikeOp::MergeLikeOp;
public:
    void onNotification(Factory& factory) override;
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
    void onNotification(Factory& factory) override;
    static Dimension Create(const Dimension& input, int shift);
};

class SplitOp final: public SplitLikeOp {
protected:
    using SplitLikeOp::SplitLikeOp;
public:
    void onNotification(Factory& factory) override;
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
    void onNotification(Factory& factory) override;
    static Dimension Create(const Dimension& input, const Size& stride);
};

class UnfoldOp final: public SplitLikeOp {
protected:
    UnfoldOp(const Dimension& input):
        SplitLikeOp { input }
    {}
public:
    void onNotification(Factory& factory) override;
    static std::pair<Dimension, Dimension> Create(const Dimension& input, const Size& window);
};

} // namespace Forward

} // namespace kas
