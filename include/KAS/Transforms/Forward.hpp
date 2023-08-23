#pragma once

#include <memory>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Reduce.hpp"
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
    virtual void notifyParent() = 0;
    DimensionImpl(Factory& factory, auto&& size):
        factory { factory },
        size { std::forward<decltype(size)>(size) }
    {}
public:
    Factory& getFactory() const { return factory; }
    const Size& getSize() const { return size; }
    bool evaluated() const { return value != nullptr; }
    // Each time a dimension is evluated, its parent Op gets notified. Once all of the children of the Op are evaluated, the Op can be evaluated, propagating to top-most dimensions.
    void set(const BackwardDimension& value) {
        this->value = value.getInnerPointer();
        notifyParent();
    }
    BackwardDimension get() const {
        KAS_ASSERT(evaluated(), "Dimension is not evaluated yet");
        return value;
    }
    virtual ~DimensionImpl() = default;
};

class Pure final: public DimensionImpl {
    void notifyParent() override {}
public:
    Pure(Factory& factory, auto&& size):
        DimensionImpl { factory, std::forward<decltype(size)>(size) }
    {}
};

class Expand final: public DimensionImpl {
    const ::kas::ExpandOp *op;
    void notifyParent() override;
public:
    Expand(Factory& factory, auto&& size):
        DimensionImpl { factory, std::forward<decltype(size)>(size) }
    {}
    const ::kas::ExpandOp *getOp() const;
};

class Dimension {
    std::shared_ptr<DimensionImpl> inner;
public:
    Dimension(std::shared_ptr<DimensionImpl> inner): inner { std::move(inner) } {}
    Factory& getFactory() const { return inner->getFactory(); }
    const Size& getSize() const { return inner->getSize(); }
    std::string sizeToString() const;
    bool evaluated() const { return inner->evaluated(); }
    const Expand *asExpanded() const { return dynamic_cast<const Expand *>(inner.get()); }
    void set(const BackwardDimension& value) { inner->set(value); }
    BackwardDimension get() const { return inner->get(); }
    operator BackwardDimension() const { return get(); }
    void output(std::size_t index);
    void reduce(Reduce::ReduceType reduceType);
};

class Factory {
    const BindingContext& ctx;
    PrimitiveOpStore store;
    std::vector<std::unique_ptr<Iterator>> iterators;
    std::vector<BackwardDimension> bottommost;
    std::unique_ptr<TensorView> result;

public:
    Factory(const BindingContext& ctx): ctx { ctx } {}
    template<typename Arg>
    Size getSize(Arg&& arg) const {
        return ctx.getSize(std::forward<decltype(arg)>(arg));
    }
    template<typename... Args>
    requires std::conjunction_v<std::is_convertible<Args, std::string>...>
    auto getSizes(Args&&... args) const -> std::array<Size, sizeof...(Args)> {
        return ctx.getSizes(std::forward<decltype(args)>(args)...);
    }
    std::vector<Size> getSizes(const std::vector<std::string>& names) const {
        return ctx.getSizes(names);
    }
    [[nodiscard]] Dimension makeDimOfSize(auto&& size) {
        return Dimension(std::make_shared<Pure>(*this, std::forward<decltype(size)>(size)));
    }
    [[nodiscard]] Dimension makeExpandOfSize(auto&& size) {
        return Dimension(std::make_shared<Expand>(*this, std::forward<decltype(size)>(size)));
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
    const Iterator *createIterator(const Size& domain, std::size_t index);
    const Reduce *createReduce(const Size& domain, Reduce::ReduceType reduceType);

    static std::vector<Topmost> ForwardDimsToBackwardDims(const std::vector<std::vector<Dimension>>& tensors);
    TensorView& buildTensorView(const std::vector<std::vector<Dimension>>& tensors, TensorExpression blending);
};

class ExpandOp {
public:
    [[nodiscard]] static Dimension Create(Factory& factory, const Size& size) {
        return factory.makeExpandOfSize(size);
    }
};

class Op {
public:
    virtual void onNotification(Factory& store) = 0;
    virtual ~Op() = default;
};

class RepeatLikeOp: public Op {
public:
    class Output final: public DimensionImpl {
        std::unique_ptr<RepeatLikeOp> op;
        void notifyParent() override {
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
    void setOutput(std::shared_ptr<Output> output) {
        this->output = std::move(output);
    }
    RepeatLikeOp(const Dimension& input): input { input } {}
};

class SplitLikeOp: public Op {
public:
    class Output final: public DimensionImpl {
        std::shared_ptr<SplitLikeOp> op;
        void notifyParent() override {
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
    void setOutput(std::shared_ptr<Output> output, Order order) {
        switch (order) {
        case Order::Left:
            this->outputLhs = std::move(output);
            break;
        case Order::Right:
            this->outputRhs = std::move(output);
            break;
        }
    }
    SplitLikeOp(const Dimension& input): input { input } {}
};

class MergeLikeOp: public Op {
public:
    class Output final: public DimensionImpl {
        std::unique_ptr<MergeLikeOp> op;
        void notifyParent() override {
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
    void setOutput(std::shared_ptr<Output> output) {
        this->output = std::move(output);
    }
    MergeLikeOp(const Dimension& lhs, const Dimension& rhs): inputLhs { lhs }, inputRhs { rhs } {}
};

class MergeOp final: public MergeLikeOp {
    using MergeLikeOp::MergeLikeOp;
public:
    void onNotification(Factory& factory) override;
    [[nodiscard]] static Dimension Create(const Dimension& lhs, const Dimension& rhs);
};

class ShareOp final: public MergeLikeOp {
    using MergeLikeOp::MergeLikeOp;
public:
    void onNotification(Factory& factory) override;
    [[nodiscard]] static Dimension Create(const Dimension& lhs, const Dimension& rhs);
};

class ShiftOp final: public RepeatLikeOp {
    int shift;
    ShiftOp(const Dimension& input, int shift):
        RepeatLikeOp { input },
        shift { shift }
    {}
public:
    void onNotification(Factory& factory) override;
    [[nodiscard]] static Dimension Create(const Dimension& input, int shift);
};

class SplitOp final: public SplitLikeOp {
    using SplitLikeOp::SplitLikeOp;
public:
    void onNotification(Factory& factory) override;
    [[nodiscard]] static std::pair<Dimension, Dimension> Create(const Dimension& input, const Size& block);
};

class StrideOp final: public RepeatLikeOp {
    Size stride;
    StrideOp(const Dimension& input, auto&& stride):
        RepeatLikeOp { input },
        stride { std::forward<decltype(stride)>(stride) }
    {}
public:
    void onNotification(Factory& factory) override;
    [[nodiscard]] static Dimension Create(const Dimension& input, const Size& stride);
};

class UnfoldOp final: public SplitLikeOp {
    UnfoldOp(const Dimension& input):
        SplitLikeOp { input }
    {}
public:
    void onNotification(Factory& factory) override;
    [[nodiscard]] static std::pair<Dimension, Dimension> Create(const Dimension& input, const Size& window);
};

} // namespace Forward

} // namespace kas
