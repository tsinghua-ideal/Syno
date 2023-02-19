#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

// There are 3 kinds of `PrimitiveOp`'s, listed below. Those classes can transform `Dimension`s, from those that index the output tensor, to forms that index the original tensors. So this is also kind of bottom-up.

class DimensionStore;
class Dimension;
using Interface = std::vector<Dimension>;

template<typename Op>
concept PrimitiveOpBase = std::same_as<Op, std::remove_cvref_t<Op>> && std::equality_comparable<Op> && requires(const Op op) {
    { op.size() } -> std::same_as<const Size&>;
    { op.initialHash() } noexcept -> std::same_as<std::size_t>;
    { Op::Type() } -> std::same_as<const char*>;
    typename Op::GenerateOptions;
};

class Dimension;
using DoubleIteratorValue = std::pair<IteratorValue, IteratorValue>;

// By repeat-like, we refer to the primitives that have one input iterator and one output iterator.
template<typename Op>
concept RepeatLikePrimitiveOp = PrimitiveOpBase<Op> && requires(const Op op) {
    { op.output } -> std::same_as<const Dimension&>;
    { op.value(std::declval<const IteratorValue&>()) } -> std::same_as<IteratorValue>;
    { op.value(std::declval<IteratorValue&&>()) } -> std::same_as<IteratorValue>;
    { Op::Generate(std::declval<DimensionStore&>(), std::declval<const Interface&>(), std::declval<typename Op::GenerateOptions>()) } -> std::same_as<std::vector<Dimension>>;
};

// By split-like, we refer to the primitives that have one input iterator and two output iterators.
template<typename Op>
concept SplitLikePrimitiveOp = PrimitiveOpBase<Op> && requires(const Op op) {
    { op.outputLhs } -> std::same_as<const Dimension&>;
    { op.outputRhs } -> std::same_as<const Dimension&>;
    { op.value(std::declval<const DoubleIteratorValue&>()) } -> std::same_as<IteratorValue>;
    { op.value(std::declval<DoubleIteratorValue&&>()) } -> std::same_as<IteratorValue>;
    { Op::Generate(std::declval<DimensionStore&>(), std::declval<const Interface&>(), std::declval<typename Op::GenerateOptions>()) } -> std::same_as<std::vector<Dimension>>;
};

enum class FirstOrSecond: bool {
    First = false,
    Second = true,
};
// By merge-like, we refer to the primitives that have two input iterators and one output iterator.
template<typename Op>
concept MergeLikePrimitiveOp = PrimitiveOpBase<Op> && requires(const Op op) {
    { op.output } -> std::same_as<const Dimension&>;
    { op.firstOrSecond } -> std::same_as<const FirstOrSecond&>;
    { op.value(std::declval<const IteratorValue&>()) } -> std::same_as<DoubleIteratorValue>;
    { op.value(std::declval<IteratorValue&&>()) } -> std::same_as<DoubleIteratorValue>;
    { Op::Generate(std::declval<DimensionStore&>(), std::declval<const Interface&>(), std::declval<typename Op::GenerateOptions>()) } -> std::same_as<std::vector<std::pair<Dimension, Dimension>>>;
};

template<typename Op>
concept PrimitiveOp = RepeatLikePrimitiveOp<Op> || SplitLikePrimitiveOp<Op> || MergeLikePrimitiveOp<Op>;

} // namespace kas
