#include <algorithm>

#include <fmt/core.h>
#include <fmt/format.h>

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::string Dimension::description(const BindingContext& ctx) const {
    return fmt::format("[{}]@{}{:x}", size().toString(ctx), type(), hash());
}

std::string Dimension::descendantsDescription(const BindingContext& ctx) const {
    struct visitor: public DimVisitor {
        const BindingContext& ctx;
        std::string result;
        void visit(const Iterator& dim) override {
            result = std::to_string(dim.getIndex());
        }
        void visit(const MapReduce& dim) override {
            result = std::to_string(dim.getPriority());
        }
        void visit(const RepeatLikeOp::Input& dim) override {
            result = dim.getOp()->output.description(ctx);
        }
        void visit(const SplitLikeOp::Input& dim) override {
            auto op = dim.getOp();
            result = fmt::format("{}, {}", op->outputLhs.description(ctx), op->outputRhs.description(ctx));
        }
        void visit(const MergeLikeOp::Input& dim) override {
            result = dim.getOp()->output.description(ctx);
        }
        visitor(const BindingContext& ctx): ctx(ctx) {}
        using DimVisitor::visit;
    };
    visitor v { ctx };
    v.visit(*this);
    return fmt::format("{}({})", description(ctx), std::move(v.result));
}

std::ostream& operator<<(std::ostream& os, kas::DimensionType t) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", t);
    return os;
}

std::ostream& operator<<(std::ostream& os, kas::Direction d) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", d);
    return os;
}

std::ostream& operator<<(std::ostream& os, kas::Order o) {
    fmt::format_to(std::ostreambuf_iterator<char>(os), "{}", o);
    return os;
}

} // namespace kas
