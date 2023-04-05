#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

std::string MapReduceOp::what(MapType type) {
    switch (type) {
        case MapType::Absolute: return "Absolute";
        case MapType::ArcTan:   return "ArcTan";
        case MapType::Exp:      return "Exp";
        case MapType::Log:      return "Log";
        case MapType::Identity: return "Identity";
        case MapType::Inverse:  return "Inverse";
        case MapType::Negative: return "Negative";
        case MapType::ReLU:     return "ReLU";
        case MapType::Sigmoid:  return "Sigmoid";
        case MapType::Sign:     return "Sign";
        case MapType::MapTypeCount: break;
    }
    KAS_UNREACHABLE();
}

std::string MapReduceOp::what(ReduceType type) {
    switch (type) {
        case ReduceType::Sum:     return "Sum";
        case ReduceType::Max:     return "Max";
        case ReduceType::Mean:    return "Mean";
        case ReduceType::Min:     return "Min";
        case ReduceType::Product: return "Product";
        case ReduceType::ReduceTypeCount: break;
    }
    KAS_UNREACHABLE();
}

std::string MapReduceOp::whatMap() const {
    return what(mapType);
}
std::string MapReduceOp::whatReduce() const {
    return what(reduceType);
}

std::string MapReduceOp::what() const {
    return what(mapType) + "+" + what(reduceType);
}

std::vector<MapReduceOp::Base> MapReduceOp::GenerateLastLevelMapReduces(const Shape& outputShape, GenerateOptions options) {
    // TODO

    const BindingContext& ctx = options.ctx;
    auto primaryMeta = ctx.getPrimaryMetadata();
    auto coefficientMeta = ctx.getCoefficientMetadata();

    constexpr std::size_t maxDepth = 2;
    Size totalOutputSize = outputShape.totalSize();
    using BaseShapeView = AbstractShape<const Base&, [](const MapReduceOp& m) -> const Size& { return m.size(); }>;
    std::vector<MapReduceOp::Base> res;

    auto recursion = [&](const auto& self, std::size_t depth, const Base& base) -> void {
        if (depth == maxDepth) {
            return;
        }

        Size totalSize = base.empty() ? totalOutputSize : totalOutputSize * BaseShapeView(base).totalSize();
        Allowance allowance = { totalSize, ctx };

        auto primary = totalSize.getPrimary();
        auto coefficient = totalSize.getCoefficient();
        const std::size_t primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
        std::map<std::size_t, std::size_t> primaryAllowance;
        for (std::size_t i = 0; i < primaryCount; ++i) {
            // Observe that in the sampling process, the primary variables are generated only by MapReduce. So we can limit it with maximumOccurrence.
            if (static_cast<std::size_t>(primary[i]) < primaryMeta[i].maximumOccurrence) {
                primaryAllowance[i] = primaryMeta[i].maximumOccurrence - static_cast<std::size_t>(primary[i]);
            }
        }
        std::vector<std::size_t> coefficientAllowance;
        for (std::size_t i = 0; i < coefficientCount; ++i) {
            // But we interpret maximumOccurrence for coefficient variables in a different way. If a single coefficient appears for too many times in a tensor, do not mess with it.
            if (std::abs(coefficient[i]) < coefficientMeta[i].maximumOccurrence) {
                coefficientAllowance.emplace_back(i);
            }
        }
        for (const auto& coefficientId: coefficientAllowance) {
            auto c = ctx.getSingleCoefficientVariableSize(coefficientId);

            // For simplicity, we only use Identity and Sum. TODO: add more.
            auto base1 = base;
            base1.emplace_back(base1.size(), c, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
            self(self, depth + 1, base1);
            res.emplace_back(std::move(base1));

            for (const auto& [primaryId, allowance]: primaryAllowance) {
                auto p = ctx.getSinglePrimaryVariableSize(primaryId);

                auto base2 = base;
                base2.emplace_back(base2.size(), p, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
                self(self, depth + 1, base2);
                res.emplace_back(std::move(base2));

                auto base3 = base;
                base3.emplace_back(base3.size(), c * p, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
                self(self, depth + 1, base3);
                res.emplace_back(std::move(base3));

                auto base4 = base;
                base4.emplace_back(base4.size(), p / c, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
                self(self, depth + 1, base4);
                res.emplace_back(std::move(base4));
            }
        }
    };
    Base base0;
    recursion(recursion, 0, base0);
    res.emplace_back(std::move(base0));
    return res;
}

} // namespace kas
