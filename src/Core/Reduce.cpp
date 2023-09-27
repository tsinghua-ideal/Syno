#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

std::string ReduceBase::what(ReduceType type) {
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

std::string ReduceBase::whatReduce() const {
    return what(reduceType);
}

std::size_t ReduceBase::pureHash() const noexcept {
    using namespace std::string_view_literals;
    std::size_t h = DimensionTypeHash(DimensionType::Reduce);
    HashCombine(h, reduceType);
    static const auto reduceDomainHash = std::hash<std::string_view>{}("ReduceDomain"sv);
    HashCombineRaw(h, reduceDomainHash);
    HashCombine(h, domain);
    return h;
}

std::size_t Reduce::hash() const noexcept {
    using namespace std::string_view_literals;
    constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
    std::size_t h = getBase().pureHash();
    static const auto reduceMultiplicityHash = std::hash<std::string_view>{}("ReduceIndex"sv);
    HashCombine(h, std::rotl(reduceMultiplicityHash, SizeTypeWidth / static_cast<int>(ExpectedMaximumReduces) * multiplicity));
    return h;
}

std::string Reduce::description(const BindingContext& ctx) const {
    return fmt::format("[{}]@Reduce", base.getDomain().toString(ctx));
}

std::string Reduce::debugDescription() const {
    return BindingContext::ApplyDebugPublicCtx(&Reduce::description, *this);
}

} // namespace kas
