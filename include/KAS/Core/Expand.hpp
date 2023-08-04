#pragma once

#include "KAS/Core/Dimension.hpp"


namespace kas {

class Expand {
public:
    Dimension output;
    Expand(const Dimension& output):
        output { output }
    {}
    struct PointerToDimension {
        const Dimension& operator()(const Expand *expand) const noexcept { return expand->output; }
    };
};

} // namespace kas

template<>
struct std::hash<kas::Topmost> {
    std::size_t operator()(const kas::Topmost& interface) const noexcept {
        constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
        using namespace std::string_view_literals;
        static const auto tHash = std::hash<std::string_view>{}("Topmost"sv);
        static const auto dHash = std::hash<std::string_view>{}("dimensions"sv);
        static const auto eHash = std::hash<std::string_view>{}("expansions"sv);
        auto h = tHash;
        const auto hasher = std::hash<std::vector<kas::Dimension>>{};
        auto dimsHash = hasher(interface.getDimensions());
        auto expsHash = hasher(interface.getExpansions() | std::views::transform(kas::Expand::PointerToDimension{}));
        kas::HashCombineRaw(h, dHash);
        kas::HashCombineRaw(h, dimsHash);
        kas::HashCombineRaw(h, std::rotl(eHash, SizeTypeWidth / 2));
        kas::HashCombineRaw(h, std::rotl(expsHash, SizeTypeWidth / 2));
        return h;
    }
};

template<>
struct std::hash<kas::GraphHandle> {
    std::size_t operator()(const kas::GraphHandle& handle) const noexcept {
        return std::hash<kas::Topmost>{}(handle.getRaw());
    }
};

template<>
struct std::hash<std::vector<kas::Topmost>> {
    template<kas::TopmostRange R>
    std::size_t operator()(R&& topmosts) const noexcept {
        using namespace std::string_view_literals;
        static const auto trHash = std::hash<std::string_view>{}("TopmostRange"sv);
        auto h = trHash;
        kas::HashCombine(h, topmosts.size());
        auto hasher = std::hash<kas::Topmost>{};
        for (const auto& topmost: topmosts) {
            kas::HashCombineRaw(h, hasher(topmost));
        }
        return h;
    }
};
