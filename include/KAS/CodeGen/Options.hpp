#pragma once

#include <cstddef>
#include <map>
#include <string>


namespace kas {
    
struct CodeGenOptions {
    bool halide = true;
    enum class AutoScheduler {
        ComputeRoot, Mullapudi2016, Li2018, Adams2019, Anderson2021,
    };
    bool useGPU = true;
    AutoScheduler scheduler = AutoScheduler::Li2018;
    std::map<std::string, std::string> extraOptions;
    std::size_t rfactorThreshold = 32;
    float inBoundsLikelyThreshold = 0.3f;
};

} // namespace kas
