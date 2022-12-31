#pragma once

#include <functional>
#include <memory>
#include <ranges>
#include <vector>
#include <sstream>


namespace kas {

template<typename T>
std::vector<T> ReplaceVector(
    const std::vector<T>& vec, 
    std::vector<std::size_t>& drops,
    std::vector<std::pair<std::size_t, T>>& adds
) {
    std::sort(drops.begin(), drops.end());
    std::sort(adds.begin(), adds.end(), std::function([](const std::pair<std::size_t, T>& a, const std::pair<std::size_t, T>& b) -> bool {
        return a.first < b.first;
    }));

    std::vector<T> newVec;
    newVec.reserve(vec.size() + adds.size() - drops.size());

    if (drops.size() > 0) {
        std::size_t dropIndex = 0;
        int nextDrop = drops[dropIndex];
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (nextDrop == i) {
                ++dropIndex;
                if (dropIndex < drops.size()) {
                    nextDrop = drops[dropIndex];
                } else {
                    nextDrop = -1;
                }
            } else {
                newVec.push_back(vec[i]);
            }
        }
    } else {
        newVec = vec;
    }

    for (const auto& add: adds) {
        newVec.insert(newVec.begin() + add.first, std::move(add.second));
    }

    return newVec;
}

template<typename T>
std::string VectorToString(const std::vector<T>& vec, std::function<std::string(const T&)> mapper) {
    std::stringstream ss;
    ss << "[";
    for (std::size_t i = 0; i < vec.size(); i++) {
        if (i != 0) {
            ss << ",";
        }
        ss << mapper(vec[i]);
    }
    ss << "]";
    return ss.str();
}

} // namespace kas
