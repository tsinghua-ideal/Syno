#pragma once

#include <memory>
#include <vector>


namespace kas {

template<typename T>
std::vector<T> ReplaceVector(
    const std::vector<T>& vec, 
    const std::vector<int>& drops,
    const std::vector<std::pair<int, T>>& adds
) {
    std::vector<T> newVec;
    newVec.reserve(vec.size() + adds.size() - drops.size());

    if (drops.size() > 0) {
        int dropIndex = 0;
        int nextDrop = drops[dropIndex];
        for (int i = 0; i < vec.size(); ++i) {
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

} // namespace kas
