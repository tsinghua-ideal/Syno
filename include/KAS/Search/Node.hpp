#pragma once

#include <cstddef>
#include <map>

#include "KAS/Search/Stage.hpp"


namespace kas {

class StageStore;

class Node {
    Stage *stage;

public:
    inline Node(): stage { nullptr } {}
    inline Node(Stage *stage): stage { stage } {}
    inline std::size_t countChildren() const { return stage->countChildren(); }
    std::map<std::string, std::size_t> childrenTypes() const;
    inline bool isFinal(std::size_t index) const { return stage->isFinal(index); }
    inline std::variant<Stage *, TensorView *> next(std::size_t index) const { return stage->next(index); }
    static inline Node AssertNotFinal(std::variant<Stage *, TensorView *> n) { return std::get<Stage *>(n); }
    static inline TensorView *AssertFinal(std::variant<Stage *, TensorView *> n) { return std::get<TensorView *>(n); }
    inline std::string shapeDescription(const BindingContext& ctx) const { return ShapeView(stage->getInterface()).toString(ctx); }
    inline std::string opType(std::size_t index) const { return stage->opType(index); }
    inline std::string opDescription(std::size_t index) const { return stage->opDescription(index); }
};

}