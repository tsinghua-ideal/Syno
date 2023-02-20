#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Search/Colors.hpp"
#include "KAS/Transforms/Finalize.hpp"


namespace kas {

class Stage {
    // The interface decides the hash. Other properties are computed.
    std::vector<Dimension> interface;
    Colors colors;
    std::vector<std::reference_wrapper<const Size>> missingSizes;
    // Lazily computed.
    std::optional<FinalizeOp> finalizer;
public:
    const FinalizeOp& getFinalizer() const;
};

} // namespace kas
