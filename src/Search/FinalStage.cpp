#include "KAS/Search/FinalStage.hpp"
#include "KAS/Search/NormalStage.hpp"


namespace kas {

std::string FinalStage::description() const {
    return value.description(parent.sampler.getBindingContext());
}

} // namespace kas
