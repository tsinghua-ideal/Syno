#include "KAS/Search/Node.hpp"
#include "KAS/Core/Dimension.hpp"


namespace kas {

std::map<std::string, std::size_t> Node::childrenTypes() const {
    std::map<std::string, std::size_t> result;
    for (std::size_t i = 0; i < countChildren(); i++) {
        ++result[opType(i)];
    }
    return result;
}

} // namespace kas
