#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

Iterator::Iterator(IteratorTransform parent):
    parent { std::move(parent) }
{}

} // namespace kas
