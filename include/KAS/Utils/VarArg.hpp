#include <cstddef>
#include <utility>


namespace kas {

namespace detail {

template<typename R, typename... Args>
using FunctionPtrOfArgs = R (*)(Args...);

template<typename T, std::size_t Dummy>
using IdentityWithDummy = T;

template<typename R, typename T, typename... LeadingArgs, std::size_t... Dummy>
decltype(auto) FunctionPtrOfTypedArgs(std::index_sequence<Dummy...>) {
    return FunctionPtrOfArgs<R, LeadingArgs..., IdentityWithDummy<T, Dummy>...>{};
}

} // namespace detail

template<typename R, typename T, std::size_t Count, typename... LeadingArgs>
using FunctionPtrOfNArgs = decltype(detail::FunctionPtrOfTypedArgs<R, T, LeadingArgs...>(std::make_index_sequence<Count>{}));

} // namespace kas
