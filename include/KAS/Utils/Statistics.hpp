#pragma once

#include <iostream>


// KAS_FOR_EACH is adapted from Recursive macros with C++20 __VA_OPT__ by David Mazières
// https://www.scs.stanford.edu/~dm/blog/va-opt.html

#define KAS_PARENS ()

#define KAS_EXPAND(...) KAS_EXPAND4(KAS_EXPAND4(KAS_EXPAND4(KAS_EXPAND4(__VA_ARGS__))))
#define KAS_EXPAND4(...) KAS_EXPAND3(KAS_EXPAND3(KAS_EXPAND3(KAS_EXPAND3(__VA_ARGS__))))
#define KAS_EXPAND3(...) KAS_EXPAND2(KAS_EXPAND2(KAS_EXPAND2(KAS_EXPAND2(__VA_ARGS__))))
#define KAS_EXPAND2(...) KAS_EXPAND1(KAS_EXPAND1(KAS_EXPAND1(KAS_EXPAND1(__VA_ARGS__))))
#define KAS_EXPAND1(...) __VA_ARGS__

#define KAS_FOR_EACH(macro, ...) \
  __VA_OPT__(KAS_EXPAND(KAS_FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define KAS_FOR_EACH_HELPER(macro, a1, ...) \
  macro(a1) \
  __VA_OPT__(KAS_FOR_EACH_AGAIN KAS_PARENS (macro, __VA_ARGS__))
#define KAS_FOR_EACH_AGAIN() KAS_FOR_EACH_HELPER


#define KAS_STATISTICS_SINGLE(name) \
    static inline std::size_t Count##name = 0;
#define KAS_STATISTICS_PRINT_SINGLE(name) \
    os << "  " << #name << " = " << Count##name << std::endl;

#define KAS_STATISTICS_DEF(...) \
    KAS_FOR_EACH(KAS_STATISTICS_SINGLE, __VA_ARGS__) \
    static void PrintStatistics(std::ostream& os) { \
        KAS_FOR_EACH(KAS_STATISTICS_PRINT_SINGLE, __VA_ARGS__) \
    }
