#pragma once
#include <cstdint>

namespace kpblas{
    struct GemvArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        float alpha;
        float beta;
        bool a_transposed;
    };
}