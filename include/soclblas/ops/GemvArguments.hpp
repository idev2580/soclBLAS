#pragma once
#include <cstdint>
#include <soclblas/ops/GemmArguments.hpp>

namespace soclblas{
    struct GemvArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        float alpha;
        float beta;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t x_n_stride;
        uint32_t x_b_stride;
        uint32_t y_m_stride;
        uint32_t y_b_stride;
    };
    GemmArguments convertGemvToGemm(const GemvArguments& args);
}