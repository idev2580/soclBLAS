#pragma once
#include <cstdint>

namespace soclblas{
    struct GemmArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
        float alpha;
        float beta;
        uint32_t a_stride;
        uint32_t b_stride;
        uint32_t c_stride;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t b_n_stride;
        uint32_t b_p_stride;
        uint32_t c_m_stride;
        uint32_t c_p_stride;
    };
    constexpr uint32_t GemmContiguousFlagATransposed = 1u << 0;
    constexpr uint32_t GemmContiguousFlagBTransposed = 1u << 1;
    constexpr uint32_t GemmContiguousFlagCTransposed = 1u << 2;
    constexpr uint32_t GemmContiguousFlagsMask =
        GemmContiguousFlagATransposed |
        GemmContiguousFlagBTransposed |
        GemmContiguousFlagCTransposed;

    struct GemmContiguousArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
        float alpha;
        float beta;
        // Strides are expressed in float elements. The contiguous shader reads
        // A/B/C as vec4, so matrix and major strides must be 4-float aligned.
        // m/n/p must also be divisible by the configured block sizes.
        uint32_t a_stride;
        uint32_t b_stride;
        uint32_t c_stride;
        uint32_t a_major_stride;
        uint32_t b_major_stride;
        uint32_t c_major_stride;
        uint32_t flags;
    };

    struct GemmOutPlaceArguments{
        uint32_t b;
        uint32_t m;
        uint32_t n;
        uint32_t p;
        float alpha;
        float beta;
        uint32_t a_stride;
        uint32_t b_stride;
        uint32_t c_stride;
        uint32_t a_m_stride;
        uint32_t a_n_stride;
        uint32_t b_n_stride;
        uint32_t b_p_stride;
        uint32_t c_m_stride;
        uint32_t c_p_stride;
        uint32_t out_c_stride;
        uint32_t out_c_m_stride;
        uint32_t out_c_p_stride;

        static GemmOutPlaceArguments sameOutputLayout(const GemmArguments& args);
        void fromInPlace(const GemmArguments& args);
    };
}
