#pragma once
#include <soclblas/ops/GemmOutPlace.hpp>

namespace soclblas{
    class GemmOutPlaceNaiveFP32: public GemmOutPlace{
        public:
        GemmOutPlaceNaiveFP32(
            socl::Context& ctx,
            uint32_t tile_m,
            uint32_t tile_n,
            uint32_t tile_k
        );
    };
}
