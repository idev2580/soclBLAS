#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/GemvOutPlace.hpp>

namespace soclblas{
    class GemvOutPlaceNaiveFP32: public GemvOutPlace{
        public:
        GemvOutPlaceNaiveFP32(
            socl::Context& ctx,
            uint32_t tile_m,
            uint32_t tile_n,
            uint32_t tile_k
        );
    };
}
