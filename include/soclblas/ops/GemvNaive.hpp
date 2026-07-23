#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/Gemv.hpp>

namespace soclblas{
    class GemvNaiveFP32: public Gemv{
        public:
        GemvNaiveFP32(
            socl::Context& ctx,
            uint32_t tile_m,
            uint32_t tile_n,
            uint32_t tile_k
        );
    };
}
