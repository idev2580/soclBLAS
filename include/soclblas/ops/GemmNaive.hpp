#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/Gemm.hpp>
#include <GemmNaiveFP32_SPIRV>

namespace soclblas{
    class GemmNaiveFP32: public Gemm{
        public:
        GemmNaiveFP32(
            socl::Context& ctx,
            uint32_t tile_m,
            uint32_t tile_n,
            uint32_t tile_k
        );
    };
}
