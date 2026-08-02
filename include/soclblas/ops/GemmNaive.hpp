#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/Gemm.hpp>

namespace soclblas{
    class GemmNaiveFP32: public Gemm{
        public:
        GemmNaiveFP32(
            socl::Context& ctx,
            uint32_t subgroup_tile_m,
            uint32_t subgroup_tile_n,
            uint32_t subgroup_tile_p,
            uint32_t subgroup_tile_cnt_m,
            uint32_t subgroup_tile_cnt_p,
            uint32_t shared_tile_n_multiplier,
            uint32_t reg_tile_m,
            uint32_t k_unroll,
            uint32_t reg_tile_p
        );
    };
}
