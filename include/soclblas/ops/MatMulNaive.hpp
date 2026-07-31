#pragma once
#include "socl/Buffer.hpp"
#include "socl/Context.hpp"
#include <soclblas/ops/MatMul.hpp>
#include <MatMulNaiveFP32_SPIRV>

namespace soclblas{
    class MatMulNaiveFP32: public MatMul{
        public:
        MatMulNaiveFP32(
            socl::Context& ctx,
            uint32_t subgroup_tile_m,
            uint32_t subgroup_tile_n,
            uint32_t subgroup_tile_p,
            uint32_t subgroup_tile_cnt_m = 2,
            uint32_t subgroup_tile_cnt_p = 2,
            uint32_t shared_tile_n_multiplier = 2,
            uint32_t reg_tile_m = 8,
            uint32_t reg_tile_n = 8,
            uint32_t reg_tile_p = 8
        );
    };
}
