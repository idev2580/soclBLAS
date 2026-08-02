#pragma once
#include <soclblas/ops/GemmNaiveTemplate.hpp>
#include <soclblas/ops/Gemv.hpp>

namespace soclblas{
    class GemvNaiveTemplateFP32: public Gemv{
        public:
        GemvNaiveTemplateFP32(
            socl::Context& ctx,
            std::string_view epilogueSource,
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
