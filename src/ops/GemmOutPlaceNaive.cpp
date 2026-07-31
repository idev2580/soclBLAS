#include <soclblas/ops/GemmOutPlaceNaive.hpp>
#include <GemmOutPlaceNaiveFP32_SPIRV>

namespace soclblas{
    GemmOutPlaceNaiveFP32::GemmOutPlaceNaiveFP32(
        socl::Context& ctx,
        uint32_t subgroup_tile_m,
        uint32_t subgroup_tile_n,
        uint32_t subgroup_tile_p,
        uint32_t subgroup_tile_cnt_m,
        uint32_t subgroup_tile_cnt_p,
        uint32_t shared_tile_n_multiplier,
        uint32_t reg_tile_m,
        uint32_t reg_tile_n,
        uint32_t reg_tile_p
    ):GemmOutPlace(
        ctx,
        std::span<const uint32_t>(GemmOutPlaceNaiveFP32_SPIRV),
        subgroup_tile_m,
        subgroup_tile_n,
        subgroup_tile_p,
        subgroup_tile_cnt_m,
        subgroup_tile_cnt_p,
        shared_tile_n_multiplier,
        reg_tile_m,
        reg_tile_n,
        reg_tile_p
    ){}
}
