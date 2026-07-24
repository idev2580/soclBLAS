#include <soclblas/ops/GemmOutPlaceNaive.hpp>
#include <GemmOutPlaceNaiveFP32_SPIRV>

namespace soclblas{
    GemmOutPlaceNaiveFP32::GemmOutPlaceNaiveFP32(
        socl::Context& ctx,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_k
    ):GemmOutPlace(
        ctx, std::span<const uint32_t>(GemmOutPlaceNaiveFP32_SPIRV), tile_m, tile_n, tile_k
    ){}
}
