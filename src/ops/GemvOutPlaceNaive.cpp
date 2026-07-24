#include <soclblas/ops/GemvOutPlaceNaive.hpp>
#include <GemmOutPlaceNaiveFP32_SPIRV>

namespace soclblas{
    GemvOutPlaceNaiveFP32::GemvOutPlaceNaiveFP32(
        socl::Context& ctx,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_k
    ):GemvOutPlace(
        ctx, std::span<const uint32_t>(GemmOutPlaceNaiveFP32_SPIRV), tile_m, tile_n, tile_k
    ){}
}
