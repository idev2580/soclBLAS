#include <soclblas/ops/GemvNaive.hpp>
#include <GemmNaiveFP32_SPIRV>

namespace soclblas{
    GemvNaiveFP32::GemvNaiveFP32(
        socl::Context& ctx,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_k
    ):Gemv(
        ctx, std::span<const uint32_t>(GemmNaiveFP32_SPIRV), tile_m, tile_n, tile_k
    ){}
}
