#include <soclblas/ops/GemmNaive.hpp>

namespace soclblas{
    GemmNaiveFP32::GemmNaiveFP32(
        socl::Context& ctx,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_k
    ):Gemm(
        ctx, std::span<const uint32_t>(GemmNaiveFP32_SPIRV), tile_m, tile_n, tile_k
    ){}
}
