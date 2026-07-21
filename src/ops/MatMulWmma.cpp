#include <soclblas/ops/MatMulWmma.hpp>

namespace soclblas{
    MatMulWmmaFP32::MatMulWmmaFP32(
        socl::Context& ctx,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_k
    ):MatMul(
        ctx, std::span<const uint32_t>(MatMulWmmaFP32_SPIRV), tile_m, tile_n, tile_k
    ){}
}