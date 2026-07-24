#include "soclblas/ops/GemvArguments.hpp"
#include <soclblas/ops/GemvOutPlace.hpp>
#include <soclblas/ops/GemmOutPlace.hpp>

namespace soclblas{
    GemvOutPlace::GemvOutPlace(
        socl::Context& ctx,
        std::span<const uint32_t> gemmShaderBytecodes,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_p
    ):gemm(
        ctx, gemmShaderBytecodes, tile_m, tile_n, tile_p
    ){}
    void GemvOutPlace::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        GemmArguments m_args = convertGemvToGemm(*(GemvArguments*)args);
        gemm.execute(inputs, inouts, outputs, &m_args, sizeof(GemmArguments));
    }

    void GemvOutPlace::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        socl::Buffer outC,
        const GemvArguments& args
    ){
        GemmArguments m_args = convertGemvToGemm(args);
        gemm(A,B,C,outC,m_args);
    }
}