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
        GemmOutPlaceArguments m_args = convertGemvToGemm(*(GemvOutPlaceArguments*)args);
        gemm.execute(inputs, inouts, outputs, &m_args, sizeof(GemmOutPlaceArguments));
    }

    void GemvOutPlace::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        socl::Buffer outC,
        const GemvOutPlaceArguments& args
    ){
        GemmOutPlaceArguments m_args = convertGemvToGemm(args);
        gemm(A,B,C,outC,m_args);
    }

    void GemvOutPlace::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        socl::Buffer outC,
        const GemvArguments& args
    ){
        const GemvOutPlaceArguments outPlaceArgs = GemvOutPlaceArguments::sameOutputLayout(args);
        (*this)(A, B, C, outC, outPlaceArgs);
    }
}
