#include <soclblas/ops/Gemv.hpp>
#include <soclblas/ops/Gemm.hpp>

namespace soclblas{
    Gemv::Gemv(
        socl::Context& ctx,
        std::span<const uint32_t> gemmShaderBytecodes,
        uint32_t tile_m,
        uint32_t tile_n,
        uint32_t tile_p
    ):gemm(
        ctx, gemmShaderBytecodes, tile_m, tile_n, tile_p
    ){}
    GemmArguments convertArgs(const GemvArguments args){
        //TODO: Fill GemmArguments from GemvArguments
        return {
            .b = 1,
            .m = args.m,
            .n = args.n,
            .p = args.b,
            .alpha = args.alpha,
            .beta = args.beta,
            .a_m_stride = args.a_m_stride,
            .a_n_stride = args.a_n_stride,
            .b_n_stride = args.x_n_stride,
            .b_p_stride = args.x_b_stride,
            .c_m_stride = args.y_m_stride,
            .c_p_stride = args.y_b_stride
        };
    }
    void Gemv::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        GemmArguments m_args = convertArgs(*(GemvArguments*)args);
        gemm.execute(inputs, inouts, outputs, &m_args, sizeof(GemmArguments));
    }

    void Gemv::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        const GemvArguments& args
    ){
        GemmArguments m_args = convertArgs(args);
        gemm(A,B,C,m_args);
    }
}