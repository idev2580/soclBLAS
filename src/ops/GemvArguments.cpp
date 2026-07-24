#include <soclblas/ops/GemvArguments.hpp>

namespace soclblas{
    GemmArguments convertGemvToGemm(const GemvArguments& args){
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
}