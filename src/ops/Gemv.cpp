#include "soclblas/ops/GemvArguments.hpp"
#include <soclblas/ops/Gemv.hpp>
#include <soclblas/ops/Gemm.hpp>

namespace soclblas{
    Gemv::Gemv(socl::Context&):gemm(nullptr){}
    DispatchPlan Gemv::execute(
        std::span<const BufferView> inputs,
        std::span<const BufferView> inouts,
        std::span<const BufferView> outputs,
        const void* args,
        std::size_t argsSize
    ){
        GemmArguments m_args = convertGemvToGemm(*(GemvArguments*)args);
        return gemm->execute(inputs, inouts, outputs, &m_args, sizeof(GemmArguments));
    }

    DispatchPlan Gemv::operator()(
        BufferView A,
        BufferView B,
        BufferView C,
        const GemvArguments& args
    ){
        GemmArguments m_args = convertGemvToGemm(args);
        return (*gemm)(A,B,C,m_args);
    }
}
