#include "soclblas/ops/GemvArguments.hpp"
#include <soclblas/ops/Gemv.hpp>
#include <soclblas/ops/Gemm.hpp>

namespace soclblas{
    Gemv::Gemv(socl::Context&):gemm(nullptr){}
    socl::DispatchToken Gemv::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        GemmArguments m_args = convertGemvToGemm(*(GemvArguments*)args);
        return gemm->execute(inputs, inouts, outputs, &m_args, sizeof(GemmArguments));
    }

    socl::DispatchToken Gemv::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        const GemvArguments& args
    ){
        GemmArguments m_args = convertGemvToGemm(args);
        return (*gemm)(A,B,C,m_args);
    }
}
