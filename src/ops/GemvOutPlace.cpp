#include "soclblas/ops/GemvArguments.hpp"
#include <soclblas/ops/GemvOutPlace.hpp>
#include <soclblas/ops/GemmOutPlace.hpp>

namespace soclblas{
    GemvOutPlace::GemvOutPlace(socl::Context&):gemm(nullptr){}
    DispatchPlan GemvOutPlace::execute(
        std::span<socl::Buffer> inputs,
        std::span<socl::Buffer> inouts,
        std::span<socl::Buffer> outputs,
        const void* args,
        std::size_t argsSize
    ){
        GemmOutPlaceArguments m_args = convertGemvToGemm(*(GemvOutPlaceArguments*)args);
        return gemm->execute(
            inputs,
            inouts,
            outputs,
            &m_args,
            sizeof(GemmOutPlaceArguments)
        );
    }

    DispatchPlan GemvOutPlace::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        socl::Buffer outC,
        const GemvOutPlaceArguments& args
    ){
        GemmOutPlaceArguments m_args = convertGemvToGemm(args);
        return (*gemm)(A,B,C,outC,m_args);
    }

    DispatchPlan GemvOutPlace::operator()(
        socl::Buffer A,
        socl::Buffer B,
        socl::Buffer C,
        socl::Buffer outC,
        const GemvArguments& args
    ){
        const GemvOutPlaceArguments outPlaceArgs = GemvOutPlaceArguments::sameOutputLayout(args);
        return (*this)(A, B, C, outC, outPlaceArgs);
    }
}
