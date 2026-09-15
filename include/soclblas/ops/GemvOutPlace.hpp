#pragma once
#include "soclblas/ops/Operator.hpp"
#include "soclblas/ops/GemmOutPlace.hpp"
#include <soclblas/ops/GemvArguments.hpp>
#include <cstdint>
#include <memory>

namespace soclblas{
    class GemvOutPlace: public Operator{
        protected:
        std::unique_ptr<GemmOutPlace> gemm;
        explicit GemvOutPlace(socl::Context& ctx);

        public:
        virtual DispatchPlan execute(
            std::span<const BufferView> inputs,
            std::span<const BufferView> inouts,
            std::span<const BufferView> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual DispatchPlan operator()(
            BufferView A,
            BufferView X,
            BufferView Y,
            BufferView outY,
            const GemvOutPlaceArguments& args
        );
        virtual DispatchPlan operator()(
            BufferView A,
            BufferView X,
            BufferView Y,
            BufferView outY,
            const GemvArguments& args
        );
    };
}
