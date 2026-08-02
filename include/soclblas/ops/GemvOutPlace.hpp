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
        virtual socl::DispatchToken execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) override;

        virtual socl::DispatchToken operator()(
            socl::Buffer A,
            socl::Buffer X,
            socl::Buffer Y,
            socl::Buffer outY,
            const GemvOutPlaceArguments& args
        );
        virtual socl::DispatchToken operator()(
            socl::Buffer A,
            socl::Buffer X,
            socl::Buffer Y,
            socl::Buffer outY,
            const GemvArguments& args
        );
    };
}
