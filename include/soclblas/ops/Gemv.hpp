#pragma once
#include "soclblas/ops/Operator.hpp"
#include "soclblas/ops/Gemm.hpp"
#include <soclblas/ops/GemvArguments.hpp>
#include <cstdint>
#include <memory>

namespace soclblas{
    class Gemv: public Operator{
        protected:
        std::unique_ptr<Gemm> gemm;
        explicit Gemv(socl::Context& ctx);

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
            const GemvArguments& args
        );
    };
}
