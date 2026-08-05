#pragma once
#include "socl/Buffer.hpp"
#include "soclblas/ops/Operator.hpp"
#include <cstdint>

namespace soclblas{
    struct UnaryElementwiseArguments{
        uint32_t size;
    };

    struct BinaryElementwiseArguments{
        uint32_t size;
    };

    class UnaryElementwise: public Operator{
        public:
        virtual socl::DispatchToken operator()(
            socl::Buffer a,
            socl::Buffer out,
            const UnaryElementwiseArguments& args
        ) = 0;
    };
    class BinaryElementwise: public Operator{
        public:
        virtual socl::DispatchToken operator()(
            socl::Buffer a,
            socl::Buffer b,
            socl::Buffer out,
            const BinaryElementwiseArguments& args
        ) = 0;
    };
}
