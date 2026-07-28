#pragma once
#include "socl/Buffer.hpp"
#include "soclblas/ops/Operator.hpp"
// TODO: For broadcasted operations, elementwise operators should also need to support strided operation.

namespace soclblas{
    struct UnaryElementwiseArguments{

    };

    struct BinaryElementwiseArguments{
        
    };

    class UnaryElementwise: public Operator{
        public:
        virtual void operator()(
            socl::Buffer a,
            socl::Buffer out,
            const UnaryElementwiseArguments& args
        ) = 0;
    };
    class BinaryElementwise: public Operator{
        public:
        virtual void operator()(
            socl::Buffer a,
            socl::Buffer b,
            socl::Buffer out,
            const BinaryElementwiseArguments& args
        ) = 0;
    };
}