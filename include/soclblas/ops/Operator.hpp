#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <socl/Buffer.hpp>
#include <soclblas/ExecutionPlan.hpp>

namespace soclblas{
    class Operator{
        protected:
        static std::vector<std::byte> copyPushConstants(
            const void* args,
            std::size_t argsSize
        );

        public:
        virtual ~Operator() = default;

        virtual DispatchPlan execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) = 0;
    };
}
