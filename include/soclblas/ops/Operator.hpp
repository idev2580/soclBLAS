#pragma once
#include "socl/DescriptorSet.hpp"
#include <vector>
#include <socl/Context.hpp>

namespace soclblas{
    class Operator{
        public:
        virtual void execute(
            std::span<socl::Buffer> inputs,
            std::span<socl::Buffer> inouts,
            std::span<socl::Buffer> outputs,
            const void* args,
            std::size_t argsSize
        ) = 0;
    };
}