#pragma once
#include <kompute/operations/OpAlgoDispatch.hpp>
#include <kompute/Kompute.hpp>

#include <AxpyFP32_SPIRV>
namespace kpblas{
    class AxpyFP32: public kp::OpAlgoDispatch{
        public:
        AxpyFP32(
            std::vector<std::shared_ptr<kp::Tensor>> tensors, 
            std::shared_ptr<kp::Algorithm> algorithm,
            const kp::Workgroup& wgrp,
            uint32_t vector_size,
            float alpha
        );
    };
}