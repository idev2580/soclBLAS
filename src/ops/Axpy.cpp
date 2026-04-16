#include <ops/Axpy.hpp>

namespace kpblas{
    AxpyFP32::AxpyFP32(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        const kp::Workgroup& wgrp,
        uint32_t vector_size,
        float alpha
    ):kp::OpAlgoDispatch(algorithm){
        // Initialize specialized constants
        std::vector<float> specConstant;
        specConstant.resize(2);
        uint8_t* target = (uint8_t*)specConstant.data();
        memcpy(target, &vector_size, 4);
        memcpy(target+4, &alpha, 4);
        
        algorithm->rebuild(tensors, std::vector<uint32_t>(
            std::begin(AxpyFP32_SPIRV), 
            std::end(AxpyFP32_SPIRV)
            ), 
            wgrp,
            specConstant
        );
    }
}
