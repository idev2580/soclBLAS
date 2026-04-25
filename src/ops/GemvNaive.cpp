#include <ops/GemvNaive.hpp>
#include <ops/GemmNaive.hpp>

namespace kpblas{
    GemvNaiveFP32::GemvNaiveFP32(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        uint32_t b,
        uint32_t m,
        uint32_t n,
        float alpha,
        float beta,
        bool a_transposed
    ):GemmNaiveFP32(
        tensors, 
        algorithm, 
        1, 
        m, n, 
        b,  //B's column dimension should be same as the number of vector, which is given as batch.
        alpha, beta, 
        a_transposed, 
        true, // Vector will be given as column-major
        true  // Output matrix should be written in transposed way...
    ){
        //All jobs done using GEMM.
    }
}