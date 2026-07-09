#include <ops/GemvNaive.hpp>
#include <ops/GemmNaive.hpp>

namespace kpblas{
    GemvNaiveFP32::GemvNaiveFP32(
        std::vector<std::shared_ptr<kp::Tensor>> tensors, 
        std::shared_ptr<kp::Algorithm> algorithm,
        GemvArguments args
    ):GemmNaiveFP32(
        tensors, 
        algorithm, 
        {
            1, 
            args.m, args.n, 
            args.b,  //B's column dimension should be same as the number of vector, which is given as batch.
            args.alpha, args.beta, 
            (0x6u | (args.a_transposed ? 0x1u : 0x0u))
            // Vector will be given as column-major
            // Output matrix should be written in transposed way...
        }
    ){
        //All jobs done using GEMM.
    }
}