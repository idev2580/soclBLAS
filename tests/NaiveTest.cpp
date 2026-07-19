#include <iostream>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <soclblas/ops/GemmNaive.hpp>
#include "lib/MinCpuBlas.hpp"

constexpr int gemm_test_iter = 10;
constexpr int max_batch = 2;
constexpr int max_m = 1000;
constexpr int max_n = 1000;
constexpr int max_p = 1000;

bool is_equal_tensor(
    const std::vector<float>& y,
    const std::vector<float>& sol_y,
    const float eps = 0.00001 // 10^-5까지는 정상으로 본다(합의 순서가 CPU와 많이 다르다.)
){
    for(int64_t i=0; i<sol_y.size(); i++){
        const float pos_delta = std::abs(sol_y[i] - y[i]);
        const float cpu_abs = std::abs(sol_y[i]);
        const float error_rel = pos_delta / cpu_abs;

        if(error_rel > eps){
            printf("[%d]: sol_y=%.15f, y=%.15f, RelErr = %.15f\n", (int)i, sol_y[i], y[i], error_rel);
            return false;
        }
    }
    return true;
}
void print_vector(
    const std::vector<float>& v
){
    printf("[");
    for(int i=0; i<v.size(); i++){
        if(i == v.size() - 1){
            printf("%f", v[i]);
            break;
        } else {
            printf("%f, ", v[i]);
        }
    }
    printf("]\n");
}

TEST(GEMMTest, BasicAssertion){
    socl::Context ctx;
    ctx.printGpuInfo(std::cout);
    soclblas::GemmNaiveFP32 gemm(ctx, 8, 4, 4);
    auto bufferA = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_n, socl::BufferType::HostVisible);
    auto bufferB = ctx.createBuffer(sizeof(float) * max_batch * max_n * max_p, socl::BufferType::HostVisible);
    auto bufferC = ctx.createBuffer(sizeof(float) * max_batch * max_m * max_p, socl::BufferType::HostVisible);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_int_distribution<int> n_dis(1, max_n);
    std::uniform_int_distribution<int> p_dis(1, max_p);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i < gemm_test_iter; i++){
        // Generate Random vectors and alpha value
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint32_t n = n_dis(gen);
        uint32_t p = p_dis(gen);
        // batch = 1;
        // m = 4;
        // n = 4;
        // p = 4;
        float alpha = f_dis(gen);
        float beta = f_dis(gen);
        // alpha = 1.0;
        // beta = 0.0;

        std::vector<float> a(batch * m * n, 0.0);
        std::vector<float> b(batch * n * p, 0.0);
        std::vector<float> c(batch * m * p, 0.0);
        std::vector<float> cpu_c(batch * m * p, 0.0);

        const bool is_a_trans = f_dis(gen) > 0.5f;
        const bool is_b_trans = f_dis(gen) > 0.5f;

        for(uint64_t j=0; j < batch * m * n; j++){
            a[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * n * p; j++){
            b[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * m * p; j++){
            float rc = f_dis(gen);
            rc = 0.0f;
            c[j] = rc;
            cpu_c[j] = rc;
        }
        // printf("m=%d, n=%d, p=%d\n", m, n, p);
        // printf("[A]\n");
        // print_vector(a);
        // printf("[B]\n");
        // print_vector(b);
        // printf("[C]\n");
        // print_vector(c);
        // Run on CPU
        SimpleBLAS::gemm(is_a_trans, is_b_trans, alpha, beta, batch, m, n, p, a, b, cpu_c);
        // printf("[CPU_C](after solve)\n");
        // print_vector(cpu_c);

        bufferA.write(a.data(), sizeof(float) * batch * m * n);
        bufferB.write(b.data(), sizeof(float) * batch * n * p);
        bufferC.write(c.data(), sizeof(float) * batch * m * p);
        soclblas::GemmArguments gemm_args = {
            .b = batch,
            .m = m,
            .n = n,
            .p = p,
            .alpha = alpha,
            .beta = beta,
            .flags = 0
        };
        gemm_args.setFlags(is_a_trans, is_b_trans, false);
        gemm(bufferA, bufferB, bufferC, gemm_args);
        bufferC.read(c.data(), sizeof(float) * batch * m * p);
        // Test!
        const bool gemm_res = is_equal_tensor(c, cpu_c);
        EXPECT_TRUE(gemm_res) << "GEMM result is not equal to CPU result!";
        if(!gemm_res){
            printf("Transposed(A:%d, B:%d)\n", is_a_trans ? 1 : 0, is_b_trans ? 1 : 0);
            break;
        }
    }
}
