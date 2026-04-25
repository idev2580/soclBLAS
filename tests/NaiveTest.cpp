#include <iostream>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <ops/Axpy.hpp>
#include <ops/GemvNaive.hpp>
#include <ops/GemmNaive.hpp>

#include "lib/MinCpuBlas.hpp"

constexpr int axpy_test_iter = 10;
constexpr int gemv_test_iter = 10;
constexpr int gemm_test_iter = 10;

constexpr int max_batch = 2;
constexpr int max_m = 1000;
constexpr int max_n = 1000;
constexpr int max_p = 1000;

bool is_equal_tensor(
    std::shared_ptr<kp::TensorT<float>> y,
    const std::vector<float>& sol_y,
    const float eps = 0.00001 // 10^-5까지는 정상으로 본다(합의 순서가 CPU와 많이 다르다.)
){
    for(int64_t i=0; i<sol_y.size(); i++){
        const float pos_delta = std::abs(sol_y[i] - y->data()[i]);
        const float cpu_abs = std::abs(sol_y[i]);
        const float error_rel = pos_delta / cpu_abs;

        if(error_rel > eps){
            printf("[%d]: sol_y=%.15f, y=%.15f, RelErr = %.15f\n", (int)i, sol_y[i], y->data()[i], error_rel);
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
void print_vector(
    const std::shared_ptr<kp::TensorT<float>>& v
){
    printf("[");
    for(int i=0; i<v->size(); i++){
        if(i == v->size() - 1){
            printf("%f", v->data()[i]);
            break;
        } else {
            printf("%f, ", v->data()[i]);
        }
    }
    printf("]\n");
}

bool axpy_test(
    float alpha,
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& sol
){
    kp::Manager* mgr = new kp::Manager();

    // Declare tensor
    auto x_tensor = mgr->tensor(x);
    auto y_tensor = mgr->tensor(y);

    // Copy to GPU
    auto seq = mgr->sequence()
    ->record<kp::OpTensorSyncDevice>({x_tensor, y_tensor})
    ->record<kpblas::AxpyFP32>(
        {x_tensor,y_tensor}, 
        mgr->algorithm(), 
        (uint32_t)(x.size()), 
        alpha
    )
    ->record<kp::OpTensorSyncLocal>({x_tensor, y_tensor});

    // Dispatch!
    seq->eval();

    const bool is_correct = is_equal_tensor(y_tensor, sol);
    delete mgr;
    return is_correct;
}

bool gemv_test(
    float alpha,
    float beta,
    uint32_t batch,
    uint32_t m,
    uint32_t n,
    const std::vector<float>& a,
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& sol
){
    kp::Manager* mgr = new kp::Manager();

    // Declare tensor
    auto a_tensor = mgr->tensor(a);
    auto x_tensor = mgr->tensor(x);
    auto y_tensor = mgr->tensor(y);

    // Copy to GPU
    auto seq = mgr->sequence()->record<kp::OpTensorSyncDevice>({a_tensor, x_tensor, y_tensor})
    ->record<kpblas::GemvNaiveFP32>(
        {a_tensor, x_tensor,y_tensor}, mgr->algorithm(), batch, m, n, alpha, beta
    )
    ->record<kp::OpTensorSyncLocal>({a_tensor, x_tensor, y_tensor});

    // Dispatch!
    seq->eval();

    const bool is_correct = is_equal_tensor(y_tensor, sol);
    delete mgr;
    return is_correct;
}

bool gemm_test(
    bool is_a_trans,
    bool is_b_trans,
    float alpha,
    float beta,
    uint32_t batch,
    uint32_t m,
    uint32_t n,
    uint32_t p,
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c,
    const std::vector<float>& sol
){
    kp::Manager* mgr = new kp::Manager();

    // Declare tensor
    auto a_tensor = mgr->tensor(a);
    auto b_tensor = mgr->tensor(b);
    auto c_tensor = mgr->tensor(c);

    // Copy to GPU
    auto seq = mgr->sequence()->record<kp::OpTensorSyncDevice>({a_tensor, b_tensor, c_tensor})
    ->record<kpblas::GemmNaiveFP32>(
        {a_tensor, b_tensor,c_tensor}, 
        mgr->algorithm(), 
        batch, m, n, p, alpha, beta,
         is_a_trans, is_b_trans
    )
    ->record<kp::OpTensorSyncLocal>({a_tensor, b_tensor, c_tensor});

    // Dispatch!
    seq->eval();

    // printf("[C](GPU, Solved)\n");
    // print_vector(c_tensor);
    const bool is_correct = is_equal_tensor(c_tensor, sol);
    delete mgr;
    return is_correct;
}

TEST(AXPYTest, BasicAssertion){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i<axpy_test_iter; i++){
        // Generate Random vectors and alpha value
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint64_t vector_size = batch * m;
        const float alpha = f_dis(gen);
        // std::cout << "Vector Size = " << vector_size << std::endl;

        std::vector<float> x(vector_size, 0.0);
        std::vector<float> y(vector_size, 0.0);
        std::vector<float> cpu_y(vector_size, 0.0);

        for(uint64_t j=0; j<vector_size; j++){
            const float rx = f_dis(gen);
            const float ry = f_dis(gen);

            x[j] = rx;
            y[j] = ry;
            cpu_y[j] = ry;
        }
        // Run on CPU
        SimpleBLAS::axpy(alpha, x, cpu_y);
        // Test!
        bool result = axpy_test(alpha, x, y, cpu_y);
        EXPECT_TRUE(result);
    }
}

TEST(GEMVTest, BasicAssertion){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> b_dis(1, max_batch);
    std::uniform_int_distribution<int> m_dis(1, max_m);
    std::uniform_int_distribution<int> n_dis(1, max_n);
    std::uniform_real_distribution<float> f_dis(0.0, 1.0);
    for(int i=0; i < gemv_test_iter; i++){
        // Generate Random vectors and alpha value
        uint32_t batch = b_dis(gen);
        uint32_t m = m_dis(gen);
        uint32_t n = n_dis(gen);
        uint64_t matrix_size = m * n;
        const float alpha = f_dis(gen);
        const float beta = f_dis(gen);

        std::vector<float> a(matrix_size, 0.0);
        std::vector<float> x(n * batch, 0.0);
        std::vector<float> y(batch * m, 0.0);
        std::vector<float> cpu_y(batch * m, 0.0);

        for(uint64_t j=0; j < matrix_size; j++){
            a[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * n; j++){
            x[j] = f_dis(gen);
        }
        for(uint64_t j=0; j < batch * m; j++){
            const float ry = f_dis(gen);
            y[j] = ry;
            cpu_y[j] = ry;
        }
        // Run on CPU
        SimpleBLAS::gemv(false, alpha, beta, batch, m, n, a, x, cpu_y);

        // Test!
        bool gemv_res = gemv_test(alpha, beta, batch, m, n, a, x, y, cpu_y);
        EXPECT_TRUE(gemv_res);
        if(!gemv_res){
            break;
        }
    }
}

TEST(GEMMTest, BasicAssertion){
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
        uint32_t p = n_dis(gen);
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

        // Test!
        bool gemm_res = gemm_test(is_a_trans, is_b_trans, alpha, beta, batch, m, n, p, a, b, c, cpu_c);
        EXPECT_TRUE(gemm_res);
        if(!gemm_res){
            printf("Transposed(A:%d, B:%d)\n", is_a_trans ? 1 : 0, is_b_trans ? 1 : 0);
            break;
        }
    }
}