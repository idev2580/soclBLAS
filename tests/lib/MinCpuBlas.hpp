#pragma once
#include <cstdint>
#include <vector>

// Header-only simple BLAS on CPU
// Row-major assumption for all matrices
namespace SimpleBLAS{
    void axpy(
        float alpha,
        const std::vector<float>& x,
        std::vector<float>& y
    ) {
        // y = alpha * x + y
        const size_t size = x.size();
        for (size_t i = 0; i < size; ++i) {
            y[i] += alpha * x[i];
        }
    }

    void gemv(
        float alpha,
        float beta,
        uint32_t batch,
        uint32_t m,
        uint32_t n,
        const std::vector<float>& a,
        const std::vector<float>& x,
        std::vector<float>& y
    ) {
        // 행렬 a는 배치에 무관하게 공유됨 (크기: m * n)
        // 벡터 x는 배치마다 존재 (크기: batch * n)
        // 벡터 y는 배치마다 존재 (크기: batch * m)
        
        for (uint32_t b = 0; b < batch; ++b) {
            uint32_t x_offset = b * n;
            uint32_t y_offset = b * m;

            for (uint32_t i = 0; i < m; ++i) {
                float sum = 0.0f;
                uint32_t a_row_offset = i * n;
                
                for (uint32_t j = 0; j < n; ++j) {
                    sum += a[a_row_offset + j] * x[x_offset + j];
                }

                // beta가 0일 경우, 기존 y값의 쓰레기값이나 NaN이 곱해지는 것을 방지
                if (beta == 0.0f) {
                    y[y_offset + i] = alpha * sum;
                } else {
                    y[y_offset + i] = alpha * sum + beta * y[y_offset + i];
                }
            }
        }
    }

    void gemm(
        float alpha,
        float beta,
        uint32_t batch,
        uint32_t m,
        uint32_t n,
        uint32_t p,
        const std::vector<float>& a,
        const std::vector<float>& b,
        std::vector<float>& c
    ) {
        // 행렬 a, b, c 모두 배치마다 존재
        // a 크기: batch * m * n
        // b 크기: batch * n * p
        // c 크기: batch * m * p
        
        for (uint32_t b_idx = 0; b_idx < batch; ++b_idx) {
            uint32_t a_offset = b_idx * m * n;
            uint32_t b_offset = b_idx * n * p;
            uint32_t c_offset = b_idx * m * p;

            // Cache-friendly 루프 적용 (m -> n(inner) -> p 연속 접근)
            for (uint32_t i = 0; i < m; ++i) {
                // 1. C 행렬 초기화 및 beta 곱 적용
                for (uint32_t k = 0; k < p; ++k) {
                    if (beta == 0.0f) {
                        c[c_offset + i * p + k] = 0.0f;
                    } else {
                        c[c_offset + i * p + k] *= beta;
                    }
                }
                
                // 2. A 행렬(m*n)과 B 행렬(n*p) 곱셈 누적
                for (uint32_t j = 0; j < n; ++j) {
                    float a_val = alpha * a[a_offset + i * n + j];
                    uint32_t b_row_offset = b_offset + j * p;
                    uint32_t c_row_offset = c_offset + i * p;
                    
                    // B와 C의 메모리가 연속적으로 접근되도록 가장 안쪽 루프를 p로 설정
                    for (uint32_t k = 0; k < p; ++k) {
                        c[c_row_offset + k] += a_val * b[b_row_offset + k];
                    }
                }
            }
        }
    }
}
