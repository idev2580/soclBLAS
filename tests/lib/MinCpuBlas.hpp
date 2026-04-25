#pragma once
#include <cstdint>
#include <vector>

// Header-only simple BLAS on CPU
// Row-major assumption for all matrices
namespace SimpleBLAS {

    void axpy(
        float alpha,
        const std::vector<float>& x,
        std::vector<float>& y
    ) {
        const size_t size = x.size();
        for (size_t i = 0; i < size; ++i) {
            y[i] += alpha * x[i];
        }
    }

    void gemv(
        bool transA,
        float alpha,
        float beta,
        uint32_t batch,
        uint32_t m,
        uint32_t n,
        const std::vector<float>& a,
        const std::vector<float>& x,
        std::vector<float>& y
    ) {
        // transA = false: A(m*n), x(n), y(m)
        // transA = true : A^T(n*m), x(m), y(n)
        
        uint32_t out_dim = transA ? n : m;
        uint32_t in_dim  = transA ? m : n;

        for (uint32_t b = 0; b < batch; ++b) {
            uint32_t x_offset = b * in_dim;
            uint32_t y_offset = b * out_dim;

            // 1. 초기화 및 beta 적용
            for (uint32_t i = 0; i < out_dim; ++i) {
                if (beta == 0.0f) {
                    y[y_offset + i] = 0.0f;
                } else {
                    y[y_offset + i] *= beta;
                }
            }

            if (!transA) {
                // Normal A: Row-major dot product
                for (uint32_t i = 0; i < m; ++i) {
                    float sum = 0.0f;
                    uint32_t a_row_offset = i * n;
                    for (uint32_t j = 0; j < n; ++j) {
                        sum += a[a_row_offset + j] * x[x_offset + j];
                    }
                    y[y_offset + i] += alpha * sum;
                }
            } else {
                // Transposed A (A^T): A의 행을 순차적으로 읽어 y에 분배 (Cache-friendly)
                for (uint32_t i = 0; i < m; ++i) {
                    float x_val = alpha * x[x_offset + i];
                    uint32_t a_row_offset = i * n;
                    for (uint32_t j = 0; j < n; ++j) {
                        y[y_offset + j] += x_val * a[a_row_offset + j];
                    }
                }
            }
        }
    }

    void gemm(
        bool transA,
        bool transB,
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
        // C(m*p) = alpha * op(A) * op(B) + beta * C
        // !transA: A(m*n), transA: A(n*m)
        // !transB: B(n*p), transB: B(p*n)
        
        uint32_t a_stride = m * n;
        uint32_t b_stride = n * p;
        uint32_t c_stride = m * p;

        for (uint32_t b_idx = 0; b_idx < batch; ++b_idx) {
            uint32_t a_offset = b_idx * a_stride;
            uint32_t b_offset = b_idx * b_stride;
            uint32_t c_offset = b_idx * c_stride;

            // 1. C 행렬 beta 적용 및 초기화
            for (uint32_t i = 0; i < m * p; ++i) {
                if (beta == 0.0f) {
                    c[c_offset + i] = 0.0f;
                } else {
                    c[c_offset + i] *= beta;
                }
            }

            // 2. 연산 루프 (Transpose 조합에 따른 최적화)
            if (!transA && !transB) {
                // Case: A * B
                for (uint32_t i = 0; i < m; ++i) {
                    for (uint32_t j = 0; j < n; ++j) {
                        float a_val = alpha * a[a_offset + i * n + j];
                        for (uint32_t k = 0; k < p; ++k) {
                            c[c_offset + i * p + k] += a_val * b[b_offset + j * p + k];
                        }
                    }
                }
            } 
            else if (transA && !transB) {
                // Case: A^T * B
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t i = 0; i < m; ++i) {
                        float a_val = alpha * a[a_offset + j * m + i];
                        for (uint32_t k = 0; k < p; ++k) {
                            c[c_offset + i * p + k] += a_val * b[b_offset + j * p + k];
                        }
                    }
                }
            }
            else if (!transA && transB) {
                // Case: A * B^T
                for (uint32_t i = 0; i < m; ++i) {
                    for (uint32_t k = 0; k < p; ++k) {
                        float sum = 0.0f;
                        for (uint32_t j = 0; j < n; ++j) {
                            sum += a[a_offset + i * n + j] * b[b_offset + k * n + j];
                        }
                        c[c_offset + i * p + k] += alpha * sum;
                    }
                }
            }
            else {
                // Case: A^T * B^T
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t i = 0; i < m; ++i) {
                        float a_val = alpha * a[a_offset + j * m + i];
                        for (uint32_t k = 0; k < p; ++k) {
                            c[c_offset + i * p + k] += a_val * b[b_offset + k * n + j];
                        }
                    }
                }
            }
        }
    }
}