#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <socl/Context.hpp>
#include <soclblas/ops/GemmContiguousNaive.hpp>
#include <soclblas/ops/GemmNaive.hpp>
#include <soclblas/ops/GemmShared.hpp>

namespace {
    constexpr uint32_t block_m = 128;
    constexpr uint32_t block_n = 32;
    constexpr uint32_t block_p = 128;
    constexpr uint32_t thread_tile_m = 8;
    constexpr uint32_t thread_tile_p = 8;

    void fill_random(std::vector<float>& values, std::mt19937& generator) {
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for(float& value : values) {
            value = distribution(generator);
        }
    }

    testing::AssertionResult tensors_near(
        const std::vector<float>& expected,
        const std::vector<float>& actual,
        float absolute_tolerance = 2.0e-4f,
        float relative_tolerance = 2.0e-4f
    ) {
        if(expected.size() != actual.size()) {
            return testing::AssertionFailure()
                << "size mismatch: expected=" << expected.size()
                << ", actual=" << actual.size();
        }

        for(std::size_t i = 0; i < expected.size(); i++) {
            const float expected_value = expected[i];
            const float actual_value = actual[i];
            if(expected_value == actual_value) {
                continue;
            }
            if(!std::isfinite(expected_value) || !std::isfinite(actual_value)) {
                return testing::AssertionFailure()
                    << "non-finite mismatch at index " << i
                    << ": expected=" << expected_value
                    << ", actual=" << actual_value;
            }

            const float difference = std::abs(expected_value - actual_value);
            const float allowed_error =
                absolute_tolerance +
                relative_tolerance * std::max(std::abs(expected_value), std::abs(actual_value));
            if(difference > allowed_error) {
                return testing::AssertionFailure()
                    << "mismatch at index " << i
                    << ": expected=" << expected_value
                    << ", actual=" << actual_value
                    << ", absolute_error=" << difference
                    << ", allowed_error=" << allowed_error;
            }
        }

        return testing::AssertionSuccess();
    }
}

TEST(PerformanceShaderCorrectnessTest, GemmSharedMatchesNaiveForStridedEdgeTiles) {
    constexpr uint32_t batch = 2;
    constexpr uint32_t m = 137;
    constexpr uint32_t n = 37;
    constexpr uint32_t p = 143;

    // A and B use transposed, padded physical layouts. C is row-major with
    // padding so the comparison also detects writes outside logical elements.
    constexpr uint32_t a_m_stride = 1;
    constexpr uint32_t a_n_stride = m + 3;
    constexpr uint32_t a_stride = n * a_n_stride + 5;
    constexpr uint32_t b_n_stride = 1;
    constexpr uint32_t b_p_stride = n + 5;
    constexpr uint32_t b_stride = p * b_p_stride + 3;
    constexpr uint32_t c_m_stride = p + 7;
    constexpr uint32_t c_p_stride = 1;
    constexpr uint32_t c_stride = m * c_m_stride + 11;

    std::mt19937 generator(2580);
    std::vector<float> a(std::size_t(batch) * a_stride);
    std::vector<float> b(std::size_t(batch) * b_stride);
    std::vector<float> initial_c(std::size_t(batch) * c_stride);
    fill_random(a, generator);
    fill_random(b, generator);
    fill_random(initial_c, generator);

    std::vector<float> expected_c(initial_c.size());
    std::vector<float> actual_c(initial_c.size());

    socl::Context context;
    soclblas::GemmNaiveFP32 reference_gemm(context, 8, 4, 4);
    soclblas::GemmSharedFP32 tested_gemm(
        context,
        block_m,
        block_n,
        block_p,
        thread_tile_m,
        thread_tile_p
    );

    auto buffer_a = context.createBuffer(a.size() * sizeof(float), socl::BufferType::Auto);
    auto buffer_b = context.createBuffer(b.size() * sizeof(float), socl::BufferType::Auto);
    auto buffer_expected_c =
        context.createBuffer(expected_c.size() * sizeof(float), socl::BufferType::Auto);
    auto buffer_actual_c =
        context.createBuffer(actual_c.size() * sizeof(float), socl::BufferType::Auto);

    buffer_a.write(a.data(), a.size() * sizeof(float));
    buffer_b.write(b.data(), b.size() * sizeof(float));
    for(uint32_t case_index = 0; case_index < 2; case_index++) {
        const float alpha = case_index == 0 ? 1.0f : -0.75f;
        const float beta = case_index == 0 ? 0.0f : 0.25f;
        SCOPED_TRACE(testing::Message() << "alpha=" << alpha << ", beta=" << beta);

        expected_c = initial_c;
        actual_c = initial_c;
        buffer_expected_c.write(expected_c.data(), expected_c.size() * sizeof(float));
        buffer_actual_c.write(actual_c.data(), actual_c.size() * sizeof(float));

        const soclblas::GemmArguments arguments = {
            .b = batch,
            .m = m,
            .n = n,
            .p = p,
            .alpha = alpha,
            .beta = beta,
            .a_stride = a_stride,
            .b_stride = b_stride,
            .c_stride = c_stride,
            .a_m_stride = a_m_stride,
            .a_n_stride = a_n_stride,
            .b_n_stride = b_n_stride,
            .b_p_stride = b_p_stride,
            .c_m_stride = c_m_stride,
            .c_p_stride = c_p_stride
        };

        auto reference_token =
            reference_gemm(buffer_a, buffer_b, buffer_expected_c, arguments);
        reference_token.wait();
        auto tested_token = tested_gemm(buffer_a, buffer_b, buffer_actual_c, arguments);
        tested_token.wait();

        buffer_expected_c.read(expected_c.data(), expected_c.size() * sizeof(float));
        buffer_actual_c.read(actual_c.data(), actual_c.size() * sizeof(float));

        EXPECT_TRUE(tensors_near(expected_c, actual_c));
    }
}

TEST(PerformanceShaderCorrectnessTest, GemmContiguousMatchesNaiveForAllLayouts) {
    struct TestConfig {
        uint32_t block_m;
        uint32_t block_n;
        uint32_t block_p;
        uint32_t thread_tile_m;
        uint32_t thread_tile_p;
    };
    constexpr TestConfig configs[] = {
        {128, 32, 128, 8, 8},
        {128, 16, 128, 16, 4},
        {128, 16, 128, 4, 16},
        {128, 16, 256, 8, 16},
        {64, 16, 64, 8, 2},
        {256, 24, 128, 16, 8},
    };
    constexpr uint32_t batch = 2;
    constexpr uint32_t m = 256;
    constexpr uint32_t n = 96;
    constexpr uint32_t p = 256;
    constexpr uint32_t a_stride = m * n;
    constexpr uint32_t b_stride = n * p;
    constexpr uint32_t c_stride = m * p;

    socl::Context context;
    soclblas::GemmNaiveFP32 reference_gemm(context, 8, 4, 4);

    std::vector<float> a(std::size_t(batch) * a_stride);
    std::vector<float> b(std::size_t(batch) * b_stride);
    std::vector<float> initial_c(std::size_t(batch) * c_stride);
    std::vector<float> expected_c(initial_c.size());
    std::vector<float> actual_c(initial_c.size());

    auto buffer_a = context.createBuffer(a.size() * sizeof(float), socl::BufferType::Auto);
    auto buffer_b = context.createBuffer(b.size() * sizeof(float), socl::BufferType::Auto);
    auto buffer_expected_c =
        context.createBuffer(expected_c.size() * sizeof(float), socl::BufferType::Auto);
    auto buffer_actual_c =
        context.createBuffer(actual_c.size() * sizeof(float), socl::BufferType::Auto);

    std::mt19937 generator(2580);
    for(const TestConfig& config : configs) {
        SCOPED_TRACE(
            testing::Message()
                << "block=" << config.block_m << "x" << config.block_n << "x" << config.block_p
                << ", thread_tile=" << config.thread_tile_m << "x" << config.thread_tile_p
        );
        soclblas::GemmContiguousNaiveFP32 tested_gemm(
            context,
            config.block_m,
            config.block_n,
            config.block_p,
            config.thread_tile_m,
            config.thread_tile_p
        );

        for(uint32_t flags = 0; flags <= soclblas::GemmContiguousFlagsMask; flags++) {
            const bool a_transposed = (flags & soclblas::GemmContiguousFlagATransposed) != 0;
            const bool b_transposed = (flags & soclblas::GemmContiguousFlagBTransposed) != 0;
            const bool c_transposed = (flags & soclblas::GemmContiguousFlagCTransposed) != 0;
            SCOPED_TRACE(
                testing::Message()
                    << "flags=" << flags
                    << ", A_transposed=" << a_transposed
                    << ", B_transposed=" << b_transposed
                    << ", C_transposed=" << c_transposed
            );

            fill_random(a, generator);
            fill_random(b, generator);
            fill_random(initial_c, generator);
            expected_c = initial_c;
            actual_c = initial_c;

            buffer_a.write(a.data(), a.size() * sizeof(float));
            buffer_b.write(b.data(), b.size() * sizeof(float));
            buffer_expected_c.write(expected_c.data(), expected_c.size() * sizeof(float));
            buffer_actual_c.write(actual_c.data(), actual_c.size() * sizeof(float));

            const float alpha = a_transposed ? 0.75f : 1.0f;
            const float beta = b_transposed ? -0.25f : 0.0f;
            const soclblas::GemmArguments reference_arguments = {
                .b = batch,
                .m = m,
                .n = n,
                .p = p,
                .alpha = alpha,
                .beta = beta,
                .a_stride = a_stride,
                .b_stride = b_stride,
                .c_stride = c_stride,
                .a_m_stride = a_transposed ? 1u : n,
                .a_n_stride = a_transposed ? m : 1u,
                .b_n_stride = b_transposed ? 1u : p,
                .b_p_stride = b_transposed ? n : 1u,
                .c_m_stride = c_transposed ? 1u : p,
                .c_p_stride = c_transposed ? m : 1u
            };
            const soclblas::GemmContiguousArguments tested_arguments = {
                .b = batch,
                .m = m,
                .n = n,
                .p = p,
                .alpha = alpha,
                .beta = beta,
                .a_stride = a_stride,
                .b_stride = b_stride,
                .c_stride = c_stride,
                .a_major_stride = a_transposed ? m : n,
                .b_major_stride = b_transposed ? n : p,
                .c_major_stride = c_transposed ? m : p,
                .flags = flags
            };

            auto reference_token =
                reference_gemm(buffer_a, buffer_b, buffer_expected_c, reference_arguments);
            reference_token.wait();
            auto tested_token = tested_gemm(buffer_a, buffer_b, buffer_actual_c, tested_arguments);
            tested_token.wait();

            buffer_expected_c.read(expected_c.data(), expected_c.size() * sizeof(float));
            buffer_actual_c.read(actual_c.data(), actual_c.size() * sizeof(float));

            EXPECT_TRUE(tensors_near(expected_c, actual_c));
        }
    }
}
