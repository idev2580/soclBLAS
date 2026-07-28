#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <socl/Context.hpp>
#include <soclblas/ops/GemmShared.hpp>

namespace {
    struct GemmPerfConfig {
        uint32_t batch = 4096;
        uint32_t m = 256;
        uint32_t n = 256;
        uint32_t p = 256;
        uint32_t tile_m = 64;
        uint32_t tile_n = 16;
        uint32_t tile_p = 64;
        uint32_t thread_tile_m = 4;
        uint32_t thread_tile_p = 4;
        uint32_t seed = 2580;
    };

    uint32_t parse_u32_arg(
        int argc,
        char** argv,
        int index,
        uint32_t default_value,
        const char* name
    ) {
        if(index >= argc) {
            return default_value;
        }

        const unsigned long value = std::stoul(argv[index]);
        if(value == 0 || value > std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range(std::string(name) + " must fit in uint32_t and be greater than zero");
        }
        return static_cast<uint32_t>(value);
    }

    GemmPerfConfig parse_config(int argc, char** argv) {
        GemmPerfConfig config;
        config.batch = parse_u32_arg(argc, argv, 1, config.batch, "batch");
        config.m = parse_u32_arg(argc, argv, 2, config.m, "m");
        config.n = parse_u32_arg(argc, argv, 3, config.n, "n");
        config.p = parse_u32_arg(argc, argv, 4, config.p, "p");
        config.tile_m = parse_u32_arg(argc, argv, 5, config.tile_m, "tile_m");
        config.tile_n = parse_u32_arg(argc, argv, 6, config.tile_n, "tile_n");
        config.tile_p = parse_u32_arg(argc, argv, 7, config.tile_p, "tile_p");
        config.thread_tile_m = parse_u32_arg(argc, argv, 8, config.thread_tile_m, "thread_tile_m");
        config.thread_tile_p = parse_u32_arg(argc, argv, 9, config.thread_tile_p, "thread_tile_p");
        return config;
    }

    void validate_tile_config(const GemmPerfConfig& config) {
        if(config.thread_tile_m > 8 || config.thread_tile_p > 8) {
            throw std::out_of_range("thread_tile_m and thread_tile_p must be less than or equal to 8");
        }

        if(config.tile_m % config.thread_tile_m != 0 || config.tile_p % config.thread_tile_p != 0) {
            throw std::out_of_range("tile_m and tile_p must be divisible by thread tile sizes");
        }

        const uint64_t workgroup_threads =
            uint64_t(config.tile_m / config.thread_tile_m) *
            uint64_t(config.tile_p / config.thread_tile_p);
        if(workgroup_threads > 256) {
            throw std::out_of_range("(tile_m / thread_tile_m) * (tile_p / thread_tile_p) must be less than or equal to 256");
        }

        const uint64_t shared_elements =
            uint64_t(config.tile_m) * uint64_t(config.tile_n) +
            uint64_t(config.tile_n) * uint64_t(config.tile_p);
        const uint64_t shared_bytes = shared_elements * sizeof(float);
        if(shared_bytes > 32ull * 1024ull) {
            throw std::out_of_range("shared memory tile must be less than or equal to 32 KiB");
        }
    }

    uint64_t matrix_elements(uint32_t batch, uint32_t rows, uint32_t cols) {
        return uint64_t(batch) * uint64_t(rows) * uint64_t(cols);
    }

    size_t checked_bytes(uint64_t elements) {
        if(elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
            throw std::overflow_error("matrix allocation size exceeds size_t");
        }
        return static_cast<size_t>(elements) * sizeof(float);
    }

    size_t checked_total_bytes(size_t a_bytes, size_t b_bytes, size_t c_bytes) {
        if(a_bytes > std::numeric_limits<size_t>::max() - b_bytes) {
            throw std::overflow_error("total matrix allocation size exceeds size_t");
        }
        const size_t ab_bytes = a_bytes + b_bytes;
        if(ab_bytes > std::numeric_limits<size_t>::max() - c_bytes) {
            throw std::overflow_error("total matrix allocation size exceeds size_t");
        }
        return ab_bytes + c_bytes;
    }

    uint32_t checked_stride(uint32_t lhs, uint32_t rhs, const char* name) {
        const uint64_t value = uint64_t(lhs) * uint64_t(rhs);
        if(value > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(std::string(name) + " exceeds uint32_t");
        }
        return static_cast<uint32_t>(value);
    }

    double bytes_to_mib(size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    void fill_random(std::vector<float>& values, std::mt19937& gen) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(float& value : values) {
            value = dist(gen);
        }
    }

    void print_usage(const char* program_name) {
        std::cout
            << "Usage: " << program_name
            << " [batch] [m] [n] [p] [tile_m] [tile_n] [tile_p] [thread_tile_m] [thread_tile_p]\n";
    }
}

int main(int argc, char** argv) {
    try {
        if(argc > 1 && std::string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        const GemmPerfConfig config = parse_config(argc, argv);
        validate_tile_config(config);
        const uint64_t a_elements = matrix_elements(config.batch, config.m, config.n);
        const uint64_t b_elements = matrix_elements(config.batch, config.n, config.p);
        const uint64_t c_elements = matrix_elements(config.batch, config.m, config.p);

        const size_t a_bytes = checked_bytes(a_elements);
        const size_t b_bytes = checked_bytes(b_elements);
        const size_t c_bytes = checked_bytes(c_elements);
        const size_t total_bytes = checked_total_bytes(a_bytes, b_bytes, c_bytes);
        const uint32_t a_stride = checked_stride(config.m, config.n, "a_stride");
        const uint32_t b_stride = checked_stride(config.n, config.p, "b_stride");
        const uint32_t c_stride = checked_stride(config.m, config.p, "c_stride");

        std::cout << std::fixed << std::setprecision(3);
        std::cout
            << "GEMM performance test\n"
            << "  batch=" << config.batch
            << ", m=" << config.m
            << ", n=" << config.n
            << ", p=" << config.p << "\n"
            << "  tile_m=" << config.tile_m
            << ", tile_n=" << config.tile_n
            << ", tile_p=" << config.tile_p << "\n"
            << "  thread_tile_m=" << config.thread_tile_m
            << ", thread_tile_p=" << config.thread_tile_p << "\n"
            << "  host/device buffer bytes=" << total_bytes
            << " (" << bytes_to_mib(total_bytes) << " MiB)\n";

        std::mt19937 gen(config.seed);
        std::vector<float> a(static_cast<size_t>(a_elements));
        std::vector<float> b(static_cast<size_t>(b_elements));
        std::vector<float> c(static_cast<size_t>(c_elements), 0.0f);
        // fill_random(a, gen);
        // fill_random(b, gen);

        socl::Context ctx({0});
        ctx.printGpuInfo(std::cout);
        soclblas::GemmSharedFP32 gemm(
            ctx,
            config.tile_m,
            config.tile_n,
            config.tile_p,
            config.thread_tile_m,
            config.thread_tile_p
        );

        auto bufferA = ctx.createBuffer(a_bytes, socl::BufferType::Auto);
        auto bufferB = ctx.createBuffer(b_bytes, socl::BufferType::Auto);
        auto bufferC = ctx.createBuffer(c_bytes, socl::BufferType::Auto);

        bufferA.write(a.data(), a_bytes);
        bufferB.write(b.data(), b_bytes);
        bufferC.write(c.data(), c_bytes);

        soclblas::GemmArguments args = {
            .b = config.batch,
            .m = config.m,
            .n = config.n,
            .p = config.p,
            .alpha = 1.0f,
            .beta = 0.0f,
            .a_stride = a_stride,
            .b_stride = b_stride,
            .c_stride = c_stride,
            .a_m_stride = config.n,
            .a_n_stride = 1,
            .b_n_stride = config.p,
            .b_p_stride = 1,
            .c_m_stride = config.p,
            .c_p_stride = 1
        };

        const auto start = std::chrono::steady_clock::now();
        gemm(bufferA, bufferB, bufferC, args);
        const auto end = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double total_flops =
            2.0 *
            static_cast<double>(config.batch) *
            static_cast<double>(config.m) *
            static_cast<double>(config.n) *
            static_cast<double>(config.p);
        const double gflops = total_flops / seconds / 1.0e9;

        std::cout
            << "  elapsed_ms=" << seconds * 1000.0 << "\n"
            << "  total_flops=" << total_flops << "\n"
            << "  gpu_gflops=" << gflops << "\n";
    } catch(const std::exception& e) {
        std::cerr << "Performance test failed: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
