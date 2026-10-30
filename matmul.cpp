#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Naive O(n^3) multiplication: C = A * B.
// Matrices are stored in row-major order as one-dimensional arrays.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void matmul(const std::vector<double>& a,
            const std::vector<double>& b,
            std::vector<double>& c,
            std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main() {
    const std::vector<std::size_t> sizes{4, 8, 12, 16, 32, 64, 128, 256};
    // Aim for enough work at every size to reduce timer noise.
    constexpr std::size_t target_multiply_adds = 100'000'000;

    std::mt19937 generator(4630);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    double checksum = 0.0;

    std::cout << std::left << std::setw(8) << "n"
              << std::setw(14) << "repetitions"
              << std::setw(16) << "total (ms)"
              << std::setw(18) << "average (us)"
              << "GFLOP/s\n";

    for (const std::size_t n : sizes) {
        std::vector<double> a(n * n);
        std::vector<double> b(n * n);
        std::vector<double> c(n * n, 0.0);

        for (double& value : a) value = distribution(generator);
        for (double& value : b) value = distribution(generator);

        // Repeat small multiplications so their timings are measurable.
        const std::size_t operations_per_multiply = n * n * n;
        const std::size_t repetitions =
            std::max<std::size_t>(1, target_multiply_adds / operations_per_multiply);

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            matmul(a, b, c, n);
        }
        const auto stop = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = stop - start;
        const double average_seconds = elapsed.count() / repetitions;
        const double floating_point_operations =
            2.0 * static_cast<double>(operations_per_multiply);
        const double gflops = floating_point_operations / average_seconds / 1.0e9;

        // Use the result so an optimizing compiler cannot discard the work.
        checksum += c[(n / 2) * n + (n / 2)];

        std::cout << std::left << std::setw(8) << n
                  << std::setw(14) << repetitions
                  << std::setw(16) << std::fixed << std::setprecision(3)
                  << elapsed.count() * 1.0e3
                  << std::setw(18) << average_seconds * 1.0e6
                  << std::setprecision(3) << gflops << '\n';
    }

    std::cerr << "checksum: " << checksum << '\n';
    return 0;
}
