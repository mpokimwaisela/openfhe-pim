#include "pim.hpp"
#include <iostream>
#include <vector>
#include <random>

constexpr size_t POLY_SIZE = 8192;
constexpr uint64_t MODULUS = (1ULL << 60) - 93;

uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t r = a + b;
    return (r >= mod) ? r - mod : r;
}

int main() {
    std::cout << "== Minimal PIM Addition with Profiling ==\n";

    pim::Init(509);  

    std::vector<uint64_t> A_cpu(POLY_SIZE), B_cpu(POLY_SIZE);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, MODULUS - 1);
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A_cpu[i] = dist(rng);
        B_cpu[i] = dist(rng);
    }

    pim::Vector<uint64_t> A_pim(POLY_SIZE), B_pim(POLY_SIZE), C_pim(POLY_SIZE);

    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A_pim[i] = A_cpu[i];
        B_pim[i] = B_cpu[i];
    }

    {
        PROFILE_SCOPE("PIM EltwiseAddMod");
        pim::EltwiseAddMod(C_pim, A_pim, B_pim, MODULUS);
    }

    bool ok = true;
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = mod_add(A_cpu[i], B_cpu[i], MODULUS);
        if (C_pim[i] != expected) {
            std::cerr << "Mismatch at " << i << ": got " << C_pim[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    std::cout << (ok ? "✅ Addition Correct\n" : "❌ Addition Incorrect\n");

    // Dump profiling report
    pim::Profiler::instance().print_report();

    return ok ? 0 : 1;
}
