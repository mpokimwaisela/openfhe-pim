// ─────────────────────────── pim_operations_test.cpp ───────────────────────────
/**
 * @file pim_operations_test.cpp
 * @brief Rigorous test suite for PIM operations with large polynomials
 * 
 * This test suite validates PIM operations correctness using polynomial size 8192
 * and large 60-bit modulus, comparing PIM results against CPU reference implementations.
 */

#include "../host/pim.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <chrono>
#include <cassert>
#include <cstdint>

// Test configuration
const size_t POLY_SIZE = 8192;
const uint64_t MODULUS_60BIT = (1ULL << 60) - 93;  // Large 60-bit prime: 1152921504606846883
const uint64_t SMALL_MODULUS = 40961;  // 2^15 + 1, commonly used in FHE

// Utility functions for correctness verification
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = a + b;
    return (result >= mod) ? result - mod : result;
}

uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    return (a >= b) ? a - b : a + mod - b;
}

uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    return ((__uint128_t)a * b) % mod;
}

// Generate test vectors with controlled patterns
void generate_test_vectors(std::vector<uint64_t>& A, std::vector<uint64_t>& B, uint64_t modulus, int seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, modulus - 1);
    
    A.resize(POLY_SIZE);
    B.resize(POLY_SIZE);
    
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }
}

bool test_basic_arithmetic_rigorous(uint64_t modulus, const std::string& test_name) {
    std::cout << "\n=== Rigorous Basic Arithmetic Test: " << test_name << " ===\n";
    std::cout << "Polynomial size: " << POLY_SIZE << ", Modulus: " << modulus << std::endl;
    
    // Generate test data
    std::vector<uint64_t> A_cpu, B_cpu;
    generate_test_vectors(A_cpu, B_cpu, modulus);
    
    // Create PIM vectors
    pim::Vector<uint64_t> A_pim(POLY_SIZE), B_pim(POLY_SIZE), C_pim(POLY_SIZE);
    
    // Copy data to PIM vectors
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A_pim[i] = A_cpu[i];
        B_pim[i] = B_cpu[i];
    }
    
    bool all_tests_passed = true;
    
    // Test 1: Addition
    std::cout << "Testing modular addition... ";
    auto start = std::chrono::high_resolution_clock::now();
    pim::EltwiseAddMod(C_pim, A_pim, B_pim, modulus);
    auto end = std::chrono::high_resolution_clock::now();
    auto pim_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify correctness
    bool add_correct = true;
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = mod_add(A_cpu[i], B_cpu[i], modulus);
        if (C_pim[i] != expected) {
            std::cout << "FAILED at index " << i << ": got " << C_pim[i] << ", expected " << expected << std::endl;
            std::cout << "A[" << i << "] = " << A_cpu[i] << ", B[" << i << "] = " << B_cpu[i] << std::endl;
            add_correct = false;
            break;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (add_correct) {
        std::cout << "PASSED (PIM: " << pim_time.count() << "μs, CPU verify: " << cpu_time.count() << "μs)" << std::endl;
    } else {
        all_tests_passed = false;
    }
    
    // Test 2: Subtraction
    std::cout << "Testing modular subtraction... ";
    start = std::chrono::high_resolution_clock::now();
    pim::EltwiseSubMod(C_pim, A_pim, B_pim, modulus);
    end = std::chrono::high_resolution_clock::now();
    pim_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool sub_correct = true;
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = mod_sub(A_cpu[i], B_cpu[i], modulus);
        if (C_pim[i] != expected) {
            std::cout << "FAILED at index " << i << ": got " << C_pim[i] << ", expected " << expected << std::endl;
            sub_correct = false;
            break;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (sub_correct) {
        std::cout << "PASSED (PIM: " << pim_time.count() << "μs, CPU verify: " << cpu_time.count() << "μs)" << std::endl;
    } else {
        all_tests_passed = false;
    }
    
    // Test 3: Multiplication
    std::cout << "Testing modular multiplication... ";
    start = std::chrono::high_resolution_clock::now();
    pim::EltwiseMulMod(C_pim, A_pim, B_pim, modulus);
    end = std::chrono::high_resolution_clock::now();
    pim_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool mul_correct = true;
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = mod_mul(A_cpu[i], B_cpu[i], modulus);
        if (C_pim[i] != expected) {
            std::cout << "FAILED at index " << i << ": got " << C_pim[i] << ", expected " << expected << std::endl;
            mul_correct = false;
            break;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (mul_correct) {
        std::cout << "PASSED (PIM: " << pim_time.count() << "μs, CPU verify: " << cpu_time.count() << "μs)" << std::endl;
    } else {
        all_tests_passed = false;
    }
    
    return all_tests_passed;
}

bool test_scalar_operations_rigorous(uint64_t modulus, const std::string& test_name) {
    std::cout << "\n=== Rigorous Scalar Operations Test: " << test_name << " ===\n";
    std::cout << "Polynomial size: " << POLY_SIZE << ", Modulus: " << modulus << std::endl;
    
    // Generate test data
    std::vector<uint64_t> A_cpu, B_cpu;
    generate_test_vectors(A_cpu, B_cpu, modulus, 123);
    
    uint64_t scalar = 0x123456789ABCDEFULL % modulus;  // Large scalar
    
    pim::Vector<uint64_t> A_pim(POLY_SIZE), C_pim(POLY_SIZE);
    
    // Copy data to PIM vectors
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A_pim[i] = A_cpu[i];
    }
    
    std::cout << "Using scalar: " << scalar << std::endl;
    
    bool all_tests_passed = true;
    
    // Test 1: Scalar Addition
    std::cout << "Testing scalar addition... ";
    auto start = std::chrono::high_resolution_clock::now();
    pim::EltwiseAddScalarMod(C_pim, A_pim, scalar, modulus);
    auto end = std::chrono::high_resolution_clock::now();
    auto pim_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool add_correct = true;
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = mod_add(A_cpu[i], scalar, modulus);
        if (C_pim[i] != expected) {
            std::cout << "FAILED at index " << i << ": got " << C_pim[i] << ", expected " << expected << std::endl;
            add_correct = false;
            break;
        }
    }
    
    if (add_correct) {
        std::cout << "PASSED (PIM: " << pim_time.count() << "μs)" << std::endl;
    } else {
        all_tests_passed = false;
    }
    
    // Test 2: Scalar Subtraction  
    std::cout << "Testing scalar subtraction... ";
    start = std::chrono::high_resolution_clock::now();
    pim::EltwiseSubScalarMod(C_pim, A_pim, scalar, modulus);
    end = std::chrono::high_resolution_clock::now();
    pim_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool sub_correct = true;
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = mod_sub(A_cpu[i], scalar, modulus);
        if (C_pim[i] != expected) {
            std::cout << "FAILED at index " << i << ": got " << C_pim[i] << ", expected " << expected << std::endl;
            sub_correct = false;
            break;
        }
    }
    
    if (sub_correct) {
        std::cout << "PASSED (PIM: " << pim_time.count() << "μs)" << std::endl;
    } else {
        all_tests_passed = false;
    }
    
    // Test 3: Scalar Multiplication
    std::cout << "Testing scalar multiplication... ";
    start = std::chrono::high_resolution_clock::now();
    pim::EltwiseScalarMulMod(C_pim, A_pim, scalar, modulus);
    end = std::chrono::high_resolution_clock::now();
    pim_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    bool mul_correct = true;
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = mod_mul(A_cpu[i], scalar, modulus);
        if (C_pim[i] != expected) {
            std::cout << "FAILED at index " << i << ": got " << C_pim[i] << ", expected " << expected << std::endl;
            mul_correct = false;
            break;
        }
    }
    
    if (mul_correct) {
        std::cout << "PASSED (PIM: " << pim_time.count() << "μs)" << std::endl;
    } else {
        all_tests_passed = false;
    }
    
    return all_tests_passed;
}

bool test_edge_cases(uint64_t modulus, const std::string& test_name) {
    std::cout << "\n=== Edge Cases Test: " << test_name << " ===\n";
    
    pim::Vector<uint64_t> A_pim(POLY_SIZE), B_pim(POLY_SIZE), C_pim(POLY_SIZE);
    
    // Test 1: All zeros
    std::cout << "Testing with all zeros... ";
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A_pim[i] = 0;
        B_pim[i] = 0;
    }
    
    pim::EltwiseAddMod(C_pim, A_pim, B_pim, modulus);
    bool zero_test = true;
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        if (C_pim[i] != 0) {
            zero_test = false;
            break;
        }
    }
    std::cout << (zero_test ? "PASSED" : "FAILED") << std::endl;
    
    // Test 2: Maximum values
    std::cout << "Testing with maximum values... ";
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A_pim[i] = modulus - 1;
        B_pim[i] = modulus - 1;
    }
    
    pim::EltwiseAddMod(C_pim, A_pim, B_pim, modulus);
    bool max_test = true;
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        uint64_t expected = (2 * (modulus - 1)) % modulus;
        if (C_pim[i] != expected) {
            max_test = false;
            break;
        }
    }
    std::cout << (max_test ? "PASSED" : "FAILED") << std::endl;
    
    // Test 3: Powers of 2
    std::cout << "Testing with powers of 2... ";
    bool power2_test = true;
    for (size_t i = 0; i < std::min(POLY_SIZE, size_t(60)); ++i) {
        A_pim[i] = (1ULL << i) % modulus;
        B_pim[i] = (1ULL << i) % modulus;
    }
    for (size_t i = 60; i < POLY_SIZE; ++i) {
        A_pim[i] = 1;
        B_pim[i] = 1;
    }
    
    pim::EltwiseAddMod(C_pim, A_pim, B_pim, modulus);
    
    for (size_t i = 0; i < std::min(POLY_SIZE, size_t(60)); ++i) {
        uint64_t expected = (2 * ((1ULL << i) % modulus)) % modulus;
        if (C_pim[i] != expected) {
            power2_test = false;
            break;
        }
    }
    std::cout << (power2_test ? "PASSED" : "FAILED") << std::endl;
    
    return zero_test && max_test && power2_test;
}

void performance_comparison(uint64_t modulus, const std::string& test_name) {
    std::cout << "\n=== Performance Comparison: " << test_name << " ===\n";
    
    // Generate test data
    std::vector<uint64_t> A_cpu, B_cpu, C_cpu(POLY_SIZE);
    generate_test_vectors(A_cpu, B_cpu, modulus, 999);
    
    // CPU baseline
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        C_cpu[i] = mod_add(A_cpu[i], B_cpu[i], modulus);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_add_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        C_cpu[i] = mod_mul(A_cpu[i], B_cpu[i], modulus);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cpu_mul_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // PIM performance
    pim::Vector<uint64_t> A_pim(POLY_SIZE), B_pim(POLY_SIZE), C_pim(POLY_SIZE);
    for (size_t i = 0; i < POLY_SIZE; ++i) {
        A_pim[i] = A_cpu[i];
        B_pim[i] = B_cpu[i];
    }
    
    start = std::chrono::high_resolution_clock::now();
    pim::EltwiseAddMod(C_pim, A_pim, B_pim, modulus);
    end = std::chrono::high_resolution_clock::now();
    auto pim_add_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    pim::EltwiseMulMod(C_pim, A_pim, B_pim, modulus);
    end = std::chrono::high_resolution_clock::now();
    auto pim_mul_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Addition  - CPU: " << cpu_add_time.count() << "μs, PIM: " << pim_add_time.count() << "μs";
    if (cpu_add_time.count() > 0) {
        std::cout << " (speedup: " << std::fixed << std::setprecision(2) 
                  << (double)cpu_add_time.count() / pim_add_time.count() << "x)";
    }
    std::cout << std::endl;
    
    std::cout << "Multiplication - CPU: " << cpu_mul_time.count() << "μs, PIM: " << pim_mul_time.count() << "μs";
    if (cpu_mul_time.count() > 0) {
        std::cout << " (speedup: " << std::fixed << std::setprecision(2) 
                  << (double)cpu_mul_time.count() / pim_mul_time.count() << "x)";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "        PIM Operations Rigorous Test Suite\n";
    std::cout << "=================================================================\n";
    std::cout << "Polynomial size: " << POLY_SIZE << std::endl;
    std::cout << "60-bit modulus: " << MODULUS_60BIT << std::endl;
    std::cout << "Small modulus:  " << SMALL_MODULUS << std::endl;
    std::cout << "=================================================================\n";
    
    try {
        pim::Init(256);  
        std::cout << "Initialized PIM system with " << pim::GetNumDPUs() << " DPU(s)\n";
        
        bool all_tests_passed = true;
        
        // Test with small modulus first
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TESTING WITH SMALL MODULUS (2^15 + 1)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        if (!test_basic_arithmetic_rigorous(SMALL_MODULUS, "Small Modulus")) {
            all_tests_passed = false;
        }
        
        if (!test_scalar_operations_rigorous(SMALL_MODULUS, "Small Modulus")) {
            all_tests_passed = false;
        }
        
        if (!test_edge_cases(SMALL_MODULUS, "Small Modulus")) {
            all_tests_passed = false;
        }
        
        performance_comparison(SMALL_MODULUS, "Small Modulus");
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TESTING WITH LARGE 60-BIT MODULUS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        if (!test_basic_arithmetic_rigorous(MODULUS_60BIT, "60-bit Modulus")) {
            all_tests_passed = false;
        }
        
        if (!test_scalar_operations_rigorous(MODULUS_60BIT, "60-bit Modulus")) {
            all_tests_passed = false;
        }
        
        if (!test_edge_cases(MODULUS_60BIT, "60-bit Modulus")) {
            all_tests_passed = false;
        }
        
        performance_comparison(MODULUS_60BIT, "60-bit Modulus");
        
        // Final summary
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "FINAL RESULTS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        if (all_tests_passed) {
            std::cout << "✅ ALL TESTS PASSED! PIM operations are working correctly." << std::endl;
        } else {
            std::cout << "❌ SOME TESTS FAILED! Please check the output above." << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during test execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
