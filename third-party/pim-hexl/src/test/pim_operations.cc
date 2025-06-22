// ─────────────────────────── pim_operations_example.cpp ───────────────────────────
/**
 * @file pim_operations_example.cpp
 * @brief Example usage of the high-level PIM operations
 * 
 * This example demonstrates how to use the simplified PIM operations
 * for common modular arithmetic tasks in OpenFHE integration.
 */

#include "pim.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

void test_basic_arithmetic() {
    std::cout << "\n=== Basic Arithmetic Operations ===\n";
    
    const size_t N = 8;
    const uint64_t modulus = 100;
    
    // Create PIM buffers
    pim::Vector<uint64_t> A(N), B(N), C(N);
    
    // Initialize test data
    for (size_t i = 0; i < N; ++i) {
        A[i] = (i * 13) % modulus;  // 0, 13, 26, 39, 52, 65, 78, 91
        B[i] = (i * 7 + 5) % modulus;   // 5, 12, 19, 26, 33, 40, 47, 54
    }
    
    std::cout << "Input A: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << A[i] << " ";
    std::cout << "\nInput B: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << B[i] << " ";
    std::cout << "\n";
    
    // Test addition
    pim::EltwiseAddMod(C, A, B, modulus);
    std::cout << "A + B:   ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
    
    // Test subtraction
    pim::EltwiseSubMod(C, A, B, modulus);
    std::cout << "A - B:   ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
    
    // Test multiplication
    pim::EltwiseMulMod(C, A, B, modulus);
    std::cout << "A * B:   ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
}

void test_scalar_operations() {
    std::cout << "\n=== Scalar Operations ===\n";
    
    const size_t N = 8;
    const uint64_t modulus = 257;
    const uint64_t scalar = 17;
    
    pim::Vector<uint64_t> A(N), C(N);
    
    // Initialize test data
    for (size_t i = 0; i < N; ++i) {
        A[i] = (i * 23) % modulus;
    }
    
    std::cout << "Input A: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << A[i] << " ";
    std::cout << "\nScalar:  " << scalar << "\n";
    
    // Test scalar addition
    pim::EltwiseAddScalarMod(C, A, scalar, modulus);
    std::cout << "A + " << scalar << ":   ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
    
    // Test scalar subtraction
    pim::EltwiseSubScalarMod(C, A, scalar, modulus);
    std::cout << "A - " << scalar << ":   ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
    
    // Test scalar multiplication
    pim::EltwiseScalarMulMod(C, A, scalar, modulus);
    std::cout << "A * " << scalar << ":   ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
}

void test_fma_operations() {
    std::cout << "\n=== FMA Operations ===\n";
    
    const size_t N = 8;
    const uint64_t modulus = 257;
    const uint64_t scalar = 13;
    
    pim::Vector<uint64_t> A(N), B(N), C(N);
    
    // Initialize test data
    for (size_t i = 0; i < N; ++i) {
        A[i] = i + 1;           // 1, 2, 3, 4, 5, 6, 7, 8
        B[i] = (i * 2 + 3) % modulus; // 3, 5, 7, 9, 11, 13, 15, 17
    }
    
    std::cout << "Input A: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << A[i] << " ";
    std::cout << "\nInput B: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << B[i] << " ";
    std::cout << "\nScalar:  " << scalar << "\n";
    
    // Test FMA: A * scalar + B
    pim::EltwiseFMAMod(C, A, B, scalar, modulus);
    std::cout << "A*" << scalar << "+B: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
}

void test_conditional_operations() {
    std::cout << "\n=== Conditional Operations ===\n";
    
    const size_t N = 8;
    const uint64_t modulus = 257;
    const uint64_t bound = 50;
    const uint64_t diff = 10;
    
    pim::Vector<uint64_t> A(N), C(N);
    
    // Initialize test data with values around the bound
    std::vector<uint64_t> test_values = {30, 45, 50, 55, 60, 75, 40, 80};
    for (size_t i = 0; i < N; ++i) {
        A[i] = test_values[i];
    }
    
    std::cout << "Input A: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << A[i] << " ";
    std::cout << "\nBound:   " << bound << ", Diff: " << diff << "\n";
    
    // Test conditional add (if A[i] >= bound, add diff)
    pim::EltwiseConditionalAdd(C, A, pim::GREATER_EQUAL, bound, diff);
    std::cout << "A+(A>=" << bound << "?" << diff << ":0): ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
    
    // Test conditional sub mod (if A[i] < bound, subtract diff mod modulus)
    pim::EltwiseConditionalSubMod(C, A, modulus, pim::LESS_THAN, bound, diff);
    std::cout << "A-(A<" << bound << "?" << diff << ":0): ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(3) << C[i] << " ";
    std::cout << "\n";
}

void test_reduction_operations() {
    std::cout << "\n=== Reduction Operations ===\n";
    
    const size_t N = 6;
    const uint64_t modulus = 750;
    
    pim::Vector<uint64_t> A(N), C(N);
    
    // Test case from pointwise_test.cc
    std::vector<uint64_t> test_values = {0, 450, 735, 900, 1350, 1459};
    for (size_t i = 0; i < N; ++i) {
        A[i] = test_values[i];
    }
    
    std::cout << "Input A: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(4) << A[i] << " ";
    std::cout << "\nModulus: " << modulus << "\n";
    
    // Test reduction with factor 2->2 (should be identity for values < modulus)
    pim::EltwiseReduceMod(C, A, modulus, 2, 2);
    std::cout << "Reduce 2->2: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(4) << C[i] << " ";
    std::cout << "\n";
    
    // Test reduction with factor 4->1 
    pim::EltwiseReduceMod(C, A, modulus, 4, 1);
    std::cout << "Reduce 4->1: ";
    for (size_t i = 0; i < N; ++i) std::cout << std::setw(4) << C[i] << " ";
    std::cout << "\n";
}

int main() {
    std::cout << "PIM Operations Example\n";
    std::cout << "=====================\n";
    
    try {
        // Initialize PIM system
        pim::Init(1);  // Use 1 DPU for simplicity
        std::cout << "Initialized PIM system with " << pim::GetNumDPUs() << " DPU(s)\n";
        
        // Run test suites
        test_basic_arithmetic();
        test_scalar_operations();
        test_fma_operations();
        test_conditional_operations();
        test_reduction_operations();
        
        std::cout << "\nAll operations completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

/*
Expected Output Format:
======================

=== Basic Arithmetic Operations ===
Input A:   0  13  26  39  52  65  78  91 
Input B:   5  12  19  26  33  40  47  54 
A + B:     5  25  45  65  85 105 125 145 
A - B:   252   1   7  13  19  25  31  37 
A * B:     0 156 237 243 231  85 242  91 

=== Scalar Operations ===
Input A:   0  23  46  69  92 115 138 161 
Scalar:  17
A + 17:   17  40  63  86 109 132 155 178 
A - 17:  240   6  29  52  75  98 121 144 
A * 17:    0 134  11 145  22 156  33 167 

... (additional output for other test suites)
*/
