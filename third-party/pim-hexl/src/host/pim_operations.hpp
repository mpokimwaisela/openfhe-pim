// ─────────────────────────── pim_operations.hpp ───────────────────────────
#pragma once

#include "host.hpp"
#include "../utils/host_args.hpp"
#include "../utils/common.h"
#include <stdexcept>

/**
 * @file pim_operations.hpp
 * @brief High-level PIM-HEXL operations for easy integration with OpenFHE
 * 
 * This header provides simple, user-friendly functions for performing 
 * pointwise operations on Vectors. Each function handles the low-level
 * DPU kernel invocation and argument building automatically.
 * 
 * All operations work element-wise across the distributed Vector shards.
 */

namespace pim {

// ─────────────────────────── Helper Functions ───────────────────────────

namespace detail {
    /**
     * @brief Build DPU arguments for binary operations (A op B -> C)
     */
    template<typename T>
    dpu_arguments_t make_binary_args(pimop_t op, const Vector<T>& A, 
                                   const Vector<T>& B, const Vector<T>& C,
                                   uint64_t modulus, uint64_t scalar = 0) {
        if (A.shards().empty()) 
            throw std::runtime_error("Input buffer A has no shards");
        
        size_t elems = A.shards()[0].host.size();
        
        return ArgsBuilder{}
            .A(A.shards()[0].blk.off, elems)
            .B(B.shards()[0].blk.off, elems)
            .C(C.shards()[0].blk.off, elems)
            .kernel(op)
            .mod(modulus)
            .scalar(scalar)
            .cmp(CMP_TRUE)
            .bound(0)
            .in_factor(1)
            .out_factor(1)
            .build();
    }

    /**
     * @brief Build DPU arguments for unary operations (A op scalar -> C)
     */
    template<typename T>
    dpu_arguments_t make_unary_args(pimop_t op, const Vector<T>& A, 
                                  const Vector<T>& C, uint64_t modulus, 
                                  uint64_t scalar) {
        if (A.shards().empty()) 
            throw std::runtime_error("Input buffer A has no shards");
        
        size_t elems = A.shards()[0].host.size();
        
        return ArgsBuilder{}
            .A(A.shards()[0].blk.off, elems)
            .C(C.shards()[0].blk.off, elems)
            .kernel(op)
            .mod(modulus)
            .scalar(scalar)
            .cmp(CMP_TRUE)
            .bound(0)
            .in_factor(1)
            .out_factor(1)
            .build();
    }

    /**
     * @brief Build DPU arguments for compare operations
     */
    template<typename T>
    dpu_arguments_t make_compare_args(pimop_t op, const Vector<T>& A, 
                                    const Vector<T>& C, uint64_t modulus,
                                    cmp_t comparison, uint64_t bound, uint64_t diff) {
        if (A.shards().empty()) 
            throw std::runtime_error("Input buffer A has no shards");
        
        size_t elems = A.shards()[0].host.size();
        
        return ArgsBuilder{}
            .A(A.shards()[0].blk.off, elems)
            .C(C.shards()[0].blk.off, elems)
            .kernel(op)
            .mod(modulus)
            .scalar(diff)
            .cmp(comparison)
            .bound(bound)
            .in_factor(1)
            .out_factor(1)
            .build();
    }
}

// ─────────────────────────── Arithmetic Operations ───────────────────────────

/**
 * @brief Element-wise modular addition: C[i] = (A[i] + B[i]) % modulus
 */
template<typename T>
void EltwiseAddMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& op2, uint64_t modulus) {
    auto args = detail::make_binary_args(MOD_ADD, op1, op2, destination, modulus);
    run_kernel(args, std::tie(op1, op2), std::tie(destination));
}

/**
 * @brief Element-wise modular addition with scalar: C[i] = (A[i] + scalar) % modulus
 */
template<typename T>
void EltwiseAddScalarMod(Vector<T>& destination, const Vector<T>& op1, 
                         uint64_t scalar, uint64_t modulus) {
    auto args = detail::make_unary_args(MOD_ADD_SCALAR, op1, destination, modulus, scalar);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

/**
 * @brief Element-wise modular subtraction: C[i] = (A[i] - B[i]) % modulus
 */
template<typename T>
void EltwiseSubMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& op2, uint64_t modulus) {
    auto args = detail::make_binary_args(MOD_SUB, op1, op2, destination, modulus);
    run_kernel(args, std::tie(op1, op2), std::tie(destination));
}

/**
 * @brief Element-wise modular subtraction with scalar: C[i] = (A[i] - scalar) % modulus
 */
template<typename T>
void EltwiseSubScalarMod(Vector<T>& destination, const Vector<T>& op1, 
                         uint64_t scalar, uint64_t modulus) {
    auto args = detail::make_unary_args(MOD_SUB_SCALAR, op1, destination, modulus, scalar);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

/**
 * @brief Element-wise modular multiplication: C[i] = (A[i] * B[i]) % modulus
 */
template<typename T>
void EltwiseMulMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& op2, uint64_t modulus) {
    auto args = detail::make_binary_args(MOD_MUL, op1, op2, destination, modulus);
    run_kernel(args, std::tie(op1, op2), std::tie(destination));
}

// ─────────────────────────── FMA Operations ───────────────────────────

/**
 * @brief Fused multiply-add modular: C[i] = (A[i] * scalar + B[i]) % modulus
 */
template<typename T>
void EltwiseFMAMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& addend, uint64_t scalar, uint64_t modulus) {
    if (op1.shards().empty()) 
        throw std::runtime_error("Input buffer has no shards");
    
    size_t elems = op1.shards()[0].host.size();
    
    auto args = ArgsBuilder{}
        .A(op1.shards()[0].blk.off, elems)
        .B(addend.shards()[0].blk.off, elems)
        .C(destination.shards()[0].blk.off, elems)
        .kernel(FMA_MOD)
        .mod(modulus)
        .scalar(scalar)
        .cmp(CMP_TRUE)
        .bound(0)
        .in_factor(1)
        .out_factor(1)
        .build();
    
    run_kernel(args, std::tie(op1, addend), std::tie(destination));
}

/**
 * @brief Scalar multiplication modular (no addend): C[i] = (A[i] * scalar) % modulus
 */
template<typename T>
void EltwiseScalarMulMod(Vector<T>& destination, const Vector<T>& op1, 
                         uint64_t scalar, uint64_t modulus) {
    if (op1.shards().empty()) 
        throw std::runtime_error("Input buffer has no shards");
    
    size_t elems = op1.shards()[0].host.size();
    
    auto args = ArgsBuilder{}
        .A(op1.shards()[0].blk.off, elems)
        .C(destination.shards()[0].blk.off, elems)
        .kernel(FMA_MOD)
        .mod(modulus)
        .scalar(scalar)
        .cmp(CMP_TRUE)
        .bound(0)
        .in_factor(1)
        .out_factor(1)
        .build();
    
    run_kernel(args, std::tie(op1), std::tie(destination));
}

// ─────────────────────────── Conditional Operations ───────────────────────────

/**
 * @brief Conditional addition: C[i] = A[i] + (A[i] cmp bound ? diff : 0)
 */
template<typename T>
void EltwiseConditionalAdd(Vector<T>& destination, const Vector<T>& op1, 
                           cmp_t comparison, uint64_t bound, uint64_t diff) {
    auto args = detail::make_compare_args(CMP_ADD, op1, destination, 0, comparison, bound, diff);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

/**
 * @brief Conditional modular subtraction: C[i] = A[i] - (A[i] cmp bound ? diff : 0) % modulus
 */
template<typename T>
void EltwiseConditionalSubMod(Vector<T>& destination, const Vector<T>& op1, 
                              uint64_t modulus, cmp_t comparison, uint64_t bound, uint64_t diff) {
    auto args = detail::make_compare_args(CMP_SUB_MOD, op1, destination, modulus, comparison, bound, diff);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

// ─────────────────────────── Reduction Operations ───────────────────────────

/**
 * @brief Modular reduction with scaling factors: C[i] = (A[i] / input_factor) % modulus * output_factor
 */
template<typename T>
void EltwiseReduceMod(Vector<T>& destination, const Vector<T>& op1, 
                      uint64_t modulus, uint32_t input_factor, uint32_t output_factor) {
    if (op1.shards().empty()) 
        throw std::runtime_error("Input buffer has no shards");
    
    size_t elems = op1.shards()[0].host.size();
    
    auto args = ArgsBuilder{}
        .A(op1.shards()[0].blk.off, elems)
        .C(destination.shards()[0].blk.off, elems)
        .kernel(MOD_REDUCE)
        .mod(modulus)
        .scalar(0)
        .cmp(CMP_TRUE)
        .bound(0)
        .in_factor(input_factor)
        .out_factor(output_factor)
        .build();
    
    run_kernel(args, std::tie(op1), std::tie(destination));
}

// ─────────────────────────── Utility Functions ───────────────────────────

/**
 * @brief Initialize PIM system with specified number of DPUs
 * @param num_dpus Number of DPUs to allocate (default: 1)
 * @param kernel_path Path to DPU kernel binary (default: "main.dpu")
 */
inline void Init(unsigned num_dpus = 1, const std::string& kernel_path = "main.dpu") {
    PIMManager::init(num_dpus, kernel_path);
}

/**
 * @brief Get the number of available DPUs
 */
inline unsigned GetNumDPUs() {
    return PIMManager::instance().num_dpus();
}

// ─────────────────────────── Comparison Constants ───────────────────────────

constexpr cmp_t EQUAL = CMP_EQ;
constexpr cmp_t NOT_EQUAL = CMP_NE;
constexpr cmp_t LESS_THAN = CMP_LT;
constexpr cmp_t LESS_EQUAL = CMP_LE;
constexpr cmp_t GREATER_EQUAL = CMP_NLT;
constexpr cmp_t GREATER_THAN = CMP_NLE;
constexpr cmp_t ALWAYS_TRUE = CMP_TRUE;
constexpr cmp_t ALWAYS_FALSE = CMP_FALSE;

} // namespace pim
