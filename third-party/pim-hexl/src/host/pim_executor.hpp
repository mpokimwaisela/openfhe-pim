#pragma once

#include "pim_vector.hpp"
#include "pim_launch_args.hpp"
#include "common.h"
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

namespace detail {
    template<typename T>
    size_t validate_and_get_elements(const Vector<T>& vec) {
        if (vec.empty()) throw std::runtime_error("Input buffer is empty");
        return vec.shard().bytes / sizeof(T);
    }

    template<typename T>
    dpu_arguments_t make_args(pimop_t op, const Vector<T>& A, const Vector<T>& C, 
                              dpu_word_t modulus = 0, dpu_word_t scalar = 0, dpu_word_t mu = 0,
                              cmp_t comparison = CMP_TRUE, dpu_word_t bound = 0,
                              uint32_t in_factor = 1, uint32_t out_factor = 1,
                              const Vector<T>* B = nullptr) {
        size_t elems = validate_and_get_elements(A);
        
        auto builder = ArgsBuilder{}
            .A(A.shard().off, elems)
            .C(C.shard().off, elems)
            .kernel(op)
            .mod(modulus)
            .scalar(scalar)
            .mu(mu)
            .cmp(comparison)
            .bound(bound)
            .in_factor(in_factor)
            .out_factor(out_factor);
            
        if (B) builder.B(B->shard().off, elems);
        return builder.build();
    }
}


/** @brief Element-wise modular addition: C[i] = (A[i] + B[i]) % modulus */
template<typename T>
void EltwiseAddMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& op2, dpu_word_t modulus) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(MOD_ADD, op1, destination, modulus, 0, 0, CMP_TRUE, 0, 1, 1, &op2);
    run_kernel(args, std::tie(op1, op2), std::tie(destination));
}

/** @brief Element-wise modular addition with scalar: C[i] = (A[i] + scalar) % modulus */
template<typename T>
void EltwiseAddScalarMod(Vector<T>& destination, const Vector<T>& op1, 
                         dpu_word_t scalar, dpu_word_t modulus) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(MOD_ADD_SCALAR, op1, destination, modulus, scalar);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

/** @brief Element-wise modular subtraction: C[i] = (A[i] - B[i]) % modulus */
template<typename T>
void EltwiseSubMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& op2, dpu_word_t modulus) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(MOD_SUB, op1, destination, modulus, 0, 0, CMP_TRUE, 0, 1, 1, &op2);
    run_kernel(args, std::tie(op1, op2), std::tie(destination));
}

/** @brief Element-wise modular subtraction with scalar: C[i] = (A[i] - scalar) % modulus */
template<typename T>
void EltwiseSubScalarMod(Vector<T>& destination, const Vector<T>& op1, 
                         dpu_word_t scalar, dpu_word_t modulus) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(MOD_SUB_SCALAR, op1, destination, modulus, scalar);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

/** @brief Element-wise modular multiplication: C[i] = (A[i] * B[i]) % modulus */
template<typename T>
void EltwiseMulMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& op2, dpu_word_t modulus, dpu_word_t mu = 0) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(MOD_MUL, op1, destination, modulus, 0, mu, CMP_TRUE, 0, 1, 1, &op2);
    run_kernel(args, std::tie(op1, op2), std::tie(destination));
}

/** @brief Element-wise modular multiplication with scalar: C[i] = (A[i] * scalar) % modulus */
template<typename T>
void EltwiseScalarMulMod(Vector<T>& destination, const Vector<T>& op1,
                         dpu_word_t scalar, dpu_word_t modulus, dpu_word_t mu = 0) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(MOD_MUL_SCALAR, op1, destination, modulus, scalar, mu);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

/** @brief Fused multiply-add modular: C[i] = (A[i] * scalar + B[i]) % modulus */
template<typename T>
void EltwiseFMAMod(Vector<T>& destination, const Vector<T>& op1, 
                   const Vector<T>& addend, dpu_word_t scalar, dpu_word_t modulus) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(FMA_MOD, op1, destination, modulus, scalar, 0, CMP_TRUE, 0, 1, 1, &addend);
    run_kernel(args, std::tie(op1, addend), std::tie(destination));
}


/** @brief Conditional addition: C[i] = A[i] + (A[i] cmp bound ? diff : 0) */
template<typename T>
void EltwiseConditionalAdd(Vector<T>& destination, const Vector<T>& op1, 
                           cmp_t comparison, dpu_word_t bound, dpu_word_t diff) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(CMP_ADD, op1, destination, 0, diff, 0, comparison, bound);
    run_kernel(args, std::tie(op1), std::tie(destination));
}

/** @brief Conditional modular subtraction: C[i] = A[i] - (A[i] cmp bound ? diff : 0) % modulus */
template<typename T>
void EltwiseConditionalSubMod(Vector<T>& destination, const Vector<T>& op1, 
                              dpu_word_t modulus, cmp_t comparison, dpu_word_t bound, dpu_word_t diff) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(CMP_SUB_MOD, op1, destination, modulus, diff, 0, comparison, bound);
    run_kernel(args, std::tie(op1), std::tie(destination));
}


/** @brief Modular reduction with scaling: C[i] = (A[i] / input_factor) % modulus * output_factor */
template<typename T>
void EltwiseReduceMod(Vector<T>& destination, const Vector<T>& op1, 
                      dpu_word_t modulus, uint32_t input_factor, uint32_t output_factor) {
    PROFILE_FUNCTION();
    auto args = detail::make_args(MOD_REDUCE, op1, destination, modulus, 0, 0, CMP_TRUE, 0, input_factor, output_factor);
    run_kernel(args, std::tie(op1), std::tie(destination));
}


/** @brief Initialize PIM system with specified number of DPUs */
inline void Init(unsigned num_dpus = 1, const std::string& kernel_path = "main.dpu") {
    PIMManager::init(num_dpus, kernel_path);
}

/** @brief Get the number of available DPUs */
inline unsigned GetNumDPUs() {
    return PIMManager::instance().num_dpus();
}


constexpr cmp_t EQUAL = CMP_EQ;
constexpr cmp_t NOT_EQUAL = CMP_NE;
constexpr cmp_t LESS_THAN = CMP_LT;
constexpr cmp_t LESS_EQUAL = CMP_LE;
constexpr cmp_t GREATER_EQUAL = CMP_NLT;
constexpr cmp_t GREATER_THAN = CMP_NLE;
constexpr cmp_t ALWAYS_TRUE = CMP_TRUE;
constexpr cmp_t ALWAYS_FALSE = CMP_FALSE;

} // namespace pim
