#include <cstdint>
#include "pim.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

std::pair<pim::Vector<dpu_word_t>, std::vector<dpu_word_t>> createRandomVector(size_t size, dpu_word_t modulus) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<dpu_word_t> dist(0, modulus - 1);
    std::vector<dpu_word_t> vec(size); // cpu
    pim::Vector<dpu_word_t> pim_vec(size); // pim

    for (size_t i = 0; i < size; ++i) {
        dpu_word_t value = dist(rng);
        vec[i] = value;
        pim_vec[i] = value; // Copy to PIM vector
    }
    return {pim_vec, vec};
}

std::vector<dpu_word_t> modAdd(std::vector<dpu_word_t> &a, std::vector<dpu_word_t> &b, dpu_word_t mod) {
    std::vector<dpu_word_t> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = add_mod(a[i], b[i], mod);
    }
    return result;
}

std::vector<dpu_word_t> modMul(std::vector<dpu_word_t> &a, std::vector<dpu_word_t> &b, dpu_word_t mod) {
    std::vector<dpu_word_t> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        __uint128_t exp128 = (__uint128_t)a[i] * b[i] % mod;
        result[i] = (dpu_word_t)exp128;
    }
    return result;
}



int main(){
    pim::Init(256);
    dpu_word_t modulus = (1ULL << 60) - 13;  
    std::vector<uint32_t> poly_size = {1024, 4096 ,8192, 16384, 65536}; 

    for (auto size : poly_size) {
        auto [pim_vec_1, cpu_vec_1] = createRandomVector(size, modulus);
        auto [pim_vec_2, cpu_vec_2] = createRandomVector(size, modulus);

        for (size_t i = 0; i<50; ++i) {
       // Perform CPU operations
        {
            PROFILE_SCOPE("CPU Add "+ std::to_string(size));
            cpu_vec_1 = modAdd(cpu_vec_1, cpu_vec_2, modulus);
        }
        {
            PROFILE_SCOPE("CPU Mul "+ std::to_string(size));
            cpu_vec_1 = modMul(cpu_vec_1, cpu_vec_2, modulus);
        }

        // Perform PIM operations
        {
            PROFILE_SCOPE("PIM Add "+ std::to_string(size));
            pim::EltwiseAddMod(pim_vec_1, pim_vec_1, pim_vec_2, modulus);
        }
        {
            PROFILE_SCOPE("PIM Mul "+ std::to_string(size));
            pim::EltwiseMulMod(pim_vec_1, pim_vec_1, pim_vec_2, modulus);
        }
        }
    }

};

