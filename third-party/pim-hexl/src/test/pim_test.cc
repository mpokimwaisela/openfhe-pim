#include <cstdint>
#include <gtest/gtest.h>
#include "pim.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

class PIM_Test : public ::testing::Test {
protected:
    void SetUp() override {        
        modulus = (1ULL << 60) - 59;  
        vector_size = 8192;
        
        rng.seed(42);
                
        uint64_t lo = modulus-8192;
        uint64_t hi = modulus - 1;
        dist = std::uniform_int_distribution<uint64_t>(lo, hi);
    }
    
    void TearDown() override {
    }
    
    pim::Vector<uint64_t> createRandomVector(size_t size) {
        pim::Vector<uint64_t> vec(size);
        for (size_t i = 0; i < size; ++i) {
            vec[i] = dist(rng);
        }
        return vec;
    }

    
    uint64_t modAdd(uint64_t a, uint64_t b, uint64_t mod) {
        return (a + b) % mod;
    }
    
    uint64_t modSub(uint64_t a, uint64_t b, uint64_t mod) {
        return (a >= b) ? (a - b) : (mod - (b - a));
    }
    
    uint64_t modMul(uint64_t a, uint64_t b, uint64_t mod) {
        __uint128_t exp128 = (__uint128_t)a * b % mod;
        return (uint64_t)exp128;
    }

protected:
    uint64_t modulus;
    size_t vector_size;
    uint64_t mu; 
    std::mt19937 rng;
    std::uniform_int_distribution<uint64_t> dist;
};

TEST_F(PIM_Test, EltwiseAddMod) {
    auto vec1 = createRandomVector(vector_size);
    auto vec2 = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    
    pim::EltwiseAddMod(result, vec1, vec2, modulus);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = modAdd(vec1[i], vec2[i], modulus);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i 
            << ": got " << result[i] 
            << ", expected " << expected;
    }
}

TEST_F(PIM_Test, EltwiseAddScalarMod) {
    auto vec1 = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    uint64_t scalar = 12345;
    
    pim::EltwiseAddScalarMod(result, vec1, scalar, modulus);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = modAdd(vec1[i], scalar, modulus);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i;
    }
}

TEST_F(PIM_Test, EltwiseSubMod) {
    auto vec1 = createRandomVector(vector_size);
    auto vec2 = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    
    pim::EltwiseSubMod(result, vec1, vec2, modulus);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = modSub(vec1[i], vec2[i], modulus);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i;
    }
}

TEST_F(PIM_Test, EltwiseSubScalarMod) {
    auto vec1 = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    uint64_t scalar = 54321;
    
    pim::EltwiseSubScalarMod(result, vec1, scalar, modulus);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = modSub(vec1[i], scalar, modulus);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i;
    }
}

TEST_F(PIM_Test, EltwiseMulMod) {
    auto vec1 = createRandomVector(vector_size);
    auto vec2 = createRandomVector(vector_size);
    pim::Vector<uint64_t> r1(vector_size);
    
    pim::EltwiseMulMod(r1, vec1, vec2, modulus);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = modMul(vec1[i], vec2[i], modulus);
        ASSERT_EQ(r1[i], expected) 
            << "Mismatch at index " << i;
    }
}

TEST_F(PIM_Test, EltwiseScalarMulMod) {
    auto vec1 = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    uint64_t scalar = 7;
    
    pim::EltwiseScalarMulMod(result, vec1, scalar, modulus);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = modMul(vec1[i], scalar, modulus);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i;
    }
}

TEST_F(PIM_Test, EltwiseFMAMod) {
    auto vec1 = createRandomVector(vector_size);
    auto addend = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    uint64_t scalar = 13;
    
    pim::EltwiseFMAMod(result, vec1, addend, scalar, modulus);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t mul_result = modMul(vec1[i], scalar, modulus);
        uint64_t expected = modAdd(mul_result, addend[i], modulus);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i;
    }
}

TEST_F(PIM_Test, EltwiseConditionalAdd) {
    auto vec1 = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    uint64_t bound = modulus / 2;
    uint64_t diff = 100;
    
    pim::EltwiseConditionalAdd(result, vec1, pim::LESS_THAN, bound, diff);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = vec1[i] + (vec1[i] < bound ? diff : 0);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i;
    }
}
TEST_F(PIM_Test, EltwiseConditionalSubMod) {
    auto vec1 = createRandomVector(vector_size);
    pim::Vector<uint64_t> result(vector_size);
    uint64_t bound = modulus / 2;
    uint64_t diff = 50;
    
    pim::EltwiseConditionalSubMod(result, vec1, modulus, pim::GREATER_EQUAL, bound, diff);
    
    for (size_t i = 0; i < vector_size; ++i) {
        uint64_t expected = modSub(vec1[i], (vec1[i] >= bound ? diff : 0), modulus);
        ASSERT_EQ(result[i], expected) 
            << "Mismatch at index " << i;
    }
}

// TEST_F(PIM_Test, EltwiseReduceMod) {
//     auto vec1 = createRandomVector(vector_size);
//     pim::Vector<uint64_t> result(vector_size);
//     uint64_t mod = modulus;
//     uint32_t in_factor = 2;
//     uint32_t out_factor = 2;
    
//     pim::EltwiseReduceMod(result, vec1, mod, in_factor, out_factor);
    
//     for (size_t i = 0; i < vector_size; ++i) {
//         uint64_t expected = (vec1[i] / in_factor) % mod * out_factor;
//         ASSERT_EQ(result[i], expected) 
//             << "Mismatch at index " << i;
//     }
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
        // Ensure deterministic order
    ::testing::GTEST_FLAG(shuffle) = false;
    ::testing::GTEST_FLAG(repeat) = 1;
    return RUN_ALL_TESTS();
}
