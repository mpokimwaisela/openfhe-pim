//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  This code provides basic arithmetic functionality for vectors of native integers
 */

#include "math/math-hal.h"
#include "math/hal/intnat-pim/mubintvecnatpim.h"
#include "math/nbtheory-impl.h"
#include "utils/exception.h"

namespace intnatpim {

template <class IntegerType>
NativeVectorT<IntegerType>::NativeVectorT(uint32_t length, const IntegerType& modulus,
                                          std::initializer_list<std::string> rhs) noexcept
    : m_modulus{modulus}, m_data(length) {
#ifdef WITH_PIM_HEXL
    setup_pim_serialization();
#endif
    const uint32_t vlen = (rhs.size() < m_data.size()) ? rhs.size() : m_data.size();
    for (uint32_t i = 0; i < vlen; ++i)
        m_data[i] = *(rhs.begin() + i) % m_modulus;
}

template <class IntegerType>
NativeVectorT<IntegerType>::NativeVectorT(uint32_t length, const IntegerType& modulus,
                                          std::initializer_list<uint64_t> rhs) noexcept
    : m_modulus{modulus}, m_data(length) {
#ifdef WITH_PIM_HEXL
    setup_pim_serialization();
#endif
    const uint32_t vlen = (rhs.size() < m_data.size()) ? rhs.size() : m_data.size();
    for (uint32_t i = 0; i < vlen; ++i)
        m_data[i].m_value = BasicInt(*(rhs.begin() + i)) % m_modulus.m_value;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::operator=(std::initializer_list<std::string> rhs) noexcept {
    const size_t vlen = rhs.size();
    if (m_data.size() < vlen)
        m_data.resize(vlen);
    for (size_t i = 0; i < m_data.size(); ++i) {
        if (i < vlen) {
            m_data[i] = *(rhs.begin() + i);
            if (m_modulus.m_value != 0)
                m_data[i].m_value = m_data[i].m_value % m_modulus.m_value;
        }
        else {
            m_data[i].m_value = 0;
        }
    }
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::operator=(std::initializer_list<uint64_t> rhs) noexcept {
    const size_t vlen = rhs.size();
    if (m_data.size() < vlen)
        m_data.resize(vlen);
    for (size_t i = 0; i < m_data.size(); ++i) {
        if (i < vlen) {
            m_data[i].m_value = BasicInt(*(rhs.begin() + i));
            if (m_modulus.m_value != 0)
                m_data[i].m_value = m_data[i].m_value % m_modulus.m_value;
        }
        else {
            m_data[i].m_value = 0;
        }
    }
    return *this;
}

/**Switches the integers in the vector to values corresponding to the new
 * modulus.
 * Algorithm: Integer i, Old Modulus om, New Modulus nm,
 * delta = abs(om-nm):
 *  Case 1: om < nm
 *    if i > om/2
 *      i' = i + delta
 *  Case 2: om > nm
 *    i > om/2 i' = i-delta
 */
template <class IntegerType>
void NativeVectorT<IntegerType>::SwitchModulus(const IntegerType& modulus) {
    #ifdef WITH_PIM_HEXL
        if (UsePIMAcceleration()) {
            IntegerType halfQ{m_modulus >> 1};
            IntegerType diff{(m_modulus > modulus) ? (m_modulus - modulus) : (modulus - m_modulus)};
            auto temp = *this;
            if (modulus > m_modulus) {
                
                pim::EltwiseConditionalAdd(m_data, temp.m_data,
                                         pim::GREATER_THAN,
                                         halfQ.ConvertToInt(),
                                         diff.ConvertToInt());
            } else {
                pim::EltwiseConditionalSubMod(m_data, temp.m_data,
                                            modulus.ConvertToInt(),
                                            pim::GREATER_THAN,
                                            halfQ.ConvertToInt(),
                                            diff.ConvertToInt());
            }
            this->SetModulus(modulus);
            return;
        }
    #endif

    // Original CPU fallback code
    IntegerType halfQ{m_modulus >> 1};
    IntegerType diff{(m_modulus > modulus) ? (m_modulus - modulus) : (modulus - m_modulus)};
    if (modulus > m_modulus) {
        for (auto& v : m_data)
            v.AddEqFast((v > halfQ) ? diff : 0);
    }
    else {
        for (auto& v : m_data)
            v.ModSubEq((v > halfQ) ? diff : 0, modulus);
    }
    this->SetModulus(modulus);
}

template <class IntegerType>
void NativeVectorT<IntegerType>::LazySwitchModulus(const IntegerType& modulus) {
    for (auto& v : m_data)
        v.ModEq(modulus);
    this->SetModulus(modulus);
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::MultAccEqNoCheck(const NativeVectorT& V, const IntegerType& I) {
    auto iv{I};
    auto mv{m_modulus};
    if (iv.m_value >= mv.m_value)
        iv.ModEq(mv);
    auto iinv{iv.PrepModMulConst(mv)};
    const uint32_t ringdm = m_data.size();
    for (uint32_t i = 0; i < ringdm; ++i)
        m_data[i].ModAddFastEq(V.m_data[i].ModMulFastConst(iv, mv, iinv), mv);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::Mod(const IntegerType& modulus) const {
    auto ans(*this);
    if (modulus.m_value == 2)
        return ans.ModByTwoEq();
    IntegerType halfQ{m_modulus >> 1};
    IntegerType diff{(m_modulus > modulus) ? (m_modulus - modulus) : (modulus - m_modulus)};

    if (modulus > m_modulus) {
        for (auto& v : ans.m_data)
            v.AddEqFast((v > halfQ) ? diff : 0);
    }
    else {
        for (auto& v : ans.m_data)
            v.ModSubEq((v > halfQ) ? diff : 0, modulus);
    }
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModEq(const IntegerType& modulus) {
    if (modulus.m_value == 2)
        return this->NativeVectorT::ModByTwoEq();
    IntegerType halfQ{m_modulus >> 1};
    IntegerType diff{(m_modulus > modulus) ? (m_modulus - modulus) : (modulus - m_modulus)};

    if (modulus > m_modulus) {
        for (auto& v : m_data)
            v.AddEqFast((v > halfQ) ? diff : 0);
    }
    else {
        for (auto& v : m_data)
            v.ModSubEq((v > halfQ) ? diff : 0, modulus);
    }
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModAdd(const IntegerType& b) const {
#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        NativeVectorT ans(m_data.size(), m_modulus);
        pim::EltwiseAddScalarMod(ans.m_data, m_data, b.ConvertToInt(), m_modulus.ConvertToInt());
        return ans;
    }
#endif
    auto ans(*this);
    auto mv{m_modulus};
    auto bv{b};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);

    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans.m_data[i] = ans.m_data[i].ModAddFast(bv, mv);
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModAddEq(const IntegerType& b) {
#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        auto temp = *this;
        pim::EltwiseAddScalarMod(temp.m_data, m_data, b.ConvertToInt(), m_modulus.ConvertToInt());
        return *this;
    }
#endif

    auto mv{m_modulus};
    auto bv{b};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    // Fallback to CPU implementation
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i] = m_data[i].ModAddFast(bv, mv);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModAddAtIndex(size_t i, const IntegerType& b) const {
    auto ans(*this);
    ans.at(i).ModAddEq(b, m_modulus);
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModAddAtIndexEq(size_t i, const IntegerType& b) {
    this->NativeVectorT::at(i).ModAddEq(b, m_modulus);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModAdd(const NativeVectorT& b) const {
    if (m_modulus != b.m_modulus || m_data.size() != b.m_data.size())
        OPENFHE_THROW("ModAdd called on NativeVectorT's with different parameters.");

#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        NativeVectorT ans(m_data.size(), m_modulus);

        pim::EltwiseAddMod(ans.m_data, m_data, b.m_data, m_modulus.ConvertToInt());
        return ans;
    }
#endif

    auto mv{m_modulus};
    auto ans(*this);

    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans.m_data[i].ModAddFastEq(b[i], mv);
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModAddEq(const NativeVectorT& b) {
    if (m_data.size() != b.m_data.size() || m_modulus != b.m_modulus)
        OPENFHE_THROW("ModAddEq called on NativeVectorT's with different parameters.");
#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        auto temp(*this);
        pim::EltwiseAddMod(m_data, temp.m_data, b.m_data, m_modulus.ConvertToInt());
        return *this;
    }
#endif
    // Fallback to CPU implementation
    auto mv{m_modulus};
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i].ModAddFastEq(b[i], mv);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModSub(const IntegerType& b) const {
#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        NativeVectorT ans(m_data.size(), m_modulus);
        pim::EltwiseSubScalarMod(ans.m_data, m_data, b.ConvertToInt(), m_modulus.ConvertToInt());
        return ans;
    }
#endif
    auto mv{m_modulus};
    auto bv{b};
    auto ans(*this);
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);

    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].ModSubFastEq(bv, mv);
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModSubEq(const IntegerType& b) {
#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        auto temp(*this);
        pim::EltwiseSubScalarMod(m_data, temp.m_data, b.ConvertToInt(), m_modulus.ConvertToInt());
        return *this;
    }
#endif

    auto mv{m_modulus};
    auto bv{b};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i].ModSubFastEq(bv, mv);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModSub(const NativeVectorT& b) const {
    if (m_data.size() != b.m_data.size() || m_modulus != b.m_modulus)
        OPENFHE_THROW("ModSub called on NativeVectorT's with different parameters.");

#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        NativeVectorT ans(m_data.size(), m_modulus);
        pim::EltwiseSubMod(ans.m_data, m_data, b.m_data, m_modulus.ConvertToInt());
        return ans;
    }
#endif

    auto mv{m_modulus};
    auto ans(*this);

    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].ModSubFastEq(b[i], mv);
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModSubEq(const NativeVectorT& b) {
    if (m_data.size() != b.m_data.size() || m_modulus != b.m_modulus)
        OPENFHE_THROW("ModSubEq called on NativeVectorT's with different parameters.");

#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        auto temp(*this);
        pim::EltwiseSubMod(m_data, temp.m_data, b.m_data, m_modulus.ConvertToInt());
        return *this;
    }
#endif

    // Fallback to CPU implementation
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i].ModSubFastEq(b[i], m_modulus);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModMul(const IntegerType& b) const {
#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        NativeVectorT ans(m_data.size(), m_modulus);
        pim::EltwiseScalarMulMod(ans.m_data, m_data, b.ConvertToInt(), m_modulus.ConvertToInt());
        return ans;
    }
#endif

    auto mv{m_modulus};
    auto bv{b};
    auto ans(*this);
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    auto bconst{bv.PrepModMulConst(mv)};

    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].ModMulFastConstEq(bv, mv, bconst);
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModMulEq(const IntegerType& b) {
    auto mv{m_modulus};
    auto bv{b};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);

#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        auto temp(*this);
        pim::EltwiseScalarMulMod(m_data, temp.m_data, bv.ConvertToInt(), mv.ConvertToInt());
        return *this;
    }
#endif

    auto bconst{bv.PrepModMulConst(mv)};
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i].ModMulFastConstEq(bv, mv, bconst);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModMul(const NativeVectorT& b) const {
    if (m_data.size() != b.m_data.size() || m_modulus != b.m_modulus)
        OPENFHE_THROW("ModMul called on NativeVectorT's with different parameters.");

#ifdef WITH_PIM_HEXL
    if (UsePIMAcceleration()) {
        NativeVectorT ans(m_data.size(), m_modulus);
        pim::EltwiseMulMod(ans.m_data, m_data, b.m_data, m_modulus.ConvertToInt());
        return ans;
    }
#endif

    auto ans(*this);
    uint32_t size(m_data.size());
    auto mv{m_modulus};

#ifdef NATIVEINT_BARRET_MOD
    auto mu{m_modulus.ComputeMu()};
    for (uint32_t i = 0; i < size; ++i)
        ans[i].ModMulFastEq(b[i], mv, mu);
#else
    for (uint32_t i = 0; i < size; ++i)
        ans[i].ModMulFastEq(b[i], mv);
#endif
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModMulEq(const NativeVectorT& b) {
    if (m_data.size() != b.m_data.size() || m_modulus != b.m_modulus)
        OPENFHE_THROW("ModMulEq called on NativeVectorT's with different parameters.");

    #ifdef WITH_PIM_HEXL
        if (UsePIMAcceleration()) {
            pim::EltwiseMulMod(m_data, b.m_data, m_data, m_modulus.ConvertToInt());
            return *this;
        }
    #endif

    // Fallback to CPU implementation
    auto mv{m_modulus};
    size_t size{m_data.size()};
#ifdef NATIVEINT_BARRET_MOD
    auto mu{m_modulus.ComputeMu()};
    for (size_t i = 0; i < size; ++i)
        m_data[i].ModMulFastEq(b[i], mv, mu);
#else
    for (size_t i = 0; i < size; ++i)
        m_data[i].ModMulFastEq(b[i], mv);
#endif
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModByTwo() const {
    auto ans(*this);
    auto halfQ{m_modulus.m_value >> 1};
    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].m_value = 0x1 & (ans[i].m_value ^ (ans[i].m_value > halfQ));
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModByTwoEq() {
    auto halfQ{m_modulus.m_value >> 1};
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i].m_value = 0x1 & (m_data[i].m_value ^ (m_data[i].m_value > halfQ));
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModExp(const IntegerType& b) const {
    auto mv{m_modulus};
    auto bv{b};
    auto ans(*this);
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i] = ans[i].ModExp(bv, mv);
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModExpEq(const IntegerType& b) {
    auto mv{m_modulus};
    auto bv{b};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i] = m_data[i].ModExp(bv, mv);
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::MultWithOutMod(const NativeVectorT& b) const {
    if (m_data.size() != b.m_data.size() || m_modulus != b.m_modulus)
        OPENFHE_THROW("ModMul called on NativeVectorT's with different parameters.");
    auto ans(*this);
    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].m_value = ans[i].m_value * b[i].m_value;
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::MultiplyAndRound(const IntegerType& p,
                                                                        const IntegerType& q) const {
    auto halfQ{m_modulus.m_value >> 1};
    auto mv{m_modulus};
    auto ans(*this);
    for (size_t i = 0; i < ans.m_data.size(); ++i) {
        if (ans[i].m_value > halfQ) {
            auto&& tmp{mv - ans[i]};
            ans[i] = mv - tmp.MultiplyAndRound(p, q);
        }
        else {
            ans[i] = ans[i].MultiplyAndRound(p, q).Mod(mv);
        }
    }
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::MultiplyAndRoundEq(const IntegerType& p, const IntegerType& q) {
    auto halfQ{m_modulus.m_value >> 1};
    auto mv{m_modulus};
    for (size_t i = 0; i < m_data.size(); ++i) {
        if (m_data[i].m_value > halfQ) {
            auto&& tmp{mv - m_data[i]};
            m_data[i] = mv - tmp.MultiplyAndRound(p, q);
        }
        else {
            m_data[i] = m_data[i].MultiplyAndRound(p, q).Mod(mv);
        }
    }
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::DivideAndRound(const IntegerType& q) const {
    auto halfQ{m_modulus.m_value >> 1};
    auto mv{m_modulus};
    auto ans(*this);
    for (size_t i = 0; i < ans.m_data.size(); ++i) {
        if (ans[i].m_value > halfQ) {
            auto&& tmp{mv - ans[i]};
            ans[i] = mv - tmp.DivideAndRound(q);
        }
        else {
            ans[i] = ans[i].DivideAndRound(q);
        }
    }
    return ans;
}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::DivideAndRoundEq(const IntegerType& q) {
    auto halfQ{m_modulus.m_value >> 1};
    auto mv{m_modulus};
    for (size_t i = 0; i < m_data.size(); ++i) {
        if (m_data[i].m_value > halfQ) {
            auto&& tmp{mv - m_data[i]};
            m_data[i] = mv - tmp.DivideAndRound(q);
        }
        else {
            m_data[i] = m_data[i].DivideAndRound(q);
        }
    }
    return *this;
}

template <class IntegerType>
NativeVectorT<IntegerType> NativeVectorT<IntegerType>::GetDigitAtIndexForBase(uint32_t index, uint32_t base) const {
    auto ans(*this);
    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].m_value = static_cast<BasicInt>(ans[i].GetDigitAtIndexForBase(index, base));
    return ans;
}

template class NativeVectorT<NativeInteger>;

}  // namespace intnatpim