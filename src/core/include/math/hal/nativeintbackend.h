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

#ifndef __NATIVEINTBACKEND_H__
#define __NATIVEINTBACKEND_H__

#include "math/hal/basicint.h"
#include "config_core.h"

// Use CMake option WITH_PIM_HEXL instead of hardcoded value
// Can be enabled with: cmake -DWITH_PIM_HEXL=ON

#ifdef WITH_PIM_HEXL
    #include "math/hal/intnat-pim/ubintnatpim.h"
    #include "math/hal/intnat-pim/mubintvecnatpim.h"
    #include "math/hal/intnat-pim/transformnatpim.h"

namespace lbcrypto {

using NativeInteger = intnatpim::NativeInteger;
using NativeVector  = intnatpim::NativeVector;

}  // namespace lbcrypto

#else
    #include "math/hal/intnat/ubintnat.h"
    #include "math/hal/intnat/mubintvecnat.h"
    #include "math/hal/intnat/transformnat.h"

namespace lbcrypto {

using NativeInteger = intnat::NativeInteger;
using NativeVector  = intnat::NativeVector;

}  // namespace lbcrypto

#endif

using NativeInteger = lbcrypto::NativeInteger;
using NativeVector  = lbcrypto::NativeVector;

#endif
