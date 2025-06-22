#pragma once
#include "common.h"
#include <cstdint>
#include <cstring>
#include <iostream>


static inline dpu_array_t
make_array(uint32_t off, uint32_t elems,
           uint32_t elem_size_bytes = sizeof(uint64_t))
{
    return dpu_array_t{
        .offset        = off,
        .size          = elems,
        .size_in_bytes = elems * elem_size_bytes
    };
}

class ArgsBuilder {
    dpu_arguments_t a_{};

public:
    /* fluent setters --------------------------------------------------- */
    ArgsBuilder& A(uint32_t off, uint32_t elems) { a_.A = make_array(off, elems); return *this; }
    ArgsBuilder& B(uint32_t off, uint32_t elems) { a_.B = make_array(off, elems); return *this; }
    ArgsBuilder& C(uint32_t off, uint32_t elems) { a_.C = make_array(off, elems); return *this; }

    ArgsBuilder& kernel(pimop_t k)          { a_.kernel = k; return *this; }
    ArgsBuilder& mod(uint64_t m)            { a_.mod    = m; return *this; }
    ArgsBuilder& scalar(uint64_t s)         { a_.scalar = s; return *this; }
    ArgsBuilder& cmp(cmp_t c)               { a_.cmp    = c; return *this; }
    ArgsBuilder& bound(uint64_t b)          { a_.bound  = b; return *this; }

    ArgsBuilder& mod_factor(uint32_t f)     { a_.mod_factor        = f; return *this; }
    ArgsBuilder& in_factor(uint32_t f)      { a_.input_mod_factor  = f; return *this; }
    ArgsBuilder& out_factor(uint32_t f)     { a_.output_mod_factor = f; return *this; }

    /* finalise --------------------------------------------------------- */
    [[nodiscard]] dpu_arguments_t build() const { return a_; }
};

inline std::ostream& operator<<(std::ostream& os, const dpu_array_t& arr) {
    os << "{ offset=" << arr.offset
       << ", size=" << arr.size
       << ", bytes=" << arr.size_in_bytes
       << " }";
    return os;
}

inline void debug_print_args(const dpu_arguments_t& args, std::ostream& os = std::cout) {
    os << "DPU args:\n";
    os << "  A = " << args.A << "\n";
    os << "  B = " << args.B << "\n";
    os << "  C = " << args.C << "\n";

    os << "  kernel = ";
    switch (args.kernel) {
      case MOD_ADD:           os << "MOD_ADD"; break;
      case MOD_ADD_SCALAR:    os << "MOD_ADD_SCALAR"; break;
      case CMP_ADD:           os << "CMP_ADD"; break;
      case CMP_SUB_MOD:       os << "CMP_SUB_MOD"; break;
      case FMA_MOD:           os << "FMA_MOD"; break;
      case MOD_SUB:           os << "MOD_SUB"; break;
      case MOD_SUB_SCALAR:    os << "MOD_SUB_SCALAR"; break;
      case MOD_MUL:           os << "MOD_MUL"; break;
      case MOD_REDUCE:        os << "MOD_REDUCE"; break;
      case NTT_STAGE:        os << "MOD_REDUCE"; break;
    }
    os << "\n";

    os << "  mod = " << args.mod << "\n";
    os << "  scalar = " << args.scalar << "\n";

    os << "  cmp = ";
    switch (args.cmp) {
      case CMP_EQ:    os << "CMP_EQ"; break;
      case CMP_NE:    os << "CMP_NE"; break;
      case CMP_LT:    os << "CMP_LT"; break;
      case CMP_LE:    os << "CMP_LE"; break;
      case CMP_NLT:   os << "CMP_NLT"; break;
      case CMP_NLE:   os << "CMP_NLE"; break;
      case CMP_TRUE:  os << "CMP_TRUE"; break;
      case CMP_FALSE: os << "CMP_FALSE"; break;
    }
    os << "\n";

    os << "  bound = " << args.bound << "\n";
    os << "  mod_factor = " << args.mod_factor << "\n";
    os << "  input_mod_factor = " << args.input_mod_factor << "\n";
    os << "  output_mod_factor = " << args.output_mod_factor << "\n";
}