#pragma once
#include "../op.hpp"

namespace llaisys::ops::suda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t N);
}