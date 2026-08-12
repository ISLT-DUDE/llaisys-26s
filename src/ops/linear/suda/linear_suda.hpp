#pragma once
#include "../op.hpp"

namespace llaisys::ops::suda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t B, size_t K, size_t N);
}