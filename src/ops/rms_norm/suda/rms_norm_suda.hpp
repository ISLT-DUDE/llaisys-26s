#pragma once
#include "../op.hpp"

namespace llaisys::ops::suda {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps);
}