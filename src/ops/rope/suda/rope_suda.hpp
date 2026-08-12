#pragma once
#include "../op.hpp"

namespace llaisys::ops::suda {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta,
          llaisysDataType_t dtype, size_t S, size_t H, size_t D);
}