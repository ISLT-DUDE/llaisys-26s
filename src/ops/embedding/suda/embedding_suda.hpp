#pragma once
#include "../op.hpp"

namespace llaisys::ops::suda {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t N, size_t H);
}