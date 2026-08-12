#pragma once
#include "../op.hpp"

namespace llaisys::ops::suda {
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t dtype, size_t numel);
}