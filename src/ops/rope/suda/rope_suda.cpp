#include "rope_suda.hpp"
namespace llaisys::ops::suda {
void rope(std::byte *, const std::byte *, const int64_t *, float,
          llaisysDataType_t, size_t, size_t, size_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}