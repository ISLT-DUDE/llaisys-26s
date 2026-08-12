#include "linear_suda.hpp"
namespace llaisys::ops::suda {
void linear(std::byte *, const std::byte *, const std::byte *, const std::byte *,
            llaisysDataType_t, size_t, size_t, size_t) {
    // Suda operator stub - no physical Suda device
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}