#include "rms_norm_suda.hpp"
namespace llaisys::ops::suda {
void rms_norm(std::byte *, const std::byte *, const std::byte *,
              llaisysDataType_t, size_t, size_t, float) {
    // Suda operator stub - no physical Suda device
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}