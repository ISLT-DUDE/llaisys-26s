#include "add_suda.hpp"
namespace llaisys::ops::suda {
void add(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t) {
    // Suda operator stub - no physical Suda device
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}