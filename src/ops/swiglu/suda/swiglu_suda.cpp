#include "swiglu_suda.hpp"
namespace llaisys::ops::suda {
void swiglu(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}