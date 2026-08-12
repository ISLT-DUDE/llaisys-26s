#include "argmax_suda.hpp"
namespace llaisys::ops::suda {
void argmax(std::byte *, std::byte *, const std::byte *, llaisysDataType_t, size_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}