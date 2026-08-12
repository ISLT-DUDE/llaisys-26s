#include "self_attention_suda.hpp"
namespace llaisys::ops::suda {
void self_attention(std::byte *, const std::byte *, const std::byte *, const std::byte *,
                    llaisysDataType_t, size_t, size_t, size_t, size_t, size_t, float) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}