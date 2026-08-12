#include "embedding_suda.hpp"
namespace llaisys::ops::suda {
void embedding(std::byte *, const int64_t *, const std::byte *, llaisysDataType_t, size_t, size_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}
}