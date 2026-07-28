#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int32_t *indices, const T *weight, size_t seq_len, size_t embed_dim, size_t vocab_size) {
    for (size_t i = 0; i < seq_len; i++) {
        int32_t idx = indices[i];
        const T *src_row = weight + idx * embed_dim;
        T *dst_row = out + i * embed_dim;
        std::memcpy(dst_row, src_row, embed_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *indices, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t embed_dim, size_t vocab_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int32_t *>(indices),
                          reinterpret_cast<const float *>(weight), seq_len, embed_dim, vocab_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int32_t *>(indices),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), seq_len, embed_dim, vocab_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int32_t *>(indices),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), seq_len, embed_dim, vocab_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
