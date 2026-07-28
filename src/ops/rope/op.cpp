#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#include <cmath>

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t num_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    // Compute cos and sin from pos_ids and theta
    // cos[i][j] = cos(pos_ids[i] / theta^(2j/head_dim))
    // sin[i][j] = sin(pos_ids[i] / theta^(2j/head_dim))
    size_t half_dim = head_dim / 2;
    std::vector<float> cos_buf(seq_len * head_dim);
    std::vector<float> sin_buf(seq_len * head_dim);

    // Read pos_ids as int64_t
    const int64_t *pos_data = reinterpret_cast<const int64_t *>(pos_ids->data());
    for (size_t i = 0; i < seq_len; i++) {
        float pos = static_cast<float>(pos_data[i]);
        for (size_t j = 0; j < half_dim; j++) {
            float freq = pos / std::pow(theta, 2.0f * j / head_dim);
            float c = std::cos(freq);
            float s = std::sin(freq);
            cos_buf[i * head_dim + j] = c;
            cos_buf[i * head_dim + j + half_dim] = c; // cos repeats for second half
            sin_buf[i * head_dim + j] = s;
            sin_buf[i * head_dim + j + half_dim] = s; // sin repeats for second half
        }
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(),
                         reinterpret_cast<const std::byte *>(cos_buf.data()),
                         reinterpret_cast<const std::byte *>(sin_buf.data()),
                         out->dtype(), seq_len, num_heads, head_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(),
                         reinterpret_cast<const std::byte *>(cos_buf.data()),
                         reinterpret_cast<const std::byte *>(sin_buf.data()),
                         out->dtype(), seq_len, num_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
