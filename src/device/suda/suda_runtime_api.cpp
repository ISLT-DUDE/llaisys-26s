#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::suda {

namespace runtime_api {

int getDeviceCount() {
    return 0;  // No physical Suda device available
}

void setDevice(int) {
    // Stub
}

void deviceSynchronize() {
    // Stub
}

llaisysStream_t createStream() {
    return nullptr;
}

void destroyStream(llaisysStream_t) {
    // Stub
}

void streamSynchronize(llaisysStream_t) {
    // Stub
}

void *mallocDevice(size_t) {
    return nullptr;
}

void freeDevice(void *) {
    // Stub
}

void *mallocHost(size_t size) {
    return std::malloc(size);
}

void freeHost(void *ptr) {
    std::free(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t) {
    std::memcpy(dst, src, size);  // default H2H
}

void memcpyAsync(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t) {
    // Stub
}


static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::suda