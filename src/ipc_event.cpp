#include "aegaeon/ipc_event.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace aegaeon {

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }
}

std::vector<char> get_event_ipc_handle_bytes(void* raw_event_ptr) {
    // raw_event_ptr is expected to be a pointer to cudaEvent_t (opaque)
    cudaEvent_t event = *reinterpret_cast<cudaEvent_t*>(raw_event_ptr);
    cudaIpcEventHandle_t handle;
    cudaError_t e = cudaIpcGetEventHandle(&handle, event);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cudaIpcGetEventHandle failed: ") + cudaGetErrorString(e));
    }
    std::vector<char> out(sizeof(handle));
    std::memcpy(out.data(), &handle, sizeof(handle));
    return out;
}

std::vector<char> create_event_and_get_handle() {
    cudaEvent_t evt;
    checkCuda(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    cudaIpcEventHandle_t handle;
    checkCuda(cudaIpcGetEventHandle(&handle, evt), "cudaIpcGetEventHandle");
    std::vector<char> out(sizeof(handle));
    std::memcpy(out.data(), &handle, sizeof(handle));
    // Destroy the local event (we only returned the IPC handle)
    cudaEventDestroy(evt);
    return out;
}

void wait_event_ipc_handle_bytes(const std::vector<char>& handle_bytes) {
    if (handle_bytes.size() != sizeof(cudaIpcEventHandle_t)) {
        throw std::runtime_error("Invalid handle size");
    }
    cudaIpcEventHandle_t handle;
    std::memcpy(&handle, handle_bytes.data(), sizeof(handle));
    cudaEvent_t evt;
    cudaError_t e = cudaIpcOpenEventHandle(&evt, handle);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cudaIpcOpenEventHandle failed: ") + cudaGetErrorString(e));
    }
    // wait
    cudaEventSynchronize(evt);
    cudaEventDestroy(evt);
}

} // namespace aegaeon
