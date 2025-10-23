#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdint>

namespace aegaeon {

struct KVEntry {
    uint64_t device_ptr; // device pointer as integer
    std::size_t size;
    void* host_pinned;    // host pinned buffer pointer (may be nullptr)
    // we store an opaque cudaEvent_t pointer (serialized when needed)
    void* event_ptr; // allocated cudaEvent_t* if inflight
};

class KVCache {
public:
    KVCache(int gpu_id);
    ~KVCache();

    // store a KV blob (on host memory) and asynchronously copy to GPU
    // data_bytes pointer copied into pinned buffer and then to device
    void store_kv_async(const std::string& req_id, const void* data_bytes, std::size_t nbytes);

    // ensure KV on GPU (block until copy complete); returns device pointer as uint64_t
    uint64_t ensure_kv_on_gpu(const std::string& req_id);

    // swapout all GPU KV to host pinned buffers (async)
    void swapout_all_async();

    // synchronous wait for all outstanding events
    void wait_all();

private:
    int gpu_id_;
    std::unordered_map<std::string, KVEntry> map_;
    std::mutex mu_;
    // an internal pinned pool for KV
    void* allocate_pinned(std::size_t bytes);
    void free_pinned(void* p);
};

} // namespace aegaeon
