#pragma once
#include <cstdint>
#include <mutex>
#include <vector>

namespace aegaeon {

class SlabAllocator {
public:
    SlabAllocator(int gpu_id, std::size_t total_bytes);
    ~SlabAllocator();

    // allocate returns device pointer as uint64_t (uintptr_t)
    uint64_t allocate(std::size_t bytes, std::size_t alignment = 256);
    // free is a no-op in simple bump allocator; we provide reset to reuse whole slab
    void reset();
    // release all device memory
    void destroy();

    int gpu_id() const { return gpu_id_; }
    std::size_t total_bytes() const { return total_bytes_; }

private:
    int gpu_id_;
    std::size_t total_bytes_;
    uint64_t base_ptr_; // device pointer stored as integer
    std::size_t offset_;
    std::mutex mu_;
};

} // namespace aegaeon
