#pragma once
#include <cstddef>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace aegaeon {

class PinnedPool {
public:
    PinnedPool(std::size_t buffer_size, int pool_size = 4);
    ~PinnedPool();

    // acquire returns host pointer (void*) to a pinned buffer
    void* acquire();
    void release(void* host_ptr);

    std::size_t buffer_size() const { return buffer_size_; }

private:
    std::size_t buffer_size_;
    int pool_size_;
    std::vector<void*> buffers_;
    std::mutex mu_;
    std::condition_variable cv_;
};

} // namespace aegaeon
