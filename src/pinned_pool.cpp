#include "aegaeon/pinned_pool.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace aegaeon {

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg)+": "+cudaGetErrorString(e));
    }
}

PinnedPool::PinnedPool(std::size_t buffer_size, int pool_size)
    : buffer_size_(buffer_size), pool_size_(pool_size) {
    buffers_.reserve(pool_size_);
    for (int i = 0; i < pool_size_; ++i) {
        void* p = nullptr;
        // allocate page-locked host memory; portable so other processes could access?
        checkCuda(cudaHostAlloc(&p, buffer_size_, cudaHostAllocDefault), "cudaHostAlloc");
        buffers_.push_back(p);
    }
}

PinnedPool::~PinnedPool() {
    for (void* p : buffers_) {
        if (p) {
            cudaFreeHost(p);
        }
    }
    buffers_.clear();
}

void* PinnedPool::acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{ return !buffers_.empty(); });
    void* p = buffers_.back();
    buffers_.pop_back();
    return p;
}

void PinnedPool::release(void* host_ptr) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        buffers_.push_back(host_ptr);
    }
    cv_.notify_one();
}

} // namespace aegaeon
