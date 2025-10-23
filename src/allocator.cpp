#include "aegaeon/allocator.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace aegaeon {

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::string s = std::string(msg) + ": " + cudaGetErrorString(e);
        throw std::runtime_error(s);
    }
}

SlabAllocator::SlabAllocator(int gpu_id, std::size_t total_bytes)
    : gpu_id_(gpu_id), total_bytes_(total_bytes), base_ptr_(0), offset_(0) {
    // set device
    checkCuda(cudaSetDevice(gpu_id_), "cudaSetDevice");
    void* devPtr = nullptr;
    checkCuda(cudaMalloc(&devPtr, total_bytes_), "cudaMalloc slab");
    base_ptr_ = reinterpret_cast<uint64_t>(devPtr);
}

SlabAllocator::~SlabAllocator() {
    destroy();
}

uint64_t SlabAllocator::allocate(std::size_t bytes, std::size_t alignment) {
    std::lock_guard<std::mutex> lk(mu_);
    // simple align-up
    std::size_t cur = offset_;
    std::size_t align_mask = alignment - 1;
    std::size_t aligned = ( (cur + align_mask) & ~align_mask );
    if (aligned + bytes > total_bytes_) {
        throw std::runtime_error("SlabAllocator: out of memory");
    }
    offset_ = aligned + bytes;
    uint64_t ptr = base_ptr_ + aligned;
    return ptr;
}

void SlabAllocator::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    offset_ = 0;
}

void SlabAllocator::destroy() {
    std::lock_guard<std::mutex> lk(mu_);
    if (base_ptr_ != 0) {
        void* p = reinterpret_cast<void*>(base_ptr_);
        // assume current device set to gpu_id_
        cudaError_t e = cudaFree(p);
        if (e != cudaSuccess) {
            // best-effort: print error
            std::cerr << "cudaFree failed: " << cudaGetErrorString(e) << std::endl;
        }
        base_ptr_ = 0;
        offset_ = 0;
    }
}

} // namespace aegaeon
