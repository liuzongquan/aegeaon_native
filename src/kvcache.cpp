#include "aegaeon/kvcache.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace aegaeon {

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }
}

KVCache::KVCache(int gpu_id): gpu_id_(gpu_id) {
    // nothing heavy
}

KVCache::~KVCache() {
    // free pinned & device buffers
    for (auto &kv : map_) {
        if (kv.second.device_ptr) {
            void* p = reinterpret_cast<void*>(kv.second.device_ptr);
            cudaFree(p);
        }
        if (kv.second.host_pinned) {
            cudaFreeHost(kv.second.host_pinned);
        }
        if (kv.second.event_ptr) {
            cudaEvent_t* e = reinterpret_cast<cudaEvent_t*>(kv.second.event_ptr);
            cudaEventDestroy(*e);
            delete e;
        }
    }
}

void* KVCache::allocate_pinned(std::size_t bytes) {
    void* hostp = nullptr;
    checkCuda(cudaHostAlloc(&hostp, bytes, cudaHostAllocDefault), "cudaHostAlloc");
    return hostp;
}

void KVCache::free_pinned(void* p) {
    if (p) cudaFreeHost(p);
}

void KVCache::store_kv_async(const std::string& req_id, const void* data_bytes, std::size_t nbytes) {
    std::lock_guard<std::mutex> lk(mu_);
    // allocate device memory from cudaMalloc (or slab in future)
    checkCuda(cudaSetDevice(gpu_id_), "cudaSetDevice");
    void* devp = nullptr;
    checkCuda(cudaMalloc(&devp, nbytes), "cudaMalloc for KV");

    // allocate pinned host buffer and copy input there
    void* hostp = allocate_pinned(nbytes);
    memcpy(hostp, data_bytes, nbytes);

    // create stream for async copy
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    // async copy
    checkCuda(cudaMemcpyAsync(devp, hostp, nbytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");

    // create event to track completion
    cudaEvent_t* evt = new cudaEvent_t;
    checkCuda(cudaEventCreateWithFlags(evt, cudaEventDisableTiming), "cudaEventCreate");
    checkCuda(cudaEventRecord(*evt, stream), "cudaEventRecord");
    // do not destroy stream; user may reuse; we will let runtime clean when process ends (or we could destroy after event)
    // store metadata
    KVEntry entry;
    entry.device_ptr = reinterpret_cast<uint64_t>(devp);
    entry.size = nbytes;
    entry.host_pinned = hostp;
    entry.event_ptr = reinterpret_cast<void*>(evt);
    map_[req_id] = entry;
    // detach stream (it's fine to leave created streams for each copy in this simple prototype)
}

uint64_t KVCache::ensure_kv_on_gpu(const std::string& req_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = map_.find(req_id);
    if (it == map_.end()) throw std::runtime_error("KV not found");
    KVEntry &e = it->second;
    if (e.event_ptr) {
        cudaEvent_t* evt = reinterpret_cast<cudaEvent_t*>(e.event_ptr);
        // block until event done
        checkCuda(cudaEventSynchronize(*evt), "cudaEventSynchronize");
        // after sync we can destroy event and clear pointer
        cudaEventDestroy(*evt);
        delete evt;
        e.event_ptr = nullptr;
    }
    return e.device_ptr;
}

void KVCache::swapout_all_async() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &p : map_) {
        KVEntry &e = p.second;
        if (e.device_ptr == 0) continue;
        // allocate pinned host buffer
        void* hostp = allocate_pinned(e.size);
        // create stream
        cudaStream_t stream;
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        // async copy D2H
        void* devp = reinterpret_cast<void*>(e.device_ptr);
        checkCuda(cudaMemcpyAsync(hostp, devp, e.size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
        // record event
        cudaEvent_t* evt = new cudaEvent_t;
        checkCuda(cudaEventCreateWithFlags(evt, cudaEventDisableTiming), "cudaEventCreate");
        checkCuda(cudaEventRecord(*evt, stream), "cudaEventRecord");
        // free device memory right away? we should free after event completes; but we'll free and keep hostp
        // To be safe in prototype: synchronize event then free; but we do async: postpone free (not implemented fully)
        // For simplicity, we will do cudaEventSynchronize immediately (blocking) — in production, use daemon to free later.
        checkCuda(cudaEventSynchronize(*evt), "cudaEventSynchronize immediate (swapout)");
        // copy is done — free device memory and store host buffer
        cudaFree(devp);
        if (e.event_ptr) {
            cudaEvent_t* evold = reinterpret_cast<cudaEvent_t*>(e.event_ptr);
            cudaEventDestroy(*evold);
            delete evold;
        }
        e.device_ptr = 0;
        e.host_pinned = hostp;
        e.event_ptr = nullptr;
        // destroy event
        cudaEventDestroy(*evt);
        delete evt;
    }
}

void KVCache::wait_all() {
    // naive: iterate and synchronize any outstanding events
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &p : map_) {
        KVEntry &e = p.second;
        if (e.event_ptr) {
            cudaEvent_t* evt = reinterpret_cast<cudaEvent_t*>(e.event_ptr);
            checkCuda(cudaEventSynchronize(*evt), "cudaEventSynchronize wait_all");
            cudaEventDestroy(*evt);
            delete evt;
            e.event_ptr = nullptr;
        }
    }
}

} // namespace aegaeon
