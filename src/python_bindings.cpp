#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <vector>
#include <map>
#include <mutex>
#include <cstring>
#include <iostream>

#include "aegaeon/allocator.h"
#include "aegaeon/pinned_pool.h"
#include "aegaeon/ipc_event.h"
#include "aegaeon/kvcache.h"

#include <cuda_runtime.h>

namespace py = pybind11;
using namespace aegaeon;

struct PrefetchChunk {
    void* device_ptr;
    std::size_t size;
    int slab_id;
    cudaEvent_t* evt;
};

static std::map<int, std::vector<PrefetchChunk>> g_prefetch_chunks;
static int g_next_slab_id = 0;
static std::mutex g_prefetch_mu;

PYBIND11_MODULE(aegaeon_native, m) {
    m.doc() = "Aegaeon native module with SlabAllocator, PinnedPool and Prefetch management";

    py::class_<SlabAllocator>(m, "SlabAllocator")
        .def(py::init<int, std::size_t>())
        .def("allocate", &SlabAllocator::allocate)
        .def("reset", &SlabAllocator::reset)
        .def("destroy", &SlabAllocator::destroy)
        .def_property_readonly("gpu_id", &SlabAllocator::gpu_id)
        .def_property_readonly("total_bytes", &SlabAllocator::total_bytes);

    py::class_<PinnedPool>(m, "PinnedPool")
        .def(py::init<std::size_t, int>(), py::arg("buffer_size"), py::arg("pool_size") = 4)
        .def("acquire", [](PinnedPool &p) {
            return reinterpret_cast<uint64_t>(p.acquire());
        })
        .def("release", [](PinnedPool &p, uint64_t addr) {
            p.release(reinterpret_cast<void*>(addr));
        })
        .def_property_readonly("buffer_size", &PinnedPool::buffer_size);

    // KVCache binding
    py::class_<KVCache>(m, "KVCache")
        .def(py::init<int>())
        .def("store_kv_async", [](KVCache &k, const std::string &req_id, py::bytes data) {
            std::string s = data;
            k.store_kv_async(req_id, s.data(), s.size());
        })
        .def("ensure_kv_on_gpu", &KVCache::ensure_kv_on_gpu)
        .def("swapout_all_async", &KVCache::swapout_all_async)
        .def("wait_all", &KVCache::wait_all);

    // IPC helpers
    m.def("create_event_handle_bytes", []() {
        auto v = create_event_and_get_handle();
        return py::bytes(v.data(), v.size());
    });

    m.def("wait_event_handle_bytes", [](py::bytes b) {
        std::string s = b;
        std::vector<char> v(s.begin(), s.end());
        wait_event_ipc_handle_bytes(v);
    });

    // Prefetch
    m.def("prefetch_file_to_gpu_chunks",
        [](const std::string &filepath, int gpu_id, std::size_t chunk_bytes) {
            std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
            if (!ifs.is_open()) {
                throw std::runtime_error("cannot open file: " + filepath);
            }
            std::streamsize file_size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);

            // slab allocator: one slab per call
            int slab_id;
            {
                std::lock_guard<std::mutex> lk(g_prefetch_mu);
                slab_id = g_next_slab_id++;
            }
            std::unique_ptr<SlabAllocator> slab = std::make_unique<SlabAllocator>(gpu_id, file_size);
            PinnedPool pinned(chunk_bytes, 2);
            cudaStream_t stream;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

            std::vector<py::dict> results;
            std::size_t offset = 0;
            while (offset < (size_t)file_size) {
                std::size_t to_read = std::min(chunk_bytes, (size_t)file_size - offset);
                void* hostp = reinterpret_cast<void*>(pinned.acquire());
                ifs.read(reinterpret_cast<char*>(hostp), to_read);
                std::streamsize got = ifs.gcount();
                if (got <= 0) break;

                void* devp = slab->allocate(to_read);
                cudaMemcpyAsync(devp, hostp, to_read, cudaMemcpyHostToDevice, stream);

                cudaEvent_t* evt = new cudaEvent_t;
                cudaEventCreateWithFlags(evt, cudaEventDisableTiming);
                cudaEventRecord(*evt, stream);

                cudaIpcEventHandle_t handle;
                cudaIpcGetEventHandle(&handle, *evt);
                std::vector<char> hb(sizeof(handle));
                memcpy(hb.data(), &handle, sizeof(handle));
                py::bytes pyhb(hb.data(), hb.size());

                py::dict d;
                d["device_ptr"] = (uint64_t)devp;
                d["size"] = (size_t)to_read;
                d["event_handle"] = pyhb;
                d["slab_id"] = slab_id;
                results.push_back(d);

                // register chunk
                {
                    std::lock_guard<std::mutex> lk(g_prefetch_mu);
                    g_prefetch_chunks[slab_id].push_back({devp, to_read, slab_id, evt});
                }
                pinned.release(reinterpret_cast<uint64_t>(hostp));
                offset += to_read;
            }

            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            return results;
        },
        py::arg("filepath"), py::arg("gpu_id"), py::arg("chunk_bytes") = 32 * 1024 * 1024
    );

    m.def("release_prefetch_chunks", [](std::vector<int> slab_ids) {
        std::lock_guard<std::mutex> lk(g_prefetch_mu);
        for (int id : slab_ids) {
            if (g_prefetch_chunks.count(id)) {
                for (auto &c : g_prefetch_chunks[id]) {
                    if (c.evt) {
                        cudaEventDestroy(*c.evt);
                        delete c.evt;
                    }
                    cudaFree(c.device_ptr);
                }
                g_prefetch_chunks.erase(id);
            }
        }
    });

    m.def("release_all_prefetch_events", []() {
        std::lock_guard<std::mutex> lk(g_prefetch_mu);
        for (auto &[id, chunks] : g_prefetch_chunks) {
            for (auto &c : chunks) {
                if (c.evt) {
                    cudaEventDestroy(*c.evt);
                    delete c.evt;
                }
                cudaFree(c.device_ptr);
            }
        }
        g_prefetch_chunks.clear();
    });
}
