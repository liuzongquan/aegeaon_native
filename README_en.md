
# Aegaeon Native (aegaeon_native) — README (EN)

## Overview
`aegaeon_native` is the native C++/CUDA extension that implements low-level, high-performance primitives used by the Aegaeon prototype:
- GPU slab allocator (allocate large device slabs and sub-allocate chunks)
- Pinned host memory pool (reusable page-locked buffers)
- Async chunked prefetch from file to GPU (cudaMemcpyAsync)
- CUDA event creation + IPC handle export/import (cudaIpcGetEventHandle / cudaIpcOpenEventHandle)
- KV cache primitives (async H2D / D2H, event-based synchronization)
- Python bindings via pybind11

This module is intended to be built on each GPU node and imported by the Python prototype (`aegaeon_proto`).

## Directory layout
```

aegaeon_native/
├── CMakeLists.txt
├── setup.py / pyproject.toml
├── include/aegaeon/*.h
└── src/*.cpp

````

## System requirements
- Debian 12 (or similar Linux)
- Python 3.10 (for building & testing)
- CUDA toolkit & driver (11.x or 12.x; ensure driver matches)
- CMake >= 3.18
- A compiler supporting C++17 (g++ >= 9 recommended)
- `pybind11` (CMake find_package or pip-installed)
- `pip`, `setuptools`, `wheel`

## Python dependencies (for building)
You can install the build-time Python deps in your venv:
```bash
pip install pybind11 cmake wheel setuptools
````

## Build & Install (pip wheel via setup.py)

Two options:

### Option A: Build via `setup.py` (recommended for pip wheel)

```bash
# from project root (aegaeon_native/)
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip
pip install pybind11 cmake wheel setuptools
python setup.py bdist_wheel
# Then install wheel into your Python environment
pip install dist/*.whl
```

### Option B: Build with CMake directly (dev)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
# copy built module (aegaeon_native*.so) into Python path or project root
cp aegaeon_native*.so ../
```

## API highlights (Python bindings)

After install, `import aegaeon_native` and you will have:

* `aegaeon_native.SlabAllocator(gpu_id, total_bytes)`
* `aegaeon_native.PinnedPool(buffer_size, pool_size)`
* `aegaeon_native.KVCache(gpu_id)`
* `aegaeon_native.create_event_handle_bytes()` — create an event and return IPC handle bytes
* `aegaeon_native.wait_event_handle_bytes(bytes)` — open IPC handle and synchronize
* `aegaeon_native.prefetch_file_to_gpu_chunks(filepath, gpu_id, chunk_bytes)` — chunked async H2D with events, returns list of dicts: `{device_ptr, size, event_handle, slab_id}`
* `aegaeon_native.release_prefetch_chunks([slab_ids])` — release device memory & events for given slab ids
* `aegaeon_native.release_all_prefetch_events()` — release all prefetch allocations

> Note: `device_ptr` returned is a raw device pointer (uint64). If you need to share device memory across processes, use CUDA IPC memory handles (not fully implemented in simple prototype). Event IPC handles are supported for synchronization across processes on the same node.

## Usage example (Python)

```python
import aegaeon_native as an

chunks = an.prefetch_file_to_gpu_chunks("/tmp/test_model.bin", gpu_id=0, chunk_bytes=32*1024*1024)
# wait for first event
an.wait_event_handle_bytes(chunks[0]["event_handle"])
# release when done
an.release_prefetch_chunks([chunks[0]["slab_id"]])
```

## Best practices & production notes

* **Slab allocation** avoids repeated `cudaMalloc` costs. Choose slab size to cover model footprint + some headroom.
* **Pinned pool reuse** is crucial — avoid `cudaHostAlloc`/`cudaFreeHost` on hot path.
* **Event lifecycle**: IPC event handles depend on the original `cudaEvent_t` remaining alive. Ensure you do not destroy events before consumers call `cudaIpcOpenEventHandle`.
* **Cross-process memory**: The prototype uses event IPC for signaling. Sharing raw device pointers across processes is non-trivial and requires `cudaIpcGetMemHandle/cudaIpcOpenMemHandle`.
* **Containerization**: If running in containers and using IPC between processes/containers, ensure host IPC and driver compatibility. `--ipc=host` and NVIDIA Container Toolkit may be required.
* **Error handling & cleanup**: Always call release APIs to free device memory to avoid leaks.

## Debugging & profiling

* Use `nvidia-smi`, `nsys` and `nvprof` (or `nv-nsight`) to profile H2D copies and kernel activity.
* Use `cuda-memcheck` and validate that event handles are opened in the same driver context.

## License

Add your license here.