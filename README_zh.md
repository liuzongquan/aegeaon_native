
# Aegaeon 原生模块 (aegaeon_native) — 说明（中文）

## 概述
`aegaeon_native` 是 Aegaeon 的本地 C++/CUDA 扩展，提供高性能的底层能力：
- GPU slab 分配器（一次性分配大块显存并切分子块）
- Pinned host buffer 池（复用 page-locked 内存）
- 异步分块文件预取到 GPU（cudaMemcpyAsync）
- CUDA 事件的 IPC 导出/导入（cudaIpcGetEventHandle / cudaIpcOpenEventHandle）
- KV 缓存辅助（异步 H2D/D2H 操作）
- 通过 pybind11 暴露给 Python

每个 GPU 节点应编译并安装该模块，供 Python 的控制平面（`aegaeon_proto`）调用。

## 目录结构
````

aegaeon_native/
├── CMakeLists.txt
├── setup.py / pyproject.toml
├── include/aegaeon/*.h
└── src/*.cpp

````

## 系统要求
- Debian 12 或类似 Linux
- Python 3.10
- CUDA toolkit & 驱动（CUDA 11.x / 12.x，可根据硬件选择）
- CMake ≥ 3.18
- 支持 C++17 的编译器（g++ >= 9）
- `pybind11`、`setuptools`、`wheel`

## 构建（依赖）
在虚拟环境中安装构建依赖：
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install pybind11 cmake wheel setuptools
````

### 使用 `setup.py` 构建 wheel（推荐）

```bash
python setup.py bdist_wheel
pip install dist/*.whl
```

### 使用 CMake 直接构建（开发）

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
# 将生成的 .so 放到 Python path 中
cp aegaeon_native*.so ../
```

## 主要 Python API（绑定函数）

* `SlabAllocator(gpu_id, total_bytes)`
* `PinnedPool(buffer_size, pool_size)`
* `KVCache(gpu_id)`
* `create_event_handle_bytes()` — 创建事件并返回 IPC handle bytes
* `wait_event_handle_bytes(bytes)` — 打开 IPC handle 并等待
* `prefetch_file_to_gpu_chunks(filepath, gpu_id, chunk_bytes)` — 分块上传并返回每块 `{device_ptr, size, event_handle, slab_id}`
* `release_prefetch_chunks([slab_ids])` — 按 slab id 释放内存/事件
* `release_all_prefetch_events()` — 释放所有预取资源

**注意**：返回的 `device_ptr` 是原始 device pointer（uint64）。跨进程共享 device memory 需要额外使用 `cudaIpcGetMemHandle/cudaIpcOpenMemHandle`，事件 IPC（当前实现）用于跨进程同步的信号。

## 使用示例（Python）

```python
import aegaeon_native as an
chunks = an.prefetch_file_to_gpu_chunks("/tmp/test_model.bin", gpu_id=0, chunk_bytes=32*1024*1024)
an.wait_event_handle_bytes(chunks[0]["event_handle"])
an.release_prefetch_chunks([chunks[0]["slab_id"]])
```

## 生产建议与注意事项

* **Slab sizing**：一次分配足够大的 slab 以避免碎片，包含模型峰值内存与缓冲区。
* **Pinned pool 重用**：避免频繁调用 `cudaHostAlloc`/`cudaFreeHost`，使用 pinned pool 来复用。
* **事件生命周期**：IPC handle 依赖原始 `cudaEvent_t` 对象存活，确保在消费者打开并等待之前不要销毁事件。
* **容器化与 IPC**：若在容器间使用 IPC，需确保驱动与容器配置支持（可能需要 `--ipc=host` 等设置）。
* **错误处理**：在高并发场景下保证释放 API 被调用以免显存泄露。

## 调试与性能分析

* 使用 `nvidia-smi`、`nsys`、`nvprof`/`nv-nsight` 来分析 H2D 传输、CUDA kernel、事件 timeline。
* 使用 `cuda-memcheck` 检查显存错误。

## 许可证

请在此处添加许可证。

