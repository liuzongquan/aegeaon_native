cmake_minimum_required(VERSION 3.18)
project(aegaeon_native LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

pybind11_add_module(aegaeon_native
    src/python_bindings.cpp
    src/allocator.cpp
    src/pinned_pool.cpp
    src/ipc_event.cpp
    src/kvcache.cpp
)

target_include_directories(aegaeon_native PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(aegaeon_native PRIVATE CUDA::cudart)
