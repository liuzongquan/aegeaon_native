#pragma once
#include <string>
#include <vector>

namespace aegaeon {
    // Serialize an Event IPC handle into bytes
    std::vector<char> get_event_ipc_handle_bytes(void* raw_event);
    // Open event from handle bytes and block until it's signaled
    void wait_event_ipc_handle_bytes(const std::vector<char>& handle_bytes);
    // Utility: create & return event handle bytes for a cudaEvent (the C++ function will create a cudaEvent,
    // but generally users create events with create_and_get_handle and then share the bytes)
    std::vector<char> create_event_and_get_handle();
}
