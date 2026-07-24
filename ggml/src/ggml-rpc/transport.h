#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    void flush();
    bool is_rdma() const;
    // Pin the local RDMA device for THIS connection (overrides auto-selection).
    // A client with several RDMA links uses this to face the right peer per worker.
    // Must be set before the caps handshake. No-op without RDMA.
    void set_rdma_device(const char * name);

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();

// Pin the local RDMA device by name (e.g. "rdma_en6"), overriding auto-detection
// and the GGML_RDMA_DEV env var. A host with several RDMA links must select the
// one facing the peer. NULL/empty clears the override. No-op without RDMA.
void rpc_transport_set_rdma_device(const char * name);
