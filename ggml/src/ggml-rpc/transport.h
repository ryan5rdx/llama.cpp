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
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    // Gate channel: a side channel for exact-size messages that land straight in
    // registered caller memory, with no header and no padding. All of these return
    // false where the transport has no such channel, and the caller falls back to
    // send_data/recv_data.
    bool gate_create  (uint8_t * local_ep);   // fills RPC_CONN_CAPS_SIZE bytes
    bool gate_activate(const uint8_t * remote_ep);
    bool gate_ready() const;
    bool gate_register (void * addr, size_t size);
    bool gate_post_recv(void * dst, size_t len, uint64_t tag);
    bool gate_send     (const void * src, size_t len);
    bool gate_wait_recv(uint64_t tag, int64_t timeout_us, int64_t (*now_us)(void));

    // Ask for a small frame size: a whole frame goes on the wire however little of it
    // is filled, so a link carrying small messages wants this and a bulk link does not.
    void prefer_small_frames();

    // unblock a peer sitting in recv_data on another thread
    void shutdown_rw();


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
