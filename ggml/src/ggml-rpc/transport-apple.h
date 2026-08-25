#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    static std::unique_ptr<apple_rdma> probe(int fd, const uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(const uint8_t * caps);

    bool send(const void * data, size_t size);
    bool recv(void * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();

    // Dedicated gate channel on a second queue pair. Every message is one exact-size
    // payload with no header and no padding, so it fits the single scatter entry the
    // provider grants and lands straight in the caller's registered memory. The
    // endpoint blob is RPC_CONN_CAPS_SIZE bytes and travels over the byte stream.
    bool gate_create  (uint8_t * local_ep);
    bool gate_activate(const uint8_t * remote_ep);
    bool gate_ready() const;
    bool gate_register (void * addr, size_t size);
    bool gate_post_recv(void * dst, size_t len, uint64_t tag);
    bool gate_send     (const void * src, size_t len);
    bool gate_wait_recv(uint64_t tag, int64_t timeout_us, int64_t (*now_us)(void));

    // Ask for a small frame size on this link. A whole frame goes on the wire however
    // little of it is filled, so a link carrying small messages wants this and a link
    // carrying bulk does not. Must be called before probe hands out caps.
    void prefer_small_frames();

    // True once the connection has failed; the caller should drop the socket.
    bool broken() const;

private:
    struct impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
