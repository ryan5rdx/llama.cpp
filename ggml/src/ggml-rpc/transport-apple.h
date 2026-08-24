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

    // Zero-copy send. Register the region once, then send_from takes the payload out of
    // it directly. Both return false when the provider cannot do it, and the caller
    // falls back to send().
    // Ask for a small frame size on this link. A whole frame goes on the wire however
    // little of it is filled, so a link carrying small messages wants this and a link
    // carrying bulk does not. Must be called before probe hands out caps.
    void prefer_small_frames();

    bool zc_register(void * addr, size_t size);
    void zc_release();
    bool send_from(const void * base, size_t off, size_t size);
    // True once the connection has failed; the caller should drop the socket.
    bool broken() const;

private:
    struct impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
