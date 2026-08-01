#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

// Apple RDMA-over-Thunderbolt (UC) transport, negotiated by socket_t over the
// bootstrap TCP connection. See transport-apple.cpp for the provider constraints.
struct apple_rdma {
    // Opens the local device whose GID is target_gid, creates the queue pair and
    // writes the local endpoint into caps (RPC_CONN_CAPS_SIZE bytes). Null if the
    // device is unusable; fd is kept as the liveness anchor.
    static std::unique_ptr<apple_rdma> probe(int fd, const uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Connect to the peer endpoint in caps, which must come from a peer that
    // advertised RDMA: both sides run a readiness handshake over the bootstrap
    // socket before this returns.
    bool activate(const uint8_t * caps);

    bool send(const void * data, size_t size);
    bool recv(void * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() const;

private:
    struct impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
