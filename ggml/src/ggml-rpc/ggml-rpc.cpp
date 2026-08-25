#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"

#include <array>
#include <atomic>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cinttypes>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

// GGML_RPC_PROFILE=N reports the allreduce gate cost every N gates (default 256).
// The gate runs once per reduction boundary, so it is the hot path under -sm tensor.
static const char * RPC_PROFILE = std::getenv("GGML_RPC_PROFILE");

// Small f32 gates are decode, large bf16 ones are prefill. They differ by orders of
// magnitude, so a combined average says nothing - keep a bucket per kind.
struct rpc_gate_bucket {
    int64_t wait_us   = 0; // gpu sync: waiting for the partial to land
    int64_t pack_us   = 0; // read the partial out of device memory
    int64_t exch_us   = 0; // send to the peer and receive its partial
    int64_t unpack_us = 0; // write the peer partial into device memory
    int64_t submit_us = 0; // encode and enqueue the reduce
    int64_t bytes     = 0;
    int64_t n_gates   = 0;
};

static rpc_gate_bucket g_gate_prof[2]; // [0] = f32 decode, [1] = bf16 prefill
static int64_t         g_gate_prof_period = 0;
static int64_t         g_gate_prof_total  = 0;

// GGML_RPC_METAL_FAST_SYNC=1 parks the allreduce gate on a backend fence: the GPU
// publishes arrival into a word the host spins on, and waits on a word the host stores
// once the peer data has landed. That removes a blocking command buffer wait and lets
// the reduce be queued while the exchange is still in flight. Off by default; the
// backend may not provide a fence, in which case the gate keeps its original shape.
static const char * RPC_FAST_SYNC = std::getenv("GGML_RPC_METAL_FAST_SYNC");

enum { RPC_FENCE_ARRIVAL = 0, RPC_FENCE_RELEASE = 1, RPC_FENCE_TIMEOUT = 2 };

// Below this both ranks send before receiving, which halves gate latency on a full duplex
// link. Above it they take turns, so a payload cannot outrun the peer's pre-posted ring.
static constexpr size_t RPC_GATE_DUPLEX_MAX = 1024 * 1024;

struct rpc_fence_api {
    void * (*init)   (ggml_backend_t)            = nullptr;
    void   (*destroy)(void *)                    = nullptr;
    volatile uint32_t * (*words)(void *)         = nullptr;
    bool   (*publish)(void *, uint32_t, const ggml_tensor *)            = nullptr;
    bool   (*arm)    (void *, uint32_t, uint32_t, const ggml_tensor *) = nullptr;
    void   (*sync)   (void *)                    = nullptr;
    size_t (*words_size)(void)                   = nullptr;
    bool   (*buffer_direct)(ggml_backend_buffer_t) = nullptr;
};

static bool rpc_fast_sync_requested() {
    return RPC_FAST_SYNC != nullptr && atoi(RPC_FAST_SYNC) != 0;
}

static bool rpc_fence_get(ggml_backend_t backend, rpc_fence_api & api) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    if (reg == nullptr) {
        return false;
    }
    api.init    = (void * (*)(ggml_backend_t))            ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_init");
    api.destroy = (void (*)(void *))                      ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_free");
    api.words   = (volatile uint32_t * (*)(void *))       ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_words");
    api.publish = (bool (*)(void *, uint32_t, const ggml_tensor *))           ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_publish");
    api.arm     = (bool (*)(void *, uint32_t, uint32_t, const ggml_tensor *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_arm");
    api.sync    = (void (*)(void *))                      ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_sync");
    api.words_size = (size_t (*)(void))                   ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_words_size");
    api.buffer_direct = (bool (*)(ggml_backend_buffer_t))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_fence_buffer_direct");
    return api.init && api.destroy && api.words && api.publish && api.arm && api.sync &&
           api.buffer_direct && api.words_size;
}

// The fence words are plain shared memory the GPU also touches, so pair every access
// with a fence rather than relying on the volatile qualifier.
static inline uint32_t rpc_fence_load(volatile uint32_t * w) {
    const uint32_t v = *w;
    std::atomic_thread_fence(std::memory_order_acquire);
    return v;
}

static inline void rpc_fence_store(volatile uint32_t * w, uint32_t v) {
    std::atomic_thread_fence(std::memory_order_release);
    *w = v;
}

// One queued gate. The ggml context that produced the tensors is gone by the time the
// service thread runs, so this carries raw host addresses only.
// The peer may run ahead, so the region its NIC writes into rotates: gate i lands in
// slot i mod RPC_GATE_SLOTS. Without that, gate i+1 could overwrite the partial the GPU
// is still reducing for gate i - the fence orders our own GPU, not the peer's NIC.
static constexpr uint32_t RPC_GATE_SLOTS = 4;

// GGML_RPC_NO_GATE_CHANNEL=1 keeps gates on the byte stream; GGML_RPC_NO_DOORBELL=1 keeps
// the channel but has the host write the release word. Both exist so a regression can be
// attributed to one change rather than the pair.
static bool rpc_gate_channel_requested() {
    const char * env = std::getenv("GGML_RPC_NO_GATE_CHANNEL");

    return !(env && atoi(env) != 0);
}

static bool rpc_doorbell_requested() {
    const char * env = std::getenv("GGML_RPC_NO_DOORBELL");

    return !(env && atoi(env) != 0);
}

struct rpc_gate {
    uint32_t seq;
    const void * send_src;
    void *       recv_dst;
    size_t       wire_bytes;
    bool         wire_bf16;

    // what the gate channel needs to register memory and pre-post the next receives
    void *       recv_base    = nullptr;   // slot 0
    size_t       recv_stride  = 0;
    void *       scratch_base = nullptr;
    size_t       scratch_size = 0;
    const void * send_base    = nullptr;
    size_t       send_size    = 0;
    void *       doorbell     = nullptr;   // this gate's staging word
};

// Bounded so a dead peer cannot leave the command processor spinning until the GPU
// watchdog kills it. Measured at roughly 1.7M iterations per second on an M1 Max, so
// the default is a few seconds.
static const uint32_t RPC_FENCE_MAX_ITERS = [] {
    const char * env = std::getenv("GGML_RPC_FENCE_MAX_ITERS");
    const long   val = env ? atol(env) : 0;

    return val > 0 ? (uint32_t) val : 8000000u;
}();

static void rpc_gate_prof_report() {
    static const char * kind_name[2] = { "decode/f32", "prefill/bf16" };
    for (int k = 0; k < 2; k++) {
        const rpc_gate_bucket & b = g_gate_prof[k];
        if (b.n_gates == 0) {
            continue;
        }
        const double n = (double) b.n_gates;
        GGML_LOG_INFO("rpc: allreduce %-12s %6" PRId64 " gates: wait %7.1f pack %6.1f exch %7.1f unpack %6.1f submit %6.1f us, %7.1f KiB\n",
                      kind_name[k], b.n_gates, b.wait_us/n, b.pack_us/n, b.exch_us/n,
                      b.unpack_us/n, b.submit_us/n, b.bytes/n/1024.0);
    }
}


namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
const size_t GRAPH_CACHE_MAX = 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t  major;
    uint8_t  minor;
    uint8_t  patch;
    uint8_t  padding;
    // port this server uses for direct server-to-server communication (collectives)
    uint16_t comm_port;
    uint8_t  conn_caps[RPC_CONN_CAPS_SIZE];
    // address a peer server should dial to reach comm_port, when that differs from the
    // address the client used. Empty means "same host the client connected to".
    char     comm_host[64];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

struct rpc_msg_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

struct rpc_msg_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank > 0: rank 0's comm port (rank 0 listens on its own configured comm port)
    char     host[64];  // rank > 0: rank 0's host
};

struct rpc_msg_comm_init_rsp {
    uint8_t ok;
};

struct rpc_msg_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

struct rpc_msg_comm_free_req {
    uint32_t device;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    // uids of graphs cached by the server for this device
    std::unordered_set<uint64_t> graph_uids;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    if (!sock->send_data(msg, msg_size)) {
        return false;
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, uint16_t * comm_port = nullptr,
                            std::string * comm_host = nullptr) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    if (comm_port != nullptr) {
        *comm_port = response.comm_port;
    }
    if (comm_host != nullptr) {
        comm_host->assign(response.comm_host, strnlen(response.comm_host, sizeof(response.comm_host)));
        LOG_DBG("[%s] server advertises comm endpoint %s:%u\n", __func__,
                comm_host->empty() ? "<client-facing host>" : comm_host->c_str(), response.comm_port);
    }
    sock->update_caps(response.conn_caps);
    return true;
}

// peer-facing address and port advertised by each server in its HELLO response, used to set up
// server-to-server collectives
struct rpc_server_comm_addr {
    std::string host;   // empty: the server did not override it, use the client's own endpoint
    uint16_t    port = 0;
};

static std::mutex server_comm_addr_mutex;
static std::unordered_map<std::string, rpc_server_comm_addr> server_comm_addrs;

static rpc_server_comm_addr rpc_server_comm_address(const std::string & endpoint) {
    std::lock_guard<std::mutex> lock(server_comm_addr_mutex);
    auto it = server_comm_addrs.find(endpoint);
    return it != server_comm_addrs.end() ? it->second : rpc_server_comm_addr{};
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    rpc_server_comm_addr comm_addr;
    if (!negotiate_hello(sock, &comm_addr.port, &comm_addr.host)) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> comm_addr_lock(server_comm_addr_mutex);
        server_comm_addrs[endpoint] = comm_addr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
    return sock;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_memset_tensor_req request = {
        /* .tensor = */ serialize_tensor(tensor),
        /* .offset = */ offset,
        /* .size   = */ size,
        /* .value  = */ value,
    };
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_MEMSET_TENSOR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t input_size = sizeof(rpc_tensor) + 4*sizeof(uint64_t) + size*n_copies;
    std::vector<uint8_t> input(input_size, 0);
    uint8_t * dest = input.data();
    memcpy(dest, &rpc_tensor, sizeof(rpc_tensor));
    dest += sizeof(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, sizeof(header));
    dest += sizeof(header);
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (const char *)data + i*stride_data, size);
    }
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_2D, input.data(), input.size());
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_2d_req request;
    request.tensor   = serialize_tensor(tensor);
    request.offset   = offset;
    request.size     = size;
    request.n_copies = n_copies;
    request.stride   = stride_tensor;
    if (stride_data == size) {
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR_2D, &request, sizeof(request), data, size*n_copies);
        RPC_STATUS_ASSERT(status);
    } else {
        std::vector<uint8_t> packed(size*n_copies);
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR_2D, &request, sizeof(request), packed.data(), packed.size());
        RPC_STATUS_ASSERT(status);
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((char *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        // TODO: cache the alloc responses to avoid extra RPC calls?
        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &cgraph->uid, sizeof(cgraph->uid));
    dest += sizeof(cgraph->uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & graph_uids = rpc_dev_ctx->graph_uids;
    bool reuse = cgraph->uid != 0 && graph_uids.count(cgraph->uid) > 0;
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        request.uid    = cgraph->uid;
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    } else {
        if (cgraph->uid != 0) {
            if (graph_uids.size() >= GRAPH_CACHE_MAX) {
                graph_uids.clear();
            }
            graph_uids.insert(cgraph->uid);
        }
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir,
               std::string bind_host, std::string comm_host, uint16_t comm_port)
        : backends(std::move(all_backends)), cache_dir(cache_dir),
          bind_host(std::move(bind_host)), comm_host(std::move(comm_host)), comm_port(comm_port) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
        for (auto & cs : comm_states) {
            cs.reset(new comm_state());
        }
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_2d(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
    bool comm_free(const rpc_msg_comm_free_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    void sync_all_backends();

    bool comm_failed = false;
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    struct comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        rpc_fence_api           fence_api;
        bool                    fence_api_ok = false;
        void *                  fence = nullptr;
        uint32_t                fence_seq = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;

        // Batched gates. The server thread encodes and queues; the service thread waits
        // for each gate's arrival, exchanges, and releases it. Without this the server
        // would have to submit and wait per gate, which is what batching removes.
        std::thread             service;
        std::mutex              mtx;
        std::condition_variable cv;
        std::deque<rpc_gate>    queue;

        // gate channel state, touched only by the service thread
        bool                    gate_armed    = false;
        bool                    gate_doorbell = false;
        bool                    gate_tried    = false;
        size_t                  gate_size     = 0;
        std::atomic<bool>       service_stop{false};
        bool                    service_failed = false;
        uint32_t                gates_done     = 0;
        uint32_t                gates_queued   = 0;

        // A client can vanish at any point, so teardown has to be in the destructor and
        // has to be able to interrupt a blocking exchange.
        ~comm_state() {
            if (!service.joinable()) {
                return;
            }
            service_stop.store(true);
            cv.notify_all();
            if (peer) {
                peer->shutdown_rw();
            }
            service.join();
        }
    };

    void comm_service(comm_state & state);
    bool comm_drain  (comm_state & state);
    void gate_arm    (comm_state & state, const rpc_gate & g);

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    // where this server accepts the peer connection for collectives. bind_host is the
    // client-facing address, used when comm_host is empty; comm_host is the explicit
    // peer-facing address from --comm-host, which is also what gets advertised in HELLO
    std::string bind_host;
    std::string comm_host;
    uint16_t    comm_port;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<std::unique_ptr<comm_state>> comm_states;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major     = RPC_PROTO_MAJOR_VERSION;
    response.minor     = RPC_PROTO_MINOR_VERSION;
    response.patch     = RPC_PROTO_PATCH_VERSION;
    response.comm_port = comm_port;
    // Advertise the peer-facing address only when one was configured explicitly. Left empty,
    // the client falls back to the address it used itself, which is right when both links
    // run over the same network and wrong when they do not.
    if (!comm_host.empty() && comm_host.size() < sizeof(response.comm_host)) {
        memcpy(response.comm_host, comm_host.c_str(), comm_host.size());
    }
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        return false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    if (request.tensor.data < buffer_start) {
        GGML_LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        return false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    if (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        return false;
    }
    if (tensor->buffer->iface.memset_tensor == nullptr) {
        GGML_LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        return false;
    }

    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", value: %u\n",
            __func__, (void *) tensor->buffer, tensor->data, request.offset, request.size, request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        if (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            GGML_LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                           "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                           __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                           result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                           tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            return nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    if (input.size() < sizeof(rpc_tensor) + 4*sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t header[4];
    memcpy(header, input.data() + sizeof(rpc_tensor), sizeof(header));
    const uint64_t offset   = header[0];
    const uint64_t size     = header[1];
    const uint64_t n_copies = header[2];
    const uint64_t stride   = header[3];

    const uint64_t data_size = input.size() - sizeof(rpc_tensor) - 4*sizeof(uint64_t);
    if (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, offset, size, n_copies, stride);

    // sanitize tensor->data
    {
        if (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            return false;
        }
        const uint64_t span = (n_copies - 1)*stride + size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                           __func__, in_tensor->data, offset, span, p0, p1);
            return false;
        }
        if (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            GGML_LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                           __func__, offset, span, ggml_nbytes(tensor));
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + 4*sizeof(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size, request.n_copies, request.stride);

    // sanitize tensor->data
    {
        if (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            return false;
        }
        if (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            return false;
        }
        const uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                           __func__, request.tensor.data, request.offset, span, p0, p1);
            return false;
        }
        if (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            GGML_LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                           __func__, request.offset, span, ggml_nbytes(tensor));
            return false;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint64_t uid;
    memcpy(&uid, src, sizeof(uid));
    src += sizeof(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    if (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
        if (graph->nodes[i] != nullptr) {
            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
        }
    }
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        GGML_LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        return false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
void rpc_server::sync_all_backends() {
    // The backend submits whatever it deferred and waits. Gates are released by the
    // service thread while that wait runs, so the drain is only for the failure status.
    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
    for (auto & cs : comm_states) {
        if (cs->fence && !comm_drain(*cs)) {
            comm_failed = true;
        }
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    if (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        return true;
    }
    comm_state & state = *comm_states[request.device];
    if (state.peer != nullptr) {
        response.ok = 1;
        return true;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    if (request.rank == 0) {
        // listen on the peer-facing address if one was configured, otherwise on the same
        // host the client-facing listener was bound to
        const std::string & listen_host = comm_host.empty() ? bind_host : comm_host;
        socket_ptr srv = socket_t::create_server(listen_host.c_str(), comm_port);
        if (srv == nullptr) {
            GGML_LOG_ERROR("[%s] failed to listen for the peer on %s:%u\n", __func__,
                           listen_host.c_str(), comm_port);
            return true;
        }
        state.peer = srv->accept();
        if (state.peer == nullptr) {
            return true;
        }
        if (!state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
            state.peer = nullptr;
            return true;
        }
        // this link carries gate partials and small control messages, never bulk
        state.peer->prefer_small_frames();
        state.peer->get_caps(local_caps);
        if (!state.peer->send_data(local_caps, sizeof(local_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->update_caps(remote_caps);
    } else {
        const std::string host(request.host, strnlen(request.host, sizeof(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        for (int i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            if (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        if (state.peer == nullptr) {
            GGML_LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            return true;
        }
        // this link carries gate partials and small control messages, never bulk
        state.peer->prefer_small_frames();
        state.peer->get_caps(local_caps);
        if (!state.peer->send_data(local_caps, sizeof(local_caps)) ||
            !state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->update_caps(remote_caps);
    }
    // Gate channel: a second queue pair carrying only exact-size gate payloads. Both
    // ranks reach this at the same point, so the endpoint swap is symmetric.
    {
        uint8_t local_ep[RPC_CONN_CAPS_SIZE]  = {};
        uint8_t remote_ep[RPC_CONN_CAPS_SIZE] = {};
        if (state.peer->gate_create(local_ep) &&
            state.peer->send_data(local_ep, sizeof(local_ep)) &&
            state.peer->flush() &&
            state.peer->recv_data(remote_ep, sizeof(remote_ep))) {
            state.peer->gate_activate(remote_ep);
        }
    }

    state.rank  = request.rank;
    state.world = request.world;
    state.fence_api_ok = rpc_fence_get(backends[request.device], state.fence_api);
    if (state.fence_api_ok && rpc_fast_sync_requested()) {
        state.fence = state.fence_api.init(backends[request.device]);
    }
    if (state.fence) {
        state.service = std::thread([this, &state] { comm_service(state); });
    }
    GGML_LOG_INFO("[%s] device %u joined pairwise comm as rank %u, fast sync %s\n",
                  __func__, request.device, request.rank, state.fence ? "on" : "off");
    response.ok = 1;
    return true;
}

// Bring the gate channel up for this payload size. Both ranks call this from the same
// gate, having just exchanged it over the byte stream, so the one-byte handshake at the
// end is a real barrier: seeing the peer's token proves it has posted its receives, and
// a send before that would be dropped.
void rpc_server::gate_arm(comm_state & state, const rpc_gate & g) {
    // Arm once, for one size. Posted receives cannot be recalled, so re-arming at a
    // different size would leave the old ones to be consumed by a send that does not
    // match them. Gates of any other size keep using the byte stream, which is a
    // different queue pair and so cannot touch these.
    if (state.gate_tried || !rpc_gate_channel_requested() || !state.peer->gate_ready()) {
        return;
    }
    state.gate_tried = true;

    bool ok = state.peer->gate_register(g.scratch_base, g.scratch_size) &&
              state.peer->gate_register(const_cast<void *>(g.send_base), g.send_size);

    // The doorbell lands four bytes on our release word, so the peer's NIC lets the GPU
    // through and the host leaves the release path. Needs the fence words registered,
    // which is a Metal allocation rather than the ggml one, so treat it as optional.
    volatile uint32_t * fw = state.fence_api.words(state.fence);
    const bool doorbell = ok && rpc_doorbell_requested() &&
                          state.peer->gate_register((void *) fw, state.fence_api.words_size());

    // post in the order the coming gates consume them: gate i lands in slot i mod N, and
    // its doorbell follows it, matching the order the peer sends them
    for (uint32_t j = 1; ok && j <= RPC_GATE_SLOTS; j++) {
        const uint32_t slot = (g.seq + j) % RPC_GATE_SLOTS;
        ok = state.peer->gate_post_recv((uint8_t *) g.recv_base + (size_t) slot * g.recv_stride,
                                        g.wire_bytes, 0);
        if (ok && doorbell) {
            ok = state.peer->gate_post_recv((void *) &fw[RPC_FENCE_RELEASE], sizeof(uint32_t), 0);
        }
    }

    // Both ranks reach this from the same gate, so the swap is a barrier: seeing the
    // peer's token proves it has posted, and a send before that would be dropped. The
    // token carries whether that rank managed to arm, so the two never disagree.
    uint8_t mine   = (ok ? 1 : 0) | ((ok && doorbell) ? 2 : 0);
    uint8_t theirs = 0;
    if (!state.peer->send_data(&mine, 1) || !state.peer->flush() ||
        !state.peer->recv_data(&theirs, 1)) {
        return;
    }

    state.gate_armed    = (mine & theirs & 1) != 0;
    state.gate_doorbell = (mine & theirs & 2) != 0;
    state.gate_size     = g.wire_bytes;
    if (state.gate_armed) {
        GGML_LOG_INFO("[%s] gate channel armed for %zu byte payloads, doorbell %s\n",
                      __func__, g.wire_bytes, state.gate_doorbell ? "on" : "off");
    } else {
        GGML_LOG_INFO("[%s] gate channel unavailable, using the byte stream\n", __func__);
    }
}

// Drains queued gates: wait for the GPU to reach each one, exchange, release it. Runs on
// its own thread so the server can keep encoding the rest of the token.
void rpc_server::comm_service(comm_state & state) {
#ifdef __APPLE__
    // this thread releases the GPU at every gate; an efficiency core adds latency there
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    volatile uint32_t * fw = state.fence_api.words(state.fence);

    while (true) {
        rpc_gate g;
        {
            std::unique_lock<std::mutex> lock(state.mtx);
            state.cv.wait(lock, [&] { return state.service_stop || !state.queue.empty(); });
            if (state.queue.empty()) {
                return; // stopping
            }
            g = state.queue.front();
            state.queue.pop_front();
        }

        bool ok;
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            ok = !state.service_failed;
        }

        const bool prof = RPC_PROFILE != nullptr;
        const int64_t t_start = prof ? ggml_time_us() : 0;

        if (ok) {
            // the publish runs after the partial, so observing it means the data is there
            const int64_t deadline = ggml_time_us() + 30*1000*1000;
            while (rpc_fence_load(&fw[RPC_FENCE_ARRIVAL]) != g.seq) {
                if (ggml_time_us() > deadline || state.service_stop) {
                    GGML_LOG_ERROR("[%s] gate %u never arrived\n", __func__, g.seq);
                    ok = false;
                    break;
                }
                std::this_thread::yield();
            }
        }

        const int64_t t_arrived = prof ? ggml_time_us() : 0;

        if (ok && rpc_fence_load(&fw[RPC_FENCE_TIMEOUT]) != 0) {
            GGML_LOG_ERROR("[%s] gate %u: a previous fence expired\n", __func__, g.seq);
            ok = false;
        }

        if (ok) {
            if (state.gate_armed && state.gate_size == g.wire_bytes) {
                // exact-size, no header, straight into the slot the reduce will read
                ok = state.peer->gate_send(g.send_src, g.wire_bytes);

                if (ok && state.gate_doorbell) {
                    // The peer's NIC writes our release word, so nothing here waits for
                    // the incoming payload: it lands, then the doorbell behind it lets
                    // that peer's GPU through. Sends on one queue pair keep their order,
                    // so the payload is always in place first.
                    *(uint32_t *) g.doorbell = g.seq;
                    ok = state.peer->gate_send(g.doorbell, sizeof(uint32_t));
                } else if (ok) {
                    ok = state.peer->gate_wait_recv(0, 30*1000*1000, ggml_time_us);
                }
                if (ok) {
                    // this gate's slot is free again and gate seq+RPC_GATE_SLOTS will
                    // land in it, so repost it now to keep the window open
                    const uint32_t slot = g.seq % RPC_GATE_SLOTS;
                    ok = state.peer->gate_post_recv(
                            (uint8_t *) g.recv_base + (size_t) slot * g.recv_stride,
                            g.wire_bytes, 0);
                    if (ok && state.gate_doorbell) {
                        ok = state.peer->gate_post_recv((void *) &fw[RPC_FENCE_RELEASE],
                                                        sizeof(uint32_t), 0);
                    }
                }
                if (!ok) {
                    GGML_LOG_ERROR("[%s] gate channel failed, falling back\n", __func__);
                    state.gate_armed = false;
                }
            } else if (g.wire_bytes <= RPC_GATE_DUPLEX_MAX) {
                // both directions in flight at once: the link is full duplex and the peer
                // has a deep pre-posted ring, so a payload this size cannot stall
                ok = state.peer->send_data(g.send_src, g.wire_bytes) &&
                     state.peer->flush() &&
                     state.peer->recv_data(g.recv_dst, g.wire_bytes);
                if (ok) {
                    gate_arm(state, g);
                }
            } else if (state.rank == 0) {
                // a big payload can outrun the ring, so the ranks take turns
                ok = state.peer->send_data(g.send_src, g.wire_bytes) &&
                     state.peer->flush() &&
                     state.peer->recv_data(g.recv_dst, g.wire_bytes);
            } else {
                ok = state.peer->recv_data(g.recv_dst, g.wire_bytes) &&
                     state.peer->send_data(g.send_src, g.wire_bytes) &&
                     state.peer->flush();
            }
            if (!ok) {
                GGML_LOG_ERROR("[%s] gate %u exchange failed\n", __func__, g.seq);
            }
        }

        const int64_t t_exchanged = prof ? ggml_time_us() : 0;

        // Release even on failure: the reduce is already queued behind this fence and
        // every later gate is queued behind that, so not releasing wedges the whole
        // batch. When the doorbell is up the peer's NIC does this for us, and writing it
        // here as well would let the reduce run before the payload landed.
        if (!state.gate_doorbell || !ok || g.wire_bytes != state.gate_size) {
            rpc_fence_store(&fw[RPC_FENCE_RELEASE], g.seq);
        }

        if (prof) {
            // pack, unpack and submit are structurally zero here: the payload moves
            // straight to and from device memory and the reduce was encoded already
            if (g_gate_prof_period == 0) {
                g_gate_prof_period = atol(RPC_PROFILE);
                if (g_gate_prof_period <= 0) {
                    g_gate_prof_period = 256;
                }
            }
            rpc_gate_bucket & b = g_gate_prof[g.wire_bf16 ? 1 : 0];
            b.wait_us += t_arrived   - t_start;
            b.exch_us += t_exchanged - t_arrived;
            b.bytes   += g.wire_bytes;
            b.n_gates++;
            if (++g_gate_prof_total % g_gate_prof_period == 0) {
                rpc_gate_prof_report();
            }
        }

        {
            std::lock_guard<std::mutex> lock(state.mtx);
            if (!ok) {
                state.service_failed = true;
            }
            state.gates_done++;
        }
        state.cv.notify_all();
    }
}

// Wait for every queued gate to be serviced. Called at a point where the batch has been
// committed, so the GPU can actually reach the fences.
bool rpc_server::comm_drain(comm_state & state) {
    std::unique_lock<std::mutex> lock(state.mtx);
    // not queue.empty(): the last gate is popped before it is serviced
    state.cv.wait(lock, [&] { return state.gates_done == state.gates_queued; });

    if (state.fence && rpc_fence_load(&state.fence_api.words(state.fence)[RPC_FENCE_TIMEOUT]) != 0) {
        // the last gate of a batch has no successor to notice this
        GGML_LOG_ERROR("[%s] a fence expired, the reduce ran on stale data\n", __func__);
        state.service_failed = true;
    }

    return !state.service_failed;
}

bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
    if (comm_failed) {
        GGML_LOG_ERROR("[%s] a previous gate failed, dropping the connection\n", __func__);
        return false;
    }
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = *comm_states[request.device];
    if (state.peer == nullptr) {
        GGML_LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        return false;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    if (t_dst == nullptr || t_dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);
    if (nbytes == 0) {
        return true;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    const bool   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    const size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    // the fenced path needs a word of its own in the scratch: the fence writes it and
    // the reduce reads the same buffer, which is what orders them
    // bf16: [wire_send][wire_recv x SLOTS][peer]   f32: [peer x SLOTS]
    // only the region the peer's NIC writes rotates; the GPU-written ones are already
    // serialized by the fence
    const size_t nic_bytes   = wire_bf16 ? (size_t) ne*2 : nbytes;
    const size_t nic_base    = wire_bf16 ? (size_t) ne*2 : 0;
    const size_t after_slots = nic_base + RPC_GATE_SLOTS*nic_bytes;
    const size_t guard_offs  = GGML_PAD(wire_bf16 ? after_slots + nbytes : after_slots, 32);
    // one doorbell staging word per slot: the NIC reads it after the post returns, so
    // the next gate must not be writing the same word
    const size_t db_offs     = guard_offs + 32;
    const size_t need        = GGML_PAD(db_offs + RPC_GATE_SLOTS*sizeof(uint32_t), 32);
    const uint32_t slot      = (state.fence_seq + 1) % RPC_GATE_SLOTS;
    if (state.scratch_size < need) {
        // Gates already encoded read this scratch, so it cannot be replaced before the
        // GPU is done with it: the pages are the caller's and Metal does not keep them
        // mapped. sync_all_backends submits the deferred work and drains the gates.
        sync_all_backends();
        ggml_backend_buffer_t grown = ggml_backend_alloc_buffer(backend, need);
        if (grown == nullptr) {
            GGML_LOG_ERROR("[%s] could not grow the comm scratch to %zu bytes\n", __func__, need);
            return false;
        }
        state.scratch.reset(grown);
        state.scratch_size = need;
    }
    char * scratch_base = (char *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        return t;
    };
    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        return t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    const bool prof = RPC_PROFILE != nullptr;
    auto tick = [prof]() -> int64_t { return prof ? ggml_time_us() : 0; };

    // the fenced path writes the peer partial into the scratch while the reduce is
    // already queued, so that write must not be a blit queued behind it
    void * const fence = (state.fence && state.fence_api.buffer_direct(state.scratch.get()))
        ? state.fence : nullptr;

    // A host-visible tensor can go on and off the wire in place, which skips the bounce
    // through send_buf/recv_buf. Worth little on decode payloads but hundreds of
    // microseconds per gate on prefill ones.
    auto direct_addr = [&](const ggml_tensor * t) -> void * {
        if (t == nullptr || !state.fence_api_ok || !state.fence_api.buffer_direct(t->buffer)) {
            return nullptr;
        }
        return t->data;
    };

    // wire staging, built up front so the batched path can encode the cast before the
    // arrival publish: the publish has to order after whatever the host will read
    ggml_tensor * t_wire_send = wire_bf16 ? new_scratch_tensor(GGML_TYPE_BF16, 0) : nullptr;
    ggml_tensor * t_wire_recv = wire_bf16
        ? new_scratch_tensor(GGML_TYPE_BF16, nic_base + slot*nic_bytes) : nullptr;
    ggml_tensor * const t_send = wire_bf16 ? t_wire_send : t_dst;
    ggml_tensor * const t_recv = t_wire_recv;

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type,
        wire_bf16 ? after_slots : slot*nic_bytes);
    ggml_tensor * t_cast = wire_bf16 ? new_cpy_node(t_wire_recv, t_peer) : nullptr;

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    auto submit_reduce = [&]() {
        if (t_cast != nullptr) {
            compute_nodes(t_cast, t_red);
        } else {
            compute_nodes(t_red, nullptr);
        }
    };

    auto arm_fence = [&](uint32_t seq) -> bool {
        ggml_tensor * t_guard = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        t_guard->buffer = state.scratch.get();
        t_guard->data   = scratch_base + guard_offs;
        return state.fence_api.arm(state.fence, seq, RPC_FENCE_MAX_ITERS, t_guard);
    };

    // Batched: encode the whole gate and hand it to the service thread. Nothing here
    // waits, so the server goes straight on to the next chunk of the token.
    if (fence) {
        const uint32_t seq = ++state.fence_seq;

        if (wire_bf16) {
            compute_nodes(new_cpy_node(t_dst, t_wire_send), nullptr);
        }

        void * const send_src = direct_addr(t_send);
        void * const recv_dst = direct_addr(t_recv ? t_recv : t_peer);
        if (send_src == nullptr || recv_dst == nullptr) {
            GGML_LOG_ERROR("[%s] batched gate needs host-visible staging\n", __func__);
            return false;
        }

        if (!state.fence_api.publish(fence, seq, t_send) || !arm_fence(seq)) {
            GGML_LOG_ERROR("[%s] failed to encode gate %u\n", __func__, seq);
            return false;
        }
        submit_reduce();

        rpc_gate g;
        g.seq          = seq;
        g.send_src     = send_src;
        g.recv_dst     = recv_dst;
        g.wire_bytes   = wire_bytes;
        g.wire_bf16    = wire_bf16;
        g.recv_base    = scratch_base + nic_base;
        g.recv_stride  = nic_bytes;
        g.scratch_base = scratch_base;
        g.scratch_size = state.scratch_size;
        g.send_base    = ggml_backend_buffer_get_base(t_send->buffer);
        g.send_size    = ggml_backend_buffer_get_size(t_send->buffer);
        g.doorbell     = scratch_base + db_offs + (size_t) slot*sizeof(uint32_t);
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.queue.push_back(g);
            state.gates_queued++;
        }
        state.cv.notify_all();

        return true;
    }

    volatile uint32_t * const fw  = fence ? state.fence_api.words(fence) : nullptr;
    const uint32_t      seq = fence ? ++state.fence_seq : 0;

    const int64_t t_start = tick();

    // wait for the pending subgraph that produced this partial
    if (fence) {
        // the publish runs after everything already queued, so observing it means the
        // partial has landed - and that the previous gate's reduce has run, which is
        // the first point its timeout word can be trusted
        if (!state.fence_api.publish(fence, seq, t_send)) {
            GGML_LOG_ERROR("[%s] fence publish failed\n", __func__);
            return false;
        }
        const int64_t deadline = ggml_time_us() + 10*1000*1000;
        while (rpc_fence_load(&fw[RPC_FENCE_ARRIVAL]) != seq) {
            if (ggml_time_us() > deadline) {
                GGML_LOG_ERROR("[%s] timed out waiting for gate arrival %u\n", __func__, seq);
                return false;
            }
            std::this_thread::yield();
        }
        if (rpc_fence_load(&fw[RPC_FENCE_TIMEOUT]) != 0) {
            GGML_LOG_ERROR("[%s] previous gate released by timeout, reduce used stale data\n", __func__);
            return false;
        }
    } else {
        ggml_backend_synchronize(backend);
    }

    const int64_t t_synced = tick();

    if (wire_bf16) {
        compute_nodes(new_cpy_node(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
    }

    const void * send_src = direct_addr(t_send);
    if (send_src == nullptr) {
        ggml_backend_tensor_get(t_send, state.send_buf.data(), 0, wire_bytes);
        send_src = state.send_buf.data();
    }

    const int64_t t_packed = tick();

    if (fence) {
        if (!arm_fence(seq)) {
            GGML_LOG_ERROR("[%s] fence arm failed\n", __func__);
            return false;
        }
        submit_reduce();
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    void * recv_dst = direct_addr(t_recv ? t_recv : t_peer);
    const bool recv_direct = recv_dst != nullptr;
    if (!recv_direct) {
        recv_dst = state.recv_buf.data();
    }

    bool exch_ok;
    if (state.rank == 0) {
        exch_ok = state.peer->send_data(send_src, wire_bytes) &&
                  state.peer->flush() &&
                  state.peer->recv_data(recv_dst, wire_bytes);
    } else {
        exch_ok = state.peer->recv_data(recv_dst, wire_bytes) &&
                  state.peer->send_data(send_src, wire_bytes) &&
                  state.peer->flush();
    }
    if (!exch_ok) {
        // release anyway, so the queued reduce drains instead of spinning to the bound
        if (fence) {
            rpc_fence_store(&fw[RPC_FENCE_RELEASE], seq);
        }
        return false;
    }

    const int64_t t_exchanged = tick();

    if (!recv_direct) {
        ggml_backend_tensor_set(wire_bf16 ? t_wire_recv : t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    const int64_t t_unpacked = tick();

    if (fence) {
        rpc_fence_store(&fw[RPC_FENCE_RELEASE], seq);
    } else {
        submit_reduce();
    }

    if (prof) {
        if (g_gate_prof_period == 0) {
            g_gate_prof_period = atol(RPC_PROFILE);
            if (g_gate_prof_period <= 0) {
                g_gate_prof_period = 256;
            }
        }
        rpc_gate_bucket & b = g_gate_prof[wire_bf16 ? 1 : 0];
        b.wait_us   += t_synced      - t_start;
        b.pack_us   += t_packed      - t_synced;
        b.exch_us   += t_exchanged   - t_packed;
        b.unpack_us += t_unpacked    - t_exchanged;
        b.submit_us += ggml_time_us() - t_unpacked;
        b.bytes     += wire_bytes;
        b.n_gates++;
        if (++g_gate_prof_total % g_gate_prof_period == 0) {
            rpc_gate_prof_report();
        }
    }
    return true;
}

bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = *comm_states[request.device];
    if (state.service.joinable()) {
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.service_stop.store(true);
        }
        state.cv.notify_all();
        state.service.join();
    }
    if (state.fence) {
        state.fence_api.destroy(state.fence);
        state.fence = nullptr;
    }
    state.peer.reset();
    state.scratch.reset();
    state.scratch_size = 0;
    state.rank = 0;
    state.world = 0;
    state.fence_seq = 0;
    state.service_stop.store(false);
    state.service_failed = false;
    state.gates_done = 0;
    state.gates_queued = 0;
    state.queue.clear();
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             const std::string & bind_host, const std::string & comm_host,
                             uint16_t comm_port, socket_ptr sock) {
    rpc_server server(backends, cache_dir, bind_host, comm_host, comm_port);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.memset_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_2d(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_2D: {
                rpc_msg_get_tensor_2d_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor_2d(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_INIT: {
                rpc_msg_comm_init_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_comm_init_rsp response;
                if (!server.comm_init(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_ALLREDUCE: {
                rpc_msg_comm_allreduce_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_allreduce(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_FREE: {
                rpc_msg_comm_free_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_free(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices,
                                   const char * comm_host, uint16_t comm_port) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    const std::string comm_host_str = comm_host ? comm_host : "";
    printf("  endpoint       : %s\n", endpoint);
    printf("  comm endpoint  : %s:%u%s\n",
           comm_host_str.empty() ? "<bind host>" : comm_host_str.c_str(), comm_port,
           comm_host_str.empty() ? " (peer dials the address the client used)" : "");
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, host, comm_host_str, comm_port, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
struct ggml_backend_rpc_comm_context {
    struct rank_info {
        std::string endpoint;
        uint32_t    device;
    };
    std::vector<rank_info> ranks;
};

static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return;
    }
    for (const auto & rank : comm_ctx->ranks) {
        rpc_msg_comm_free_req request = {rank.device};
        auto sock = get_socket(rank.endpoint);
        if (sock != nullptr) {
            send_rpc_cmd(sock, RPC_CMD_COMM_FREE, &request, sizeof(request));
        }
    }
    delete comm_ctx;
}

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    // only the pairwise (world size 2) case is implemented; other configurations fall back
    // to the generic allreduce which routes tensor data through the client
    if (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        return nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_rpc(backends[i])) {
            return nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first;
        // to use two devices on the same host, start one rpc-server per device
        for (const auto & rank : ranks) {
            if (rank.endpoint == rpc_ctx->endpoint) {
                GGML_LOG_WARN("%s: multiple ranks on endpoint %s are not supported, start one rpc-server per device\n",
                              __func__, rpc_ctx->endpoint.c_str());
                return nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device});
    }

    // Rank 1 dials rank 0 on the comm endpoint advertised in rank 0's HELLO response. When
    // rank 0 was started with --comm-host it advertises that address, which lets the peer link
    // run over a different interface than the client link; this is required where the two
    // cannot share one, as with Apple RDMA over Thunderbolt, whose interface address a local
    // client cannot connect to. Otherwise fall back to the address the client itself used,
    // which is right when every link runs over the same network.
    const rpc_server_comm_addr comm_addr = rpc_server_comm_address(ranks[0].endpoint);
    if (comm_addr.port == 0) {
        GGML_LOG_WARN("%s: server %s does not advertise a comm port\n", __func__, ranks[0].endpoint.c_str());
        return nullptr;
    }
    std::string host0 = comm_addr.host;
    if (host0.empty()) {
        int port0;
        if (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
            return nullptr;
        }
    }
    const uint16_t comm_port = comm_addr.port;
    if (host0.size() >= 64) {
        return nullptr;
    }
    LOG_DBG("[%s] rank 1 will dial rank 0 at %s:%u\n", __func__, host0.c_str(), comm_port);

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    for (size_t i = 0; i < n_backends; i++) {
        rpc_msg_comm_init_req request = {};
        request.device = ranks[i].device;
        request.rank   = (uint32_t) i;
        request.world  = (uint32_t) n_backends;
        request.port   = comm_port;
        if (i > 0) {
            memcpy(request.host, host0.c_str(), host0.size());
        }
        auto sock = get_socket(ranks[i].endpoint);
        if (sock == nullptr || !send_rpc_cmd(sock, RPC_CMD_COMM_INIT, &request, sizeof(request))) {
            return nullptr;
        }
    }
    bool ok = true;
    for (size_t i = 0; i < n_backends; i++) {
        auto sock = get_socket(ranks[i].endpoint);
        rpc_msg_comm_init_rsp response = {};
        uint64_t rsp_size = 0;
        if (sock == nullptr || !sock->recv_data(&rsp_size, sizeof(rsp_size)) || rsp_size != sizeof(response) ||
                !sock->recv_data(&response, sizeof(response)) || !response.ok) {
            GGML_LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = false;
        }
    }
    if (!ok) {
        return nullptr;
    }
    GGML_LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
                  ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    return new ggml_backend_rpc_comm_context{std::move(ranks)};
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return false;
    }
    const size_t n_ranks = comm_ctx->ranks.size();
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }
    for (size_t i = 0; i < n_ranks; i++) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            return false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    for (size_t i = 0; i < n_ranks; i++) {
        rpc_msg_comm_allreduce_req request;
        request.device = comm_ctx->ranks[i].device;
        request.tensor = serialize_tensor(tensors[i]);
        auto sock = get_socket(comm_ctx->ranks[i].endpoint);
        if (sock == nullptr || !send_rpc_cmd(sock, RPC_CMD_COMM_ALLREDUCE, &request, sizeof(request))) {
            return false;
        }
    }
    return true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_rpc_comm_init;
    }
    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_rpc_comm_free;
    }
    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .graph_uids  = */    {},
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
