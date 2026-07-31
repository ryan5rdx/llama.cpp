#include "transport.h"
#include "ggml-impl.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <poll.h>
#  include <unistd.h>
#endif
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <cerrno>
#  include <time.h>
#endif // GGML_RPC_RDMA

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

// Local RDMA device pinned via rpc_transport_set_rdma_device() (e.g. from the
// worker's --rdma-dev). Overrides the GGML_RDMA_DEV env var; empty = auto-select.
static std::mutex  g_rdma_device_mu;
static std::string g_rdma_device;   // guarded by g_rdma_device_mu

#ifdef GGML_RPC_RDMA
static constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
static constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
static constexpr int    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 x 256 KiB = 6 MiB

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd  = nullptr;
    struct ibv_cq * scq = nullptr;   // send completions
    struct ibv_cq * rcq = nullptr;   // recv completions
    struct ibv_qp * qp  = nullptr;

    void          * tx_buf = nullptr;
    struct ibv_mr * tx_mr  = nullptr;

    void          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    struct ibv_mr * rx_mr  = nullptr;
    int             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(int i) const {
        return static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    bool post_rx(int i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        return ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        if (tx_mr) ibv_dereg_mr(tx_mr);
        if (rx_mr) ibv_dereg_mr(rx_mr);
        free(tx_buf);
        free(rx_buf);
        if (qp)  ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
struct rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    int      gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;
};

struct rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

#ifdef GGML_RPC_RDMA_APPLE
// Apple RDMA-over-Thunderbolt: UC QP, two-sided IBV_WR_SEND only, no RDMA-CM.
// Per Apple TN3205 a SEND and its matching RECV must span the same number of
// Thunderbolt frames, so this is a fixed-STRIDE coalescing byte stream. Endpoints
// (GID/QPN/LID) are exchanged out of band via the TCP caps handshake; PSN is a
// fixed constant (UC does not use it for retransmit). A UC SEND with no matching
// posted recv is silently dropped, so the readiness handshake in update_caps
// ensures both peers have posted recvs before the first frame; thereafter the
// Thunderbolt link layer's credit-based flow control (TN3205) provides
// backpressure, so no application-level credit protocol is needed.
static constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
static constexpr int      RDMA_NBUF        = 16;          // ring depth (buffers per direction)
static constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame size
static constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // every SEND transfers a full STRIDE
static constexpr uint32_t RDMA_MAX_QP_WR   = 4095;        // TN3205 hardware limit, in frames
static constexpr uint32_t RDMA_PSN         = 0;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
static constexpr uint32_t RDMA_STRIDE_FRAMES = RDMA_STRIDE / RDMA_FRAME;
static constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
static constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
static constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

struct rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC
    uint32_t len;   // valid payload bytes (0 = padding)
};
static constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - sizeof(rdma_seg_hdr);

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd = nullptr;
    struct ibv_cq * cq = nullptr;     // single QP -> shared send/recv completions
    struct ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    struct ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    struct ibv_mr * recv_mr  = nullptr;

    int      nbuf = 0;                 // effective ring depth (self-tuned)
    int      send_busy[RDMA_NBUF] = {};  // posted send WQEs, per buffer (reaped -> 0)
    struct { int buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    int      inq_head = 0;
    int      inq_count = 0;
    int      pend_buf = -1;            // send buffer accumulating coalesced writes, or -1
    uint32_t pend_len = 0;
    bool     broken = false;
    uint8_t      port = 0;
    int          gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;

    bool post_recv(int i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        return ibv_post_recv(qp, &wr, &bad) == 0;
    }

    bool post_send(int i, size_t len) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        return ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        broken = true;
        if (qp) {
            struct ibv_qp_attr a = {};
            a.qp_state = IBV_QPS_ERR;
            ibv_modify_qp(qp, &a, IBV_QP_STATE);
            // drain CQ once so all WQEs flushed by ERR are reaped before
            // we dereg MRs and destroy the QP
            struct ibv_wc wc[RDMA_NBUF * 2];
            while (ibv_poll_cq(cq, RDMA_NBUF * 2, wc) > 0) {}
        }
        if (send_mr) ibv_dereg_mr(send_mr);
        if (recv_mr) ibv_dereg_mr(recv_mr);
        free(send_mem);
        free(recv_mem);
        if (qp)  ibv_destroy_qp(qp);
        if (cq)  ibv_destroy_cq(cq);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

struct rdma_local_info {
    uint32_t qpn = 0;
    uint16_t lid = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
};

struct rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA_APPLE

struct socket_t::impl {
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    bool flush();
    bool is_broken() const;
    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    bool rdma_probe();
    bool rdma_send(const void * data, size_t size);
    bool rdma_recv(void * data, size_t size);
    // GID-shaped target from this connection's local TCP address (getsockname);
    // used to auto-select the local device facing the peer on a multi-link host.
    std::optional<rdma_gid_t> rdma_build_target_gid();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
    std::string                conn_rdma_device;   // per-connection device override (highest priority)
#  ifdef GGML_RPC_RDMA_APPLE
    bool rdma_activate(uint32_t remote_qpn, uint16_t remote_lid, const uint8_t * remote_gid);
    int  rdma_progress();
    bool rdma_acquire_pending();
    bool rdma_post_pending();
    bool rdma_flush();
#  else
    bool tcp_peer_closed();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid);
    bool rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc);
#  endif
#endif // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
    if (fd != INVALID_SOCKET) closesocket(this->fd);
#else
    if (fd >= 0) close(this->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from this connection's local TCP
// address (getsockname). On a host with several RDMA links the local address of
// the socket to a given peer identifies the interface facing that peer, so the
// device whose GID equals this target is the one cabled to the peer. Handles
// IPv4, IPv4-mapped IPv6, and native IPv6 uniformly via a single memcmp.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        return std::nullopt;
    }
    rdma_gid_t target = {};
    if (addr.ss_family == AF_INET) {
        const auto * a = reinterpret_cast<const sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        return target;
    }
    if (addr.ss_family == AF_INET6) {
        const auto * a = reinterpret_cast<const sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        return target;
    }
    return std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

bool socket_t::impl::tcp_peer_closed() {
    if (fd < 0) return false;
#ifndef _WIN32
    struct pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    int r = poll(&pfd, 1, 0);
    return r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#else
    return false;
#endif
}

bool socket_t::impl::rdma_probe() {
    const char * dev_env = std::getenv("GGML_RDMA_DEV");
    const char * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    if (!target_gid) {
        return false;
    }

    const uint8_t ib_port = 1;
    int num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    if (!devs || num_devs == 0) return false;

    ibv_context * ibctx = nullptr;
    const char * matched_dev = nullptr;
    int gid_idx = gid_env ? atoi(gid_env) : -1;
    int gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    for (int d = 0; d < num_devs; d++) {
        const char * dn = ibv_get_device_name(devs[d]);
        if (dev_env && strcmp(dev_env, dn) != 0) continue;

        ibv_context * ctx = ibv_open_device(devs[d]);
        if (!ctx) continue;

        ibv_port_attr pa;
        if (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); continue; }

        int found_gid = gid_idx;
        int found_version = IBV_GID_TYPE_IB;
        if (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            int v2_idx = -1;
            int v1_idx = -1;
            for (int i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                if (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) continue;
                if (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) continue;
                if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } else if (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            if (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } else if (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } else {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            if (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        if (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            break;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    if (!ibctx) return false;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    if (!rdma->pd) return false;

    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, nullptr, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, nullptr, 0);
    if (!rdma->scq || !rdma->rcq) return false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    if (!rdma->qp) return false;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    if (!rdma->tx_buf || !rdma->rx_buf) return false;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!rdma->tx_mr || !rdma->rx_mr) return false;

    ibv_gid local_gid;
    if (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) return false;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    const char * ver_str = "";
    if (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } else if (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    GGML_LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    return true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid) {
    // RESET -> INIT
    {
        struct ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            return false;
        }
    }

    for (int i = 0; i < RDMA_RX_DEPTH; i++) {
        if (!rdma->post_rx(i)) return false;
    }

    // INIT -> RTR
    {
        struct ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            return false;
        }
    }

    // RTR -> RTS
    {
        struct ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            return false;
        }
    }

    GGML_LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    return true;
}

bool socket_t::impl::rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc) {
    for (uint64_t s = 0; ; s++) {
        int n = ibv_poll_cq(cq, 1, wc);
        if (n > 0) {
            if (wc->status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            return wc->status == IBV_WC_SUCCESS;
        }
        if (n < 0) return false;
        if ((s & 0xFFFFF) == 0 && s > 0) {
            if (tcp_peer_closed()) {
                return false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } else {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;
        struct ibv_wc wc;
        if (!rdma_poll(c->scq, &wc)) return false;

        src += chunk;
        rem -= chunk;
    }
    return true;
}

bool socket_t::impl::rdma_recv(void * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        struct ibv_wc wc;
        if (!rdma_poll(c->rcq, &wc)) return false;

        int slot = (int)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        if (!c->post_rx(slot)) return false;

        dst += got;
        rem -= got;
    }
    return true;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#ifdef GGML_RPC_RDMA_APPLE

static bool rdma_gid_is_zero(const union ibv_gid * g) {
    for (size_t i = 0; i < RDMA_GID_SIZE; i++) if (g->raw[i]) return false;
    return true;
}

// RoCEv2 IPv4-mapped GID (::ffff:a.b.c.d) - the Thunderbolt link-local address.
static bool rdma_gid_is_ipv4(const union ibv_gid * g) {
    for (int i = 0; i < 10; i++) if (g->raw[i]) return false;
    return g->raw[10] == 0xff && g->raw[11] == 0xff;
}

// Prefer a RoCEv2 IPv4-mapped GID, else the first non-zero GID. -1 if none.
static int rdma_select_gid(struct ibv_context * ctx, uint8_t port, int gid_tbl_len, union ibv_gid * out) {
    int fallback = -1;
    for (int i = 0; i < gid_tbl_len; i++) {
        union ibv_gid g;
        if (ibv_query_gid(ctx, port, i, &g) != 0) continue;
        if (rdma_gid_is_zero(&g)) continue;
        if (rdma_gid_is_ipv4(&g)) { if (out) *out = g; return i; }
        if (fallback < 0) { fallback = i; if (out) *out = g; }
    }
    return fallback;
}

// First active port; only cabled+up Thunderbolt links are ACTIVE, so device
// list[0] is often a down port. Returns 0 if none.
static uint8_t rdma_first_active_port(struct ibv_context * ctx, struct ibv_port_attr * out) {
    struct ibv_device_attr da;
    if (ibv_query_device(ctx, &da) != 0) return 0;
    for (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        struct ibv_port_attr pa;
        if (ibv_query_port(ctx, p, &pa) != 0) continue;
        if (pa.state == IBV_PORT_ACTIVE) { if (out) *out = pa; return p; }
    }
    return 0;
}

// Phase 1: pick the local device facing this peer, create a UC QP, register the
// STRIDE rings, and capture the local endpoint. Selection priority:
//   1. explicit name (per-connection map / --rdma-dev / GGML_RDMA_DEV);
//   2. else the device whose GID matches this connection's local TCP address
//      (auto per-peer selection on a multi-link host - RDMA is point-to-point,
//      so the wrong local device makes RTR fail with no path);
//   3. else the first device with an active port and usable GID.
bool socket_t::impl::rdma_probe() {
    std::string global_dev;
    {
        std::lock_guard<std::mutex> lock(g_rdma_device_mu);
        global_dev = g_rdma_device;
    }
    const char * dev = !conn_rdma_device.empty() ? conn_rdma_device.c_str()
                     : !global_dev.empty()       ? global_dev.c_str()
                     :                             std::getenv("GGML_RDMA_DEV");
    const std::optional<rdma_gid_t> target = rdma_build_target_gid();

    int ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    if (!devs || ndev <= 0) { if (devs) ibv_free_device_list(devs); return false; }

    ibv_context * ctx = nullptr;    // currently-chosen device (kept open)
    uint8_t port = 0;
    struct ibv_port_attr pa = {};
    union ibv_gid gid = {};
    int gid_idx = -1;
    const char * matched = "";
    bool gid_matched = false;       // chosen device's GID matches the peer-facing local address
    for (int d = 0; d < ndev; d++) {
        const char * name = ibv_get_device_name(devs[d]);
        if (dev && dev[0] && (!name || strcmp(name, dev) != 0)) continue;
        ibv_context * c = ibv_open_device(devs[d]);
        if (!c) continue;
        struct ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        if (!pt) { ibv_close_device(c); continue; }
        union ibv_gid g = {};
        int gi = rdma_select_gid(c, pt, p.gid_tbl_len, &g);
        if (gi < 0) { ibv_close_device(c); continue; }

        const bool this_match = target.has_value() &&
            memcmp(g.raw, target->data(), RDMA_GID_SIZE) == 0;
        if (!ctx) {                         // first usable candidate (fallback)
            ctx = c; port = pt; pa = p; gid = g; gid_idx = gi; matched = name ? name : ""; gid_matched = this_match;
        } else if (this_match && !gid_matched) {   // peer-facing device: replace fallback
            ibv_close_device(ctx);
            ctx = c; port = pt; pa = p; gid = g; gid_idx = gi; matched = name ? name : ""; gid_matched = true;
        } else {
            ibv_close_device(c);
        }
        if (gid_matched) break;             // the device facing the peer wins outright
    }
    ibv_free_device_list(devs);
    if (!ctx) return false;
    if (target.has_value() && !gid_matched && (!dev || !dev[0])) {
        GGML_LOG_INFO("RDMA(Apple/UC) no device GID matched the local address; using first active (%s). "
                      "On a multi-link host set --rdma-dev / GGML_RDMA_DEV to the device facing this peer.\n", matched);
    }

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ctx;
    rdma->port = port;
    rdma->gid_idx = gid_idx;
    rdma->path_mtu = pa.active_mtu;

    rdma->pd = ibv_alloc_pd(ctx);
    if (!rdma->pd) return false;

    // TN3205: queue depth is counted in 4 KiB frames, not work requests, so an
    // RDMA_NBUF-deep ring of STRIDE-sized messages needs NBUF * STRIDE_FRAMES.
    uint32_t max_wr = RDMA_MAX_QP_WR;
    struct ibv_device_attr da = {};
    if (ibv_query_device(ctx, &da) == 0 && da.max_qp_wr > 0) {
        max_wr = (uint32_t)da.max_qp_wr;
    }
    uint32_t want_frames = (uint32_t)RDMA_NBUF * RDMA_STRIDE_FRAMES;
    if (want_frames > max_wr) want_frames = max_wr;

    int cqe = (int)(2 * want_frames + 1);
    if (da.max_cqe > 0 && cqe > da.max_cqe) cqe = da.max_cqe;
    rdma->cq = ibv_create_cq(ctx, cqe, nullptr, nullptr, 0);
    if (!rdma->cq) return false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->cq;
    qia.recv_cq = rdma->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = want_frames;
    qia.cap.max_recv_wr  = want_frames;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    if (!rdma->qp) {
        // a provider that refuses a depth it cannot honour still works at one frame
        // per message; fall back so the connection degrades instead of dropping.
        qia.cap.max_send_wr = qia.cap.max_recv_wr = RDMA_STRIDE_FRAMES;
        rdma->qp = ibv_create_qp(rdma->pd, &qia);
    }
    if (!rdma->qp) return false;

    // TN3205: the system may adjust the requested depth, so size the ring from
    // what was actually granted rather than what was asked for.
    uint32_t got_frames = qia.cap.max_send_wr;
    {
        ibv_qp_attr       qa      = {};
        ibv_qp_init_attr  granted = {};
        if (ibv_query_qp(rdma->qp, &qa, IBV_QP_CAP, &granted) == 0) {
            got_frames = granted.cap.max_send_wr < granted.cap.max_recv_wr
                       ? granted.cap.max_send_wr : granted.cap.max_recv_wr;
        }
    }
    rdma->nbuf = (int)(got_frames / RDMA_STRIDE_FRAMES);
    if (rdma->nbuf > RDMA_NBUF) rdma->nbuf = RDMA_NBUF;
    if (rdma->nbuf < 1)         rdma->nbuf = 1;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            return false;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    const size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    if (posix_memalign((void **)&rdma->send_mem, (size_t)page, ring_bytes) != 0) rdma->send_mem = nullptr;
    if (posix_memalign((void **)&rdma->recv_mem, (size_t)page, ring_bytes) != 0) rdma->recv_mem = nullptr;
    if (!rdma->send_mem || !rdma->recv_mem) return false;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    const int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    rdma->send_mr = ibv_reg_mr(rdma->pd, rdma->send_mem, ring_bytes, mr_flags);
    rdma->recv_mr = ibv_reg_mr(rdma->pd, rdma->recv_mem, ring_bytes, mr_flags);
    if (!rdma->send_mr || !rdma->recv_mr) return false;

    // Recvs are posted after RTS (in rdma_activate), not here: Apple's provider
    // only accepts them once the QP is ready to receive.

    rdma_local = {};
    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.lid = pa.lid;
    memcpy(rdma_local.gid, gid.raw, RDMA_GID_SIZE);

    GGML_LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d select=%s\n",
                  matched, port, gid_idx, rdma_local.qpn, (unsigned)pa.lid, 128 << rdma->path_mtu,
                  (dev && dev[0]) ? "pinned" : (gid_matched ? "gid-match" : "first-active"));
    GGML_LOG_INFO("RDMA(Apple/UC) queue: frames req=%u granted=%u (%u/msg) -> ring=%d x %zu KiB\n",
                  want_frames, got_frames, RDMA_STRIDE_FRAMES, rdma->nbuf, RDMA_STRIDE / 1024);
    return true;
}

// Phase 2: given the remote endpoint, INIT -> RTR -> RTS (UC: GID/GRH addressing,
// no timeout/retry/rnr/rd_atomic).
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint16_t remote_lid, const uint8_t * remote_gid) {
    rdma_conn * c = rdma.get();
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_num = remote_qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_num      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = remote_lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        if (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            GGML_LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            return false;
        }
    }
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        if (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            GGML_LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            return false;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    // The ring depth was already fixed from the granted frame budget in rdma_probe.
    for (int i = 0; i < c->nbuf; i++) {
        if (!c->post_recv(i)) {
            GGML_LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, c->nbuf);
            return false;
        }
    }

    GGML_LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << c->path_mtu, c->nbuf);
    return true;
}

// Drain the CQ once: free completed sends, enqueue completed recv DATA segments.
// Returns count, or -1 on error.
int socket_t::impl::rdma_progress() {
    rdma_conn * c = rdma.get();
    struct ibv_wc wc[RDMA_NBUF * 2];
    int n = ibv_poll_cq(c->cq, RDMA_NBUF * 2, wc);
    if (n < 0) { GGML_LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); c->broken = true; return -1; }
    for (int j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        bool is_recv = (id & RDMA_RECV_WR) != 0;
        if (wc[j].status != IBV_WC_SUCCESS) {
            GGML_LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            c->broken = true;
            return -1;
        }
        if (is_recv) {
            int b = (int)(id & RDMA_WR_IDX_MASK);
            const rdma_seg_hdr * h = (const rdma_seg_hdr *)(c->recv_mem + (size_t)b * RDMA_STRIDE);
            if (h->magic != RDMA_SEG_MAGIC) { GGML_LOG_ERROR("RDMA(Apple/UC) bad segment magic\n"); c->broken = true; return -1; }
            if (h->len > RDMA_PAYLOAD) { GGML_LOG_ERROR("RDMA(Apple/UC) segment len %u exceeds payload\n", h->len); c->broken = true; return -1; }
            int slot = (c->inq_head + c->inq_count) % c->nbuf;
            c->inq[slot].buf  = b;
            c->inq[slot].off  = 0;
            c->inq[slot].len  = h->len;
            c->inq_count++;
        } else {
            c->send_busy[(int)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    return n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
bool socket_t::impl::rdma_acquire_pending() {
    rdma_conn * c = rdma.get();
    if (c->pend_buf >= 0) return true;
    for (;;) {
        if (c->broken) return false;
        for (int k = 0; k < c->nbuf; k++) if (!c->send_busy[k]) { c->pend_buf = k; c->pend_len = 0; return true; }
        if (rdma_progress() < 0) return false;
    }
}

// Post the coalesced pending frame (padded to a full STRIDE), consuming a credit.
bool socket_t::impl::rdma_post_pending() {
    rdma_conn * c = rdma.get();
    if (c->pend_buf < 0) return true;
    int i = c->pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(c->send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = c->pend_len;
    if (!c->post_send(i, RDMA_STRIDE)) { c->broken = true; return false; }
    c->send_busy[i] = 1;
    c->pend_buf = -1;
    c->pend_len = 0;
    return true;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * p = (const uint8_t *)data;
    while (size > 0) {
        if (c->broken) return false;
        if (!rdma_acquire_pending()) return false;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + sizeof(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        if (c->pend_len == RDMA_PAYLOAD) { if (!rdma_post_pending()) return false; }
    }
    return true;
}

bool socket_t::impl::rdma_recv(void * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * p = (uint8_t *)data;
    if (!rdma_post_pending()) return false;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    while (size > 0) {
        if (c->inq_count == 0) {
            if (c->broken) return false;
            int n = rdma_progress();
            if (n < 0) return false;
            if (n == 0) {
                // UC never signals a disconnect; the bootstrap TCP fd is the
                // liveness anchor. A peer process exit sends a FIN that shows up
                // as readable (POLLIN) on macOS. Match Linux rdma_poll: check
                // every ~1M idle iterations (0x100000).
                if ((++idle & 0xFFFFF) == 0 && idle > 1) {
                    struct pollfd pfd = { fd, POLLIN, 0 };
                    if (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        return false;
                    }
                }
            } else {
                idle = 0;
            }
            continue;
        }
        idle = 0;
        int slot = c->inq_head;
        int b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + sizeof(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        if (c->inq[slot].off == c->inq[slot].len) {
            if (!c->post_recv(b)) { c->broken = true; return false; }
            c->inq_head = (c->inq_head + 1) % c->nbuf;
            c->inq_count--;
        }
    }
    return true;
}

bool socket_t::impl::rdma_flush() {
    return rdma ? rdma_post_pending() : true;
}

#endif // GGML_RPC_RDMA_APPLE

#endif // GGML_RPC_RDMA

bool socket_t::impl::send_data(const void * data, size_t size) {
#ifdef GGML_RPC_RDMA
    if (use_rdma) {
        return rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            GGML_LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

bool socket_t::impl::recv_data(void * data, size_t size) {
#ifdef GGML_RPC_RDMA
    if (use_rdma) {
        return rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (char *)data + bytes_recv, size_to_recv, 0);
        if (n < 0) {
            GGML_LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            return false;
        }
        if (n == 0) {
            LOG_DBG("recv returned 0 (peer closed?)\n");
            return false;
        }
        bytes_recv += (size_t)n;
    }
    return true;
}

void socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    rdma_local = {};
    if (!std::getenv("GGML_RPC_NO_RDMA") && rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
#  ifdef GGML_RPC_RDMA_APPLE
        rc.lid = rdma_local.lid;
#  else
        rc.psn = rdma_local.psn;
#  endif
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, sizeof(rc));
    } else {
        rdma.reset();
    }
#endif // GGML_RPC_RDMA
}

void socket_t::impl::update_caps(const uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    if (!rdma) {
        return;
    }
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, sizeof(rc));
    if (rc.qpn == 0) {
        rdma.reset();
        return;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    bool activated = rdma_activate(rc.qpn, rc.lid, rc.gid);
    // Mutual readiness handshake over TCP (use_rdma is still false here, so this
    // runs on TCP). Both peers advertised RDMA, so each sends a 1-byte status and
    // RDMA is enabled only if BOTH sides activated: a UC SEND with no matching
    // posted recv is silently dropped (TN3205) and rdma_activate can fail on one
    // side only, so this prevents a one-sided upgrade and guarantees both peers
    // have posted recvs before the first UC frame.
    uint8_t local_ready = activated ? RDMA_SYNC_READY : 0;
    uint8_t peer_ready  = 0;
    if (!send_data(&local_ready, sizeof(local_ready)) ||
        !recv_data(&peer_ready, sizeof(peer_ready))) {
        rdma.reset();
        return;
    }
    if (activated && peer_ready == RDMA_SYNC_READY) {
        use_rdma = true;
    } else {
        GGML_LOG_ERROR("RDMA activation not mutual, staying on TCP\n");
        rdma.reset();
    }
#  else
    if (rdma_activate(rc.qpn, rc.psn, rc.gid)) {
        use_rdma = true;
    } else {
        GGML_LOG_ERROR("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#  endif
#else
    (void)remote_caps;
#endif // GGML_RPC_RDMA
}

bool socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma_flush();
    }
#endif
    return true;
}

bool socket_t::impl::is_broken() const {
#ifdef GGML_RPC_RDMA_APPLE
    return use_rdma && rdma && rdma->broken;
#else
    return false;
#endif
}


/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

bool socket_t::flush() {
    return pimpl->flush();
}

bool socket_t::is_rdma() const {
    return pimpl->use_rdma;
}

bool socket_t::is_broken() const {
    return pimpl->is_broken();
}

void socket_t::set_rdma_device(const char * name) {
#ifdef GGML_RPC_RDMA
    pimpl->conn_rdma_device = (name && name[0]) ? name : "";
#else
    (void)name;
#endif
}

void socket_t::get_caps(uint8_t * local_caps) {
    return pimpl->get_caps(local_caps);
}

void socket_t::update_caps(const uint8_t * remote_caps) {
    return pimpl->update_caps(remote_caps);
}

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    return sockfd != INVALID_SOCKET;
#else
    return sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, NULL, NULL);
    if (!is_valid_fd(client_socket_fd)) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        GGML_LOG_ERROR("Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        GGML_LOG_ERROR("Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        GGML_LOG_ERROR("Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (::connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

void rpc_transport_set_rdma_device(const char * name) {
    std::lock_guard<std::mutex> lock(g_rdma_device_mu);
    g_rdma_device = (name && name[0]) ? name : "";
}

#ifdef _WIN32
static std::mutex g_rpc_transport_mu;
static bool g_rpc_transport_wsa_started = false;
#endif

bool rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (g_rpc_transport_wsa_started) {
        return true;
    }
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        return false;
    }
    g_rpc_transport_wsa_started = true;
    return true;
#else
    return true;
#endif
}

void rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (!g_rpc_transport_wsa_started) {
        return;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = false;
#endif
}
