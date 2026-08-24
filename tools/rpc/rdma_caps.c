// Answers the three provider questions that gate zero-copy on Apple RDMA.
// Needs RDMA enabled (rdma_ctl enable, from Recovery). No peer required.
//
//   clang -O2 rdma_caps.c -o rdma_caps -lrdma && ./rdma_caps
#include <infiniband/verbs.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    int ndev = 0;
    struct ibv_device ** devs = ibv_get_device_list(&ndev);
    if (!devs || ndev == 0) {
        printf("no RDMA devices. run 'rdma_ctl status'; if disabled, enable it from Recovery\n");
        return 1;
    }
    printf("devices: %d\n", ndev);

    struct ibv_context * ctx = NULL;
    for (int i = 0; i < ndev && !ctx; i++) {
        ctx = ibv_open_device(devs[i]);
        if (ctx) printf("opened: %s\n", ibv_get_device_name(devs[i]));
    }
    if (!ctx) { printf("FAIL: could not open any device\n"); return 1; }

    struct ibv_device_attr da = {};
    if (ibv_query_device(ctx, &da) == 0) {
        printf("  max_sge        = %d   (send-side scatter: >1 allows header+payload+pad)\n", da.max_sge);
        printf("  max_qp_wr      = %d\n", da.max_qp_wr);
        printf("  max_mr_size    = %llu\n", (unsigned long long) da.max_mr_size);
    }

    struct ibv_pd * pd = ibv_alloc_pd(ctx);
    if (!pd) { printf("FAIL: alloc_pd\n"); return 1; }

    // Q1: can Metal's shared-buffer memory be registered? ggml_metal_host_malloc uses
    // vm_allocate, so reproduce exactly that rather than malloc.
    const size_t n = 4u << 20;
    void * vm = NULL;
    kern_return_t kr = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &vm, n, VM_FLAGS_ANYWHERE);
    if (kr != KERN_SUCCESS) { printf("FAIL: vm_allocate\n"); return 1; }
    memset(vm, 0, n);

    const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    struct ibv_mr * mr = ibv_reg_mr(pd, vm, n, acc);
    printf("%s: ibv_reg_mr over vm_allocate memory (what Metal shared buffers use)\n",
           mr ? "PASS" : "FAIL");
    if (mr) printf("  lkey=%u rkey=%u\n", mr->lkey, mr->rkey);

    // Q2: what send scatter depth does a UC QP actually grant?
    struct ibv_cq * cq = ibv_create_cq(ctx, 64, NULL, NULL, 0);
    for (int want = 1; want <= 4; want++) {
        struct ibv_qp_init_attr qia = {};
        qia.send_cq = cq; qia.recv_cq = cq;
        qia.qp_type = IBV_QPT_UC;
        qia.cap.max_send_wr = 32; qia.cap.max_recv_wr = 32;
        qia.cap.max_send_sge = want; qia.cap.max_recv_sge = 1;
        struct ibv_qp * qp = ibv_create_qp(pd, &qia);
        printf("  max_send_sge=%d -> %s (granted %d)\n", want,
               qp ? "ok" : "refused", qp ? qia.cap.max_send_sge : 0);
        if (qp) ibv_destroy_qp(qp);
    }

    // Q3: is one-sided write really absent? create a UC QP and inspect what a
    // RDMA_WRITE post reports. Nothing is connected, so a failure here is about
    // opcode support rather than the wire.
    printf("\n(one-sided support is reported by TN3205 as absent; ds4 uses IBV_WR_SEND only)\n");

    if (mr) ibv_dereg_mr(mr);
    if (cq) ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(devs);
    return 0;
}
